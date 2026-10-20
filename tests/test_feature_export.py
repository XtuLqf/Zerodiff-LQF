"""Exercise the export contract against this repository's actual training code.

Only synthetic MAT files and random weights are used; no training is started.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import torch

import zerodiff_tools
from datasets.image_util import DATA_LOADER
from feature_export import SCHEMA_VERSION
from feature_export.data import array_sha256, file_sha256, load_awa2
from feature_export.export import canonical_sha256, main, make_protocol, parse_args, validate_protocol
from feature_export.runtime import DEFAULTS, FeatureRuntime, generate_features, load_options


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def awa2_root(tmp_path_factory):
    dataroot = tmp_path_factory.mktemp("data")
    root = dataroot / "AWA2"
    root.mkdir()
    # Non-consecutive unseen IDs detect accidental remapping to 0..9.
    labels = np.repeat(np.arange(50), 6)
    unseen = labels % 5 == 0
    train = np.flatnonzero(~unseen & (np.arange(labels.size) % 6 < 4))
    seen_test = np.flatnonzero(~unseen & (np.arange(labels.size) % 6 >= 4))
    # Reverse the split to verify that export preserves its original order.
    unseen_test = np.flatnonzero(unseen)[::-1]
    rng = np.random.default_rng(902)
    features = rng.normal(size=(labels.size, 2048)) * 0.3 + np.arange(labels.size)[:, None] / 100
    attributes = rng.uniform(0.1, 10, size=(85, 50))
    names = np.asarray([f"class_{index:02d}" for index in range(50)], dtype=object)[:, None]
    sio.savemat(root / "res101.mat", {"labels": labels[:, None] + 1,
                                      "features": np.zeros((2048, labels.size))})
    sio.savemat(root / "ce_ce.mat", {"features": features})
    sio.savemat(root / "con_paco.mat", {"features": features * 0.7 + 0.2})
    sio.savemat(root / "att_splits.mat", {
        "trainval_loc": train[:, None] + 1, "train_loc": train[:, None] + 1,
        "val_loc": unseen_test[:, None] + 1, "test_seen_loc": seen_test[:, None] + 1,
        "test_unseen_loc": unseen_test[:, None] + 1,
        "original_att": attributes, "att": attributes / 100, "allclasses_names": names,
    })
    for percent in (10, 30):
        # These indices address the trainval subset, as the original loader expects.
        subset = np.arange(max(2, train.size * percent // 100)) + 1
        sio.savemat(root / f"split_{percent}percent.mat", {"trainval_loc": subset[:, None]})
    return dataroot


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    training = {**DEFAULTS, "decoder_layer_sizes": [16, 2048], "split_percent": 100}
    config = root / "config.json"
    config.write_text(json.dumps({"training": training}), encoding="utf-8")
    options, _ = load_options(100, config)
    torch.manual_seed(521)
    generator = zerodiff_tools.DFG_Generator(options)
    drg = zerodiff_tools.DRG_Generator(options)
    dfg_path, drg_path = root / "DFG.tar", root / "DRG.tar"
    torch.save({"state_dict_G": generator.state_dict(), "state_dict_Dec": {}}, dfg_path)
    # The current DRG trainer writes both aliases.
    torch.save({"state_dict_R": drg.state_dict(), "state_dict_G_con": drg.state_dict()}, drg_path)
    return SimpleNamespace(root=root, config=config, options=options, training=training,
                           dfg=dfg_path, drg=drg_path)


@pytest.fixture(scope="module")
def runtime(model_files):
    return FeatureRuntime(model_files.options, model_files.dfg, model_files.drg, "cpu")


@pytest.mark.parametrize("percent", [10, 30, 100])
@pytest.mark.parametrize("standardization", [False, True])
@pytest.mark.parametrize("normalize_attributes", [False, True])
def test_real_features_match_training_loader(awa2_root, percent, standardization, normalize_attributes):
    options, _ = load_options(percent)
    options.dataroot = str(awa2_root)
    options.standardization = standardization
    options.class_embedding_norm = normalize_attributes
    original = DATA_LOADER(options)
    exported = load_awa2(awa2_root, options)
    np.testing.assert_array_equal(exported.real_features, original.test_unseen_feature.numpy())
    np.testing.assert_array_equal(exported.real_labels, original.test_unseen_label.numpy())
    np.testing.assert_array_equal(exported.class_ids, original.unseenclasses.numpy())
    np.testing.assert_array_equal(exported.attributes.numpy(), original.attribute.numpy())
    assert exported.class_names.tolist() == [f"class_{index:02d}" for index in range(0, 50, 5)]
    splits = sio.loadmat(awa2_root / "AWA2" / "att_splits.mat")
    np.testing.assert_array_equal(exported.real_indices, splits["test_unseen_loc"].reshape(-1) - 1)


def original_sampler(runtime):
    """Compile only the real methods; importing the training script starts training."""
    path = ROOT / "zerodiff_DFG_train.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ZERODIFF")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("sample_from_model", "sample_posterior")]
    assert len(methods) == 2
    namespace = {"torch": torch, "zerodiff_tools": zerodiff_tools}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    options = runtime.options
    original = SimpleNamespace(
        netG=runtime.netG, netR=runtime.netR, n_T=options.n_T,
        dim_v=options.resSize, dim_noise=options.noiseSize,
        posterior_coefficients=runtime.posterior, device="cpu",
    )
    for name in ("sample_from_model", "sample_posterior"):
        setattr(original, name, MethodType(namespace[name], original))
    return original


@pytest.mark.parametrize("sampling", ["one-step", "progressive"])
@pytest.mark.parametrize("count", [1, 5])
def test_sampling_matches_actual_training_methods(runtime, awa2_root, sampling, count):
    data = load_awa2(awa2_root, runtime.options)
    attributes = data.attributes[data.class_ids[0]].repeat(count, 1)
    original = original_sampler(runtime)
    torch.manual_seed(9182)
    expected, _ = original.sample_from_model(attributes, progressive=sampling == "progressive")
    rng = torch.Generator(device="cpu").manual_seed(9182)
    actual = runtime.sample(attributes, sampling, rng, rng)
    assert torch.equal(actual, expected)
    assert not runtime.netG.training and not runtime.netR.training
    assert not any(parameter.requires_grad for parameter in runtime.netG.parameters())


@pytest.mark.parametrize("sampling", ["one-step", "progressive"])
def test_generation_reproducible_with_tail_batch(runtime, awa2_root, sampling):
    data = load_awa2(awa2_root, runtime.options)
    first, labels = generate_features(runtime, data, 5, 9182, 3, sampling)
    # Model initialization/global RNG activity must not change local class streams.
    torch.randn(100)
    second, second_labels = generate_features(runtime, data, 5, 9182, 3, sampling)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(labels, np.repeat(data.class_ids, 5))
    np.testing.assert_array_equal(labels, second_labels)
    assert first.dtype == np.float32 and first.shape == (50, 2048)
    changed, _ = generate_features(runtime, data, 5, 9183, 3, sampling)
    assert not np.array_equal(first, changed)


@pytest.mark.parametrize("drg_key", ["state_dict_R", "state_dict_G_con", "fallback"])
@pytest.mark.parametrize("relation", [False, True])
def test_checkpoint_variants(model_files, tmp_path, drg_key, relation):
    drg_state = torch.load(model_files.drg, weights_only=True)["state_dict_R"]
    drg = ({"state_dict_G_con": None, "state_dict_R": drg_state} if drg_key == "fallback"
           else {drg_key: drg_state})
    drg_path, dfg_path = tmp_path / "DRG.tar", tmp_path / "DFG.tar"
    dfg = torch.load(model_files.dfg, weights_only=True)
    if relation:
        dfg.update(state_dict_VSRA={}, method_metadata={"method": "Rediff"})
    torch.save(drg, drg_path)
    torch.save(dfg, dfg_path)
    loaded = FeatureRuntime(model_files.options, dfg_path, drg_path, "cpu")
    assert loaded.has_relation_state == relation
    assert loaded.method_metadata == ({"method": "Rediff"} if relation else None)


@pytest.mark.parametrize("role", ["DFG", "DRG"])
@pytest.mark.parametrize("broken", ["missing", "shape"])
def test_incompatible_checkpoints_are_rejected(model_files, tmp_path, role, broken):
    checkpoint = {}
    if broken == "shape":
        options = deepcopy(model_files.options)
        options.decoder_layer_sizes = [17, 2048]
        generator = (zerodiff_tools.DFG_Generator(options) if role == "DFG"
                     else zerodiff_tools.DRG_Generator(options))
        checkpoint["state_dict_G" if role == "DFG" else "state_dict_R"] = generator.state_dict()
    bad = tmp_path / "bad.tar"
    torch.save(checkpoint, bad)
    with pytest.raises(ValueError, match=role + " checkpoint"):
        FeatureRuntime(model_files.options, bad if role == "DFG" else model_files.dfg,
                       bad if role == "DRG" else model_files.drg, "cpu")


def export_arguments(model_files, awa2_root, output, view="both", config=None):
    return ["--dfg-checkpoint", str(model_files.dfg), "--drg-checkpoint", str(model_files.drg),
            "--training-config", str(config or model_files.config), "--dataroot", str(awa2_root),
            "--method", "ZeroDiff", "--split-percent", "100", "--sampling", "one-step",
            "--samples-per-class", "5", "--batch-size", "3", "--seed", "9182", "--view", view,
            "--device", "cpu", "--output-dir", str(output)]


@pytest.mark.parametrize("view", ["generated-only", "real-vs-generated", "both"])
def test_export_contract(model_files, awa2_root, tmp_path, view):
    output = tmp_path / "export"
    arguments = export_arguments(model_files, awa2_root, output, view)
    main(arguments)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == canonical_sha256(protocol)
    assert manifest["method"] == "ZeroDiff" and manifest["split_percent"] == 100
    assert manifest["training_config_defaulted_fields"] == []
    assert manifest["model_source_sha256"]["zerodiff_tools.py"] == file_sha256(ROOT / "zerodiff_tools.py")
    expected_names = ({"generated_only.npz", "real_vs_generated.npz"} if view == "both" else
                      {"generated_only.npz" if view == "generated-only" else "real_vs_generated.npz"})
    assert {path.name for path in output.glob("*.npz")} == expected_names
    data = load_awa2(awa2_root, model_files.options)
    generated = {}
    for name in expected_names:
        assert manifest["artifacts"][name]["sha256"] == file_sha256(output / name)
        with np.load(output / name, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"features", "labels", "source", "sample_ids", "dataset_indices",
                                         "class_ids", "class_names", "schema_version"}
            for key in arrays.files:
                assert array_sha256(arrays[key]) == manifest["artifacts"][name]["array_sha256"][key]
            assert arrays["schema_version"].item() == SCHEMA_VERSION
            assert arrays["features"].dtype == np.float32
            assert arrays["labels"].dtype == arrays["dataset_indices"].dtype == np.int64
            assert arrays["source"].dtype == np.uint8
            assert arrays["sample_ids"].dtype.kind == arrays["class_names"].dtype.kind == "U"
            fake = arrays["source"] == 1
            np.testing.assert_array_equal(arrays["labels"][fake], np.repeat(data.class_ids, 5))
            assert np.all(arrays["dataset_indices"][fake] == -1)
            assert arrays["sample_ids"][fake].tolist() == [
                f"generated:{class_id}:{index}" for class_id in data.class_ids for index in range(5)]
            assert manifest["artifacts"][name]["generated_rows"] == 50
            generated[name] = {key: arrays[key][fake] for key in
                               ("features", "labels", "sample_ids", "dataset_indices")}
            if name == "real_vs_generated.npz":
                real = arrays["source"] == 0
                assert arrays["source"].tolist() == [0] * data.real_labels.size + [1] * 50
                np.testing.assert_array_equal(arrays["features"][real], data.real_features)
                np.testing.assert_array_equal(arrays["labels"][real], data.real_labels)
                np.testing.assert_array_equal(arrays["dataset_indices"][real], data.real_indices)
                assert arrays["sample_ids"][real].tolist() == [f"real:{i}" for i in data.real_indices]
    if view == "both":
        for key in generated["generated_only.npz"]:
            np.testing.assert_array_equal(generated["generated_only.npz"][key], generated["real_vs_generated.npz"][key])
    with pytest.raises(FileExistsError):
        main(arguments)


def test_cross_method_protocol_and_repeat_export(model_files, awa2_root, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    main(export_arguments(model_files, awa2_root, first))
    arguments = export_arguments(model_files, awa2_root, second)
    arguments[arguments.index("--method") + 1] = "Rediff"
    arguments += ["--protocol", str(first / "protocol.json")]
    main(arguments)
    for name in ("generated_only.npz", "real_vs_generated.npz"):
        with np.load(first / name, allow_pickle=False) as a, np.load(second / name, allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    bad = tmp_path / "bad"
    arguments[arguments.index("--output-dir") + 1] = str(bad)
    arguments[arguments.index("--batch-size") + 1] = "2"
    with pytest.raises(ValueError, match="protocol.generation.batch_size"):
        main(arguments)
    assert not bad.exists()


def test_protocol_shared_across_training_percentages(awa2_root):
    args = SimpleNamespace(samples_per_class=100, sampling="one-step", seed=9182, batch_size=100)
    protocols = []
    for percent in (10, 30, 100):
        options, _ = load_options(percent)
        data = load_awa2(awa2_root, options)
        protocols.append(make_protocol(data, args))
        if percent != 100:
            assert f"split_{percent}percent.mat" in data.files
    validate_protocol(protocols[0], protocols[1])
    validate_protocol(protocols[0], protocols[2])
    changed = deepcopy(protocols[0])
    changed["alignment"]["real_features_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol.alignment.real_features_sha256"):
        validate_protocol(protocols[0], changed)


@pytest.mark.parametrize("document, message", [
    ([], "JSON object"), ({"training": []}, "training.*object"),
    ({"split_percent": 30}, "split-percent"), ({"split_percent": 100.0}, "split-percent"),
    ({"preprocessing": "false"}, "JSON boolean"), ({"class_embedding_norm": 1}, "JSON boolean"),
    ({"n_T": True}, "dimensions"), ({"n_T": 0}, "dimensions"),
    ({"ddpmbeta1": "0.1"}, "betas"), ({"ddpmbeta2": float("nan")}, "betas"),
    ({"decoder_layer_sizes": None}, "two positive integers"),
    ({"decoder_layer_sizes": [True, 2048]}, "two positive integers"),
    ({"drg_sha256": ""}, "hex digest"), ({"validation": True}, "test_unseen"),
    ({"dataset": "CUB"}, "AWA2"), ({"resSize": 2048.0}, "V=2048"),
])
def test_invalid_config_rejected(tmp_path, document, message):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_options(100, config)


@pytest.mark.parametrize("nested", [False, True])
def test_flat_and_nested_config_with_bom(tmp_path, nested):
    training = {"decoder_layer_sizes": [16, 8192], "split_percent": 30}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": training} if nested else training), encoding="utf-8-sig")
    options, loaded = load_options(30, config)
    assert loaded == training
    assert options.decoder_layer_sizes == [16, 2048]


def test_config_drg_hash_and_autodetection(model_files, awa2_root, tmp_path):
    config = tmp_path / "paired.json"
    training = {**model_files.training, "drg_sha256": "0" * 64}
    config.write_text(json.dumps(training), encoding="utf-8")
    output = tmp_path / "bad_pair"
    with pytest.raises(ValueError, match="DRG checkpoint hash"):
        main(export_arguments(model_files, awa2_root, output, config=config))
    assert not output.exists()
    training["drg_sha256"] = file_sha256(model_files.drg)
    config.write_text(json.dumps(training), encoding="utf-8")
    main(export_arguments(model_files, awa2_root, tmp_path / "paired", config=config))
    arguments = export_arguments(model_files, awa2_root, tmp_path / "auto_config")
    offset = arguments.index("--training-config")
    del arguments[offset:offset + 2]
    main(arguments)
    manifest = json.loads((tmp_path / "auto_config" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_config"]["sha256"] == file_sha256(model_files.config)


def test_legacy_defaults_match_awa2_training_launcher():
    # Verify the fallback against this repository, without executing its launcher.
    tree = ast.parse((ROOT / "scripts/run_awa2_zerodiff_DFG_train.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
    constants = [item.value for item in assignment.value.elts if isinstance(item, ast.Constant)]
    for key in ("dataset", "resSize", "attSize", "noiseSize", "dim_t", "n_T"):
        index = constants.index("--" + key)
        assert str(DEFAULTS[key]) == constants[index + 1]
    assert "--class_embedding_norm" in constants and DEFAULTS["class_embedding_norm"]
    assert "--preprocessing" in constants and DEFAULTS["preprocessing"]
    options, training = load_options(100)
    assert vars(options) == {**DEFAULTS, "split_percent": 100} and training == {}


def test_sampling_choice_is_required(model_files, awa2_root, tmp_path):
    arguments = export_arguments(model_files, awa2_root, tmp_path / "unused")
    offset = arguments.index("--sampling")
    del arguments[offset:offset + 2]
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_export_does_not_require_con_paco(awa2_root, monkeypatch):
    original = sio.loadmat
    def checked_loadmat(path, *args, **kwargs):
        assert Path(path).name != "con_paco.mat"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(sio, "loadmat", checked_loadmat)
    options, _ = load_options(100)
    load_awa2(awa2_root, options)
