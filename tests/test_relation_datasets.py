"""Check dataset selection/semantic loading without optional model dependencies.

MAT I/O and tensors are explicitly simulated; no model inference is claimed.
"""

import ast
from argparse import Namespace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from feature_export.specs import DATASET_SPECS

ROOT = Path(__file__).resolve().parents[1]


def isolated_options_loader():
    tree = ast.parse((ROOT / "feature_export/runtime.py").read_text(encoding="utf-8"))
    defaults = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_options")
    namespace = {"Namespace": Namespace, "Path": Path, "json": json, "math": math,
                 "DEFAULTS": ast.literal_eval(defaults), "DATASET_SPECS": DATASET_SPECS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "feature_export/runtime.py", "exec"), namespace)
    return namespace["load_options"]


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def float(self):
        return Tensor(self.value.astype(np.float32))

    @property
    def shape(self):
        return self.value.shape

    def numpy(self):
        return self.value


class DatasetTests(unittest.TestCase):
    def test_options_match_actual_dfg_training_commands(self):
        load = isolated_options_loader()
        for dataset in DATASET_SPECS:
            with self.subTest(dataset=dataset):
                tree = ast.parse((ROOT / f"scripts/run_{dataset.lower()}_zerodiff_DFG_train.py").read_text(encoding="utf-8"))
                command = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "command" for t in node.targets))
                values = [node.value if isinstance(node, ast.Constant) else None for node in command.elts]
                options, _ = load(100, dataset=dataset)
                for key in ("dataset", "class_embedding", "resSize", "attSize", "noiseSize", "dim_t", "n_T"):
                    index = values.index("--" + key)
                    self.assertEqual(str(getattr(options, key)), values[index + 1])
                self.assertEqual(options.class_embedding_norm, "--class_embedding_norm" in values)
                self.assertEqual(options.preprocessing, "--preprocessing" in values)
                config = ROOT / "docs" / f"feature_export_{dataset.lower()}.example.json"
                configured, _ = load(100, config, dataset=dataset)
                self.assertEqual(vars(options), vars(configured))

    def test_wrong_dataset_config_rejected(self):
        with self.assertRaises(ValueError):
            isolated_options_loader()(100, ROOT / "docs/feature_export_awa2.example.json", dataset="CUB")

    def test_cli_selects_correct_split_and_output(self):
        spec = importlib.util.spec_from_file_location("diagnostic_cli", ROOT / "scripts/diagnose_awa2_relations.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for dataset, seed in (("CUB", 3483), ("SUN", 4115)):
            args = module.parse_args(["--dataset", dataset, "--dfg-checkpoint", "dfg.tar",
                                      "--drg-checkpoint", "drg.tar", "--sampling", "one-step", "--seeds", str(seed)])
            names = [p.name for p in module.input_paths(args)]
            self.assertIn(DATASET_SPECS[dataset]["embedding"] + "_splits.mat", names)
            self.assertEqual(args.seeds, [seed])
            self.assertEqual(args.output_dir.name, dataset)

    def test_loader_semantic_source_and_class_coverage(self):
        tree = ast.parse((ROOT / "feature_export/data.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("array_sha256", "matlab_name", "load_dataset")]
        for dataset, spec in DATASET_SPECS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / dataset
                root.mkdir()
                names = ["res101.mat", "ce_ce.mat", spec["embedding"] + "_splits.mat"]
                for name in names:
                    (root / name).write_bytes(b"CONSTRUCTED_TEST_ONLY")
                labels = np.repeat(np.arange(spec["classes"]), 2)
                unseen = labels >= spec["classes"] - spec["unseen"]
                features = np.arange(len(labels) * 2048, dtype=np.float32).reshape(-1, 2048)
                semantic = np.arange(spec["classes"] * spec["semantic_dim"], dtype=np.float32).reshape(spec["semantic_dim"], -1) + 1
                splits = {"trainval_loc": np.flatnonzero(~unseen) + 1,
                          "test_unseen_loc": np.flatnonzero(unseen)[::-1] + 1,
                          "allclasses_names": np.array([f"class_{i}" for i in range(spec["classes"])], dtype=object)}
                key = "att" if spec["embedding"] == "sent" else "original_att"
                splits[key] = semantic
                # A conflicting alternative catches accidental CUB/attribute mixups.
                splits["original_att" if key == "att" else "att"] = semantic * 3
                documents = {"res101.mat": {"labels": labels + 1}, "ce_ce.mat": {"features": features},
                             names[-1]: splits}
                fake_torch = SimpleNamespace(from_numpy=Tensor, isfinite=lambda t: np.isfinite(t.value),
                    nn=SimpleNamespace(functional=SimpleNamespace(normalize=lambda t, **kw:
                       Tensor(t.value / np.linalg.norm(t.value, axis=1, keepdims=True)))))
                namespace = {"Path": Path, "np": np, "json": json, "hashlib": hashlib,
                             "DATASET_SPECS": DATASET_SPECS, "VisualData": lambda *args: args,
                             "torch": fake_torch, "sio": SimpleNamespace(loadmat=lambda p, **kw: documents[p.name]),
                             "file_sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest()}
                exec(compile(ast.Module(body=functions, type_ignores=[]), "feature_export/data.py", "exec"), namespace)
                options = isolated_options_loader()(100, dataset=dataset)[0]
                options.preprocessing = False
                result = namespace["load_dataset"](Path(directory), options)
                real, real_labels, real_indices, ids, class_names, attributes, alignment, files = result
                np.testing.assert_array_equal(real, features[splits["test_unseen_loc"] - 1])
                self.assertEqual(len(ids), spec["unseen"])
                self.assertEqual(class_names[0], f"class_{ids[0]}")
                expected = semantic.T
                if spec["normalize"]:
                    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
                np.testing.assert_allclose(attributes.numpy(), expected, rtol=1e-6)
                self.assertEqual(alignment["dataset"], dataset)
                self.assertIn(names[-1], files)


if __name__ == "__main__":
    unittest.main()
