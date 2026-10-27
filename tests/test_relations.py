"""Known geometry checks, requiring only NumPy and the standard library.

These constructed features are software checks, never experimental evidence.
"""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from feature_export.relations import (
    choose_example, class_centers, cosine_distances, reference_triplets, score_triplets,
)


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnose", ROOT / "scripts" / "diagnose_awa2_relations.py")
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


class RelationTests(unittest.TestCase):
    def setUp(self):
        angles = np.deg2rad([0, 20, 70])
        self.centers = np.column_stack([np.cos(angles), np.sin(angles)])
        self.distances = cosine_distances(self.centers)
        self.rows = reference_triplets(self.distances, self.distances)

    def test_hand_calculated_order_and_reversal(self):
        np.testing.assert_allclose(self.distances[0, 1], 1 - np.cos(np.deg2rad(20)))
        scores, _ = score_triplets(self.rows, self.distances)
        self.assertEqual(scores["reversed_triplets"], 0)
        swapped = cosine_distances(self.centers[[0, 2, 1]])
        scores, outcomes = score_triplets(self.rows, swapped)
        anchor_zero = next(i for i, r in enumerate(self.rows) if r["anchor"] == 0)
        self.assertEqual(outcomes[anchor_zero]["outcome"], "reversed")
        self.assertGreater(scores["reversal_rate"], 0)

    def test_real_disagreement_excluded_before_scoring(self):
        real = cosine_distances(self.centers[[0, 2, 1]])
        rows = reference_triplets(self.distances, real)
        anchor_zero = next(r for r in rows if r["anchor"] == 0)
        self.assertFalse(anchor_zero["eligible"])
        self.assertLess(anchor_zero["real_gap"], 0)
        frozen = [r.copy() for r in rows]
        score_triplets(rows, real)
        score_triplets(rows, self.distances)
        self.assertEqual(rows, frozen)

    def test_near_ties_excluded_and_collapse_is_tie(self):
        equal = np.ones((3, 3)) - np.eye(3)
        rows = reference_triplets(equal, equal)
        scores, _ = score_triplets(rows, equal)
        self.assertEqual(scores["eligible_triplets"], 0)
        self.assertIsNone(scores["reversal_rate"])
        collapsed = cosine_distances(np.ones((3, 4)))
        scores, outcomes = score_triplets(self.rows, collapsed)
        self.assertEqual(scores["reversal_rate"], 0)
        self.assertEqual(scores["tie_rate"], 1)
        self.assertTrue(all(o["outcome"] == "tie" for o in outcomes))

    def test_gap_threshold_has_defined_boundary(self):
        semantic = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
        rows = reference_triplets(semantic, semantic, min_gap=0.5)
        row = next(r for r in rows if r["anchor"] == 0)
        self.assertEqual(row["semantic_gap"], 0.5)
        self.assertFalse(row["eligible"])

    def test_all_360_triplets_enumerated_once(self):
        distance = cosine_distances(np.random.default_rng(12).normal(size=(10, 5)))
        rows = reference_triplets(distance, distance)
        self.assertEqual(len(rows), 360)
        keys = [(r["anchor"], tuple(sorted((r["near"], r["far"])))) for r in rows]
        self.assertEqual(len(set(keys)), 360)

    def test_centers_preserve_nonconsecutive_class_ids(self):
        centers, counts = class_centers([[1, 0], [3, 0], [0, 2]], [42, 42, 7], [7, 42])
        np.testing.assert_array_equal(centers, [[0, 2], [2, 0]])
        np.testing.assert_array_equal(counts, [1, 2])
        with self.assertRaises(ValueError):
            class_centers([[1, 0]], [42], [7, 42])

    def test_zero_and_nonfinite_vectors_rejected(self):
        for vectors in ([[0, 0], [1, 0]], [[float("nan"), 1], [1, 0]]):
            with self.assertRaises(ValueError):
                cosine_distances(vectors)

    def test_example_selection_and_absence(self):
        _, kept = score_triplets(self.rows, self.distances)
        _, reversed_outcomes = score_triplets(self.rows, cosine_distances(self.centers[[0, 2, 1]]))
        self.assertIsNone(choose_example(self.rows, [kept] * 5))
        example = choose_example(self.rows, [reversed_outcomes] * 3 + [kept] * 2)
        self.assertTrue(example["majority_repeat"])
        self.assertEqual(example["reversed_seed_count"], 3)
        self.assertEqual(example["seed_index"], 0)

    def test_svg_with_and_without_example(self):
        scores, outcomes = score_triplets(self.rows, cosine_distances(self.centers[[0, 2, 1]]))
        summary = {"eligible_triplets": scores["eligible_triplets"], "total_triplets": len(self.rows),
                   "min_gap": .05, "seeds": [9182], "sampling": "one-step", "samples_per_class": 100,
                   "per_seed": [{"seed": 9182, **scores}],
                   "aggregate": {"mean_reversal_rate": scores["reversal_rate"], "sample_sd_reversal_rate": 0},
                   "example": choose_example(self.rows, [outcomes])}
        svg = diagnose.figure_svg(summary, self.rows, [outcomes], ["A & B", "C < D", "E"])
        ET.fromstring(svg)
        self.assertIn("A &amp; B", svg)
        summary.update(eligible_triplets=0, example=None, aggregate=None)
        svg = diagnose.figure_svg(summary, self.rows, [outcomes], ["A", "B", "C"])
        ET.fromstring(svg)
        self.assertIn("rate is undefined", svg)

    def test_missing_inputs_are_reported_together_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as caught:
                diagnose.main(["--dataroot", directory, "--dfg-checkpoint", directory + "/dfg.tar",
                               "--drg-checkpoint", directory + "/drg.tar", "--sampling", "one-step", "--check-inputs"])
            for name in ("res101.mat", "att_splits.mat", "ce_ce.mat", "dfg.tar", "drg.tar"):
                self.assertIn(name, str(caught.exception))

    def test_result_bundle_pipeline_with_explicit_fake_inference(self):
        """Exercise reporting/packaging only; this does not validate model inference."""
        rng = np.random.default_rng(92)
        ids = np.arange(0, 50, 5)
        reference = rng.uniform(.1, 1, (10, 85))
        centers = np.pad(reference, ((0, 0), (0, 2048 - 85)))

        class Attributes:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, key):
                return Attributes(self.value[key])

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        all_attributes = np.ones((50, 85))
        all_attributes[ids] = reference
        data = SimpleNamespace(real_features=np.repeat(centers, 2, axis=0),
                               real_labels=np.repeat(ids, 2), class_ids=ids,
                               class_names=np.array([f"class_{c}" for c in ids]),
                               attributes=Attributes(all_attributes), files={}, alignment={})
        fake_data = ModuleType("feature_export.data")
        fake_data.load_dataset = lambda *_: data
        fake_data.file_sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "FAKE_TEST_ONLY"
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_runtime = ModuleType("feature_export.runtime")
        fake_runtime.load_options = lambda *_, **kwargs: (SimpleNamespace(split_percent=100), {})
        fake_runtime.FeatureRuntime = lambda *_: SimpleNamespace(
            device="cpu", method_metadata=None, has_relation_state=False)

        def fake_generate(runtime, data, count, seed, batch, sampling):
            order = np.arange(10)
            order[[0, 1]] = order[[1, 0]]
            return np.repeat(centers[order], count, axis=0), np.repeat(ids, count)

        fake_runtime.generate_features = fake_generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AWA2").mkdir()
            for name in ("res101.mat", "att_splits.mat", "ce_ce.mat"):
                (root / "AWA2" / name).write_bytes(b"FAKE_TEST_ONLY")
            for name in ("dfg.tar", "drg.tar"):
                (root / name).write_bytes(b"FAKE_TEST_ONLY")
            output = root / "result"
            arguments = ["--dataroot", str(root), "--dfg-checkpoint", str(root / "dfg.tar"),
                         "--drg-checkpoint", str(root / "drg.tar"), "--sampling", "one-step",
                         "--seeds", "9182", "9183", "--samples-per-class", "2", "--output-dir", str(output)]
            with patch.dict("sys.modules", {"torch": fake_torch, "feature_export.data": fake_data,
                                             "feature_export.runtime": fake_runtime}):
                with patch("builtins.print"):
                    summary = diagnose.main(arguments)
                self.assertEqual(summary["total_triplets"], 360)
                self.assertGreater(summary["eligible_triplets"], 0)
                self.assertEqual(summary["conclusion"], "reversals_observed")
                with zipfile.ZipFile(output / "results.zip") as archive:
                    self.assertEqual(set(archive.namelist()), {"summary.json", "per_seed.csv", "triplets.csv",
                                     "centers.npz", "figure1.svg", "report.md", "manifest.json"})
                    self.assertIsNone(archive.testzip())
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["torch"], "FAKE_TEST_ONLY")
                    for name, digest in manifest["artifacts"].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                    ET.fromstring(archive.read("figure1.svg"))
                with self.assertRaises(FileExistsError):
                    diagnose.main(arguments)


if __name__ == "__main__":
    unittest.main()
