#!/usr/bin/env python3
"""Measure semantic ordering reversals supported by real AWA2 class centres.

Run from the repository root. No training or PaCo data is needed.
"""

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--dfg-checkpoint", type=Path, required=True)
    parser.add_argument("--drg-checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--split-percent", type=int, choices=[100, 30, 10], default=100)
    parser.add_argument("--sampling", choices=["one-step", "progressive"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9182, 9183, 9184, 9185, 9186])
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--tie-gap", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "relation_diagnostic" / "AWA2")
    parser.add_argument("--check-inputs", action="store_true", help="Check paths only; no model/dependency loading.")
    args = parser.parse_args(argv)
    if args.samples_per_class < 1 or args.batch_size < 1:
        parser.error("sample count and batch size must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**63 - 1 for s in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**63 - 1)")
    if not 0 <= args.tie_gap < args.min_gap < 1:
        parser.error("require 0 <= tie-gap < min-gap < 1")
    return args


def input_paths(args):
    paths = [args.dataroot / "AWA2" / name for name in ("res101.mat", "att_splits.mat", "ce_ce.mat")]
    if args.split_percent != 100:
        paths.append(args.dataroot / "AWA2" / f"split_{args.split_percent}percent.mat")
    paths += [args.dfg_checkpoint, args.drg_checkpoint]
    if args.training_config is None:
        candidate = args.dfg_checkpoint.parent / "config.json"
        if candidate.is_file():
            args.training_config = candidate
    if args.training_config is not None:
        paths.append(args.training_config)
    return paths


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty diagnostic table.")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_svg(summary, rows, outcomes, class_names):
    """Standalone vector figure; distances have separate axes in each space."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="540" viewBox="0 0 1120 540">',
             '<rect width="1120" height="540" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#17212b">']

    def text(x, y, value, size=15, **attrs):
        extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>')

    text(30, 35, "ZeroDiff class-relation diagnostic on AWA2 unseen classes", 23, font_weight="bold")
    text(30, 65, f'Eligible triplets: {summary["eligible_triplets"]} / {summary["total_triplets"]}; '
         f'reference and real relative gaps > {summary["min_gap"]:g}', 14)
    text(30, 108, "(a) Measured ordering reversal", 19, font_weight="bold")
    example = summary["example"]
    if example is None:
        text(35, 180, "No eligible reversed triplet was observed.", 16)
    else:
        row = rows[example["triplet_index"]]
        observation = outcomes[example["seed_index"]][example["triplet_index"]]
        a, b, c = [class_names[row[key]] for key in ("anchor", "near", "far")]
        text(35, 140, f"Anchor: {a}", 16)
        text(35, 166, f"B: {b}     C: {c}", 15)
        spaces = [("Class semantics", row["semantic_near"], row["semantic_far"]),
                  ("Real visual centres", row["real_near"], row["real_far"]),
                  ("Generated centres", observation["generated_near"], observation["generated_far"])]
        for index, (label, near, far) in enumerate(spaces):
            y = 209 + index * 83
            text(35, y, label, 15, font_weight="bold")
            maximum = max(near, far, 1e-12)
            for offset, value, color, name in [(10, near, "#247ba0", "B"), (33, far, "#e59a34", "C")]:
                text(40, y + offset + 13, name, 13)
                width = 245 * value / maximum
                parts.append(f'<rect x="62" y="{y+offset}" width="{width:.3f}" height="16" fill="{color}"/>')
                text(320, y + offset + 13, f"{value:.6f}", 13)
            text(423, y + 33, "B < C" if near < far else "B > C", 15)
        text(35, 466, f'Case reverses in {example["reversed_seed_count"]}/{example["seed_count"]} seeds; '
             f'displayed seed {summary["seeds"][example["seed_index"]]}', 13)
        text(35, 488, "Bar scales are separate across spaces; values are cosine distances.", 12)
    text(580, 108, "(b) Reversal rate across generation seeds", 19, font_weight="bold")
    records = summary["per_seed"]
    if summary["eligible_triplets"]:
        max_rate = max(record["reversal_rate"] for record in records)
        ymax = min(100, max(5, (int(max_rate * 100 / 5) + 2) * 5))
        for tick in range(6):
            y = 405 - tick * 47
            parts.append(f'<line x1="635" y1="{y}" x2="1080" y2="{y}" stroke="#dce2e7"/>')
            text(590, y + 5, f"{tick*ymax/5:g}%", 13)
        step = 430 / len(records)
        for index, record in enumerate(records):
            x = 645 + step * index
            height = record["reversal_rate"] * 100 / ymax * 235
            width = min(52, step * 0.65)
            parts.append(f'<rect x="{x:.3f}" y="{405-height:.3f}" width="{width:.3f}" height="{height:.3f}" fill="#247ba0"/>')
            text(x, 395 - height, f'{record["reversal_rate"]*100:.1f}%', 12)
            text(x, 430, record["seed"], 12)
        aggregate = summary["aggregate"]
        sd = aggregate["sample_sd_reversal_rate"]
        sd_label = f"{sd*100:.2f} percentage points" if sd is not None else "undefined (one seed)"
        text(595, 465, f'Mean {aggregate["mean_reversal_rate"]*100:.2f}%  |  '
             f'SD {sd_label}', 14)
        text(595, 488, "Repeated generation from one fixed checkpoint.", 12)
    else:
        text(595, 200, "No eligible triplets; reversal rate is undefined.", 15)
    text(30, 523, f'Sampling: {summary["sampling"]}; {summary["samples_per_class"]} generated features per class; '
         'all distances computed in the original feature spaces.', 12)
    parts.append("</g></svg>")
    return "\n".join(parts)


def run(args):
    import numpy as np
    import torch
    from feature_export.data import file_sha256, load_awa2
    from feature_export.relations import class_centers, cosine_distances, reference_triplets, score_triplets, choose_example
    from feature_export.runtime import load_options, FeatureRuntime, generate_features

    options, training = load_options(args.split_percent, args.training_config)
    if training.get("drg_sha256") and training["drg_sha256"] != file_sha256(args.drg_checkpoint):
        raise ValueError("DRG checkpoint hash disagrees with the training config.")
    data = load_awa2(args.dataroot, options)
    device = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = FeatureRuntime(options, args.dfg_checkpoint, args.drg_checkpoint, device)
    metadata_method = runtime.method_metadata
    declared_method = (metadata_method.get("method") if isinstance(metadata_method, dict) else metadata_method)
    if runtime.has_relation_state or (declared_method and str(declared_method).lower() != "zerodiff"):
        raise ValueError("This baseline diagnostic requires a pure ZeroDiff checkpoint.")
    real_centers, real_counts = class_centers(data.real_features, data.real_labels, data.class_ids)
    semantics = data.attributes[data.class_ids].cpu().numpy()
    semantic_dist = cosine_distances(semantics)
    real_dist = cosine_distances(real_centers)
    rows = reference_triplets(semantic_dist, real_dist, args.min_gap)
    outcomes_by_seed, records, generated_centers, generated_distances = [], [], [], []
    for seed in args.seeds:
        generated, labels = generate_features(runtime, data, args.samples_per_class, seed, args.batch_size, args.sampling)
        centers, counts = class_centers(generated, labels, data.class_ids)
        if not np.all(counts == args.samples_per_class):
            raise ValueError("Unexpected generated class counts.")
        distances = cosine_distances(centers)
        scores, outcomes = score_triplets(rows, distances, args.tie_gap)
        records.append({"seed": seed, **scores})
        outcomes_by_seed.append(outcomes)
        generated_centers.append(centers)
        generated_distances.append(distances)
        print(json.dumps(records[-1], allow_nan=False), flush=True)
    eligible = sum(row["eligible"] for row in rows)
    rates = [r["reversal_rate"] for r in records] if eligible else []
    aggregate = ({"mean_reversal_rate": float(np.mean(rates)),
                  "sample_sd_reversal_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                  "min_reversal_rate": min(rates), "max_reversal_rate": max(rates),
                  "mean_tie_rate": float(np.mean([r["tie_rate"] for r in records]))} if eligible else None)
    example = choose_example(rows, outcomes_by_seed)
    summary = {
        "schema_version": "awa2-class-relations-v1", "dataset": "AWA2", "split": "test_unseen_loc",
        "sampling": args.sampling, "seeds": args.seeds, "samples_per_class": args.samples_per_class,
        "min_gap": args.min_gap, "tie_gap": args.tie_gap, "distance": "cosine_of_class_mean",
        "total_triplets": len(rows), "eligible_triplets": eligible,
        "coverage": eligible / len(rows), "per_seed": records, "aggregate": aggregate, "example": example,
        "conclusion": ("undefined_no_eligible_triplets" if not eligible else
                       "reversals_observed" if any(r["reversed_triplets"] for r in records) else "no_reversals_observed"),
        "scope": "Class-level semantic ordering on one fixed checkpoint; seeds vary generation only.",
        "example_selection": "Prefer majority-repeat candidates; median reversal frequency, then lexicographic triplet order; median reversed gap seed.",
    }
    audit = []
    for index, row in enumerate(rows):
        common = {"triplet_index": index, **row}
        for role in ("anchor", "near", "far"):
            common[role + "_class_id"] = int(data.class_ids[row[role]])
            common[role + "_class_name"] = str(data.class_names[row[role]])
        for seed, outcomes in zip(args.seeds, outcomes_by_seed):
            audit.append({**common, "seed": seed, **outcomes[index]})
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".relation-diagnostic-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        write_csv(stage / "per_seed.csv", records)
        write_csv(stage / "triplets.csv", audit)
        np.savez_compressed(stage / "centers.npz", class_ids=data.class_ids, class_names=data.class_names,
                            semantics=semantics, real_centers=real_centers, real_counts=real_counts,
                            generated_centers=np.stack(generated_centers), seeds=np.asarray(args.seeds),
                            semantic_distances=semantic_dist, real_distances=real_dist,
                            generated_distances=np.stack(generated_distances))
        (stage / "figure1.svg").write_text(figure_svg(summary, rows, outcomes_by_seed, data.class_names), encoding="utf-8")
        report = ["# ZeroDiff class relation diagnostic", "", summary["scope"], "",
                  f"Eligible triplets: {eligible}/{len(rows)} ({summary['coverage']:.1%}).",
                  f"Outcome: {summary['conclusion']}."]
        if aggregate:
            sd = aggregate["sample_sd_reversal_rate"]
            sd_label = f"{sd*100:.2f} percentage points" if sd is not None else "undefined (one seed)"
            report += [f"Mean reversal rate: {aggregate['mean_reversal_rate']:.2%}; sample SD: "
                       f"{sd_label}.",
                       f"Seed range: {min(rates):.2%} to {max(rates):.2%}.",
                       f"Mean tie rate: {aggregate['mean_tie_rate']:.2%}."]
        report += ["", "Eligibility is fixed using semantics and real features before examining generated results.",
                   "Ties are reported separately and are not counted as strict reversals.",
                   "No training-seed confidence interval or causal attribution is implied.",
                   "Cosine distance is computed after averaging features, with separate scales in the case plot.",
                   "Training configuration and checkpoint pairing must match the actual baseline run."]
        (stage / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "device": str(runtime.device),
            "options": vars(options), "batch_size": args.batch_size, "data": data.files,
            "data_alignment": data.alignment, "git": git_provenance(),
            "training_config": str(args.training_config.resolve()) if args.training_config else None,
            "config_status": "provided_or_autodetected" if args.training_config else "published_defaults_user_declared",
            "checkpoint_pairing": "user_supplied; DRG hash verified if present in training config",
            "inputs": {str(p.resolve()): file_sha256(p) for p in [args.dfg_checkpoint, args.drg_checkpoint]
                       + ([args.training_config] if args.training_config else [])},
            "source_sha256": {name: file_sha256(ROOT / name) for name in
                              ["scripts/diagnose_awa2_relations.py", "feature_export/relations.py",
                               "feature_export/runtime.py", "feature_export/data.py", "zerodiff_tools.py"]},
            "artifacts": {p.name: file_sha256(p) for p in stage.iterdir()},
        }
        (stage / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        with zipfile.ZipFile(stage / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.iterdir()):
                if path.name != "results.zip":
                    archive.write(path, path.name)
        os.rename(stage, output)
    print(f"Results: {output}\nReturn this file: {output / 'results.zip'}", flush=True)
    return summary


def git_provenance():
    def query(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True,
                                           encoding="utf-8", stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": query("rev-parse", "HEAD"), "working_tree_status": query("status", "--porcelain")}


def main(argv=None):
    args = parse_args(argv)
    missing = [str(path) for path in input_paths(args) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n" + "\n".join(missing))
    if args.check_inputs:
        print("All input paths exist. Weights, configuration and dependencies have not been validated.")
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}; choose a new --output-dir.")
    return run(args)


if __name__ == "__main__":
    main()
