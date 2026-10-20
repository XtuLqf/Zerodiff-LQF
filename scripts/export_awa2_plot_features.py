#!/usr/bin/env python3
"""Export AWA2 100/30/10% features using the listed ZeroDiff checkpoints.

Run in the training environment:
    python scripts/export_awa2_plot_features.py --sampling one-step

The DRG choices follow the repository's AWA2 training commands. If the actual
DFG run used a different --netR_model_path, update checkpoint_paths accordingly.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SETTINGS = "att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0"
SAMPLE_COUNTS = {100: (5400, 1800), 30: (1800, 600), 10: (600, 200)}


def checkpoint_paths(percent):
    dfg_count, drg_count = SAMPLE_COUNTS[percent]
    directory = ROOT / "out" / "AWA2"
    dfg = directory / (f"zerodiff_DFG_{percent}percent_{CHECKPOINT_SETTINGS}"
                       f"_dist:5.0_f:1.5_num:{dfg_count}gzsl_VCS.tar")
    drg = directory / (f"zerodiff_DRG_{percent}percent_{CHECKPOINT_SETTINGS}"
                       f"_dist:0.0_num:{drg_count}_gzsl.tar")
    return dfg, drg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", choices=["one-step", "progressive"], required=True,
                        help="Use the same sampling rule as the Rediff comparison.")
    parser.add_argument("--split-percent", type=int, nargs="+", choices=[100, 30, 10],
                        default=[100, 30, 10], help="Default: export all three in this order.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default), cpu or auto.")
    parser.add_argument("--protocol", type=Path, help="Existing shared protocol, e.g. from Rediff.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "out" / "plot_features" / "AWA2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, without checking files or exporting.")
    args = parser.parse_args(argv)
    if len(set(args.split_percent)) != len(args.split_percent):
        parser.error("--split-percent must not contain duplicate percentages")
    return args


def build_jobs(args):
    shared_protocol = args.protocol.resolve() if args.protocol is not None else None
    jobs = []
    for percent in args.split_percent:
        dfg, drg = checkpoint_paths(percent)
        output = args.output_root.resolve() / f"{percent}pct"
        command = [
            sys.executable, "-m", "feature_export.export",
            "--dataset", "AWA2", "--dataroot", str(ROOT / "Dataset"),
            "--dfg-checkpoint", str(dfg), "--drg-checkpoint", str(drg),
            "--method", "ZeroDiff", "--split-percent", str(percent),
            "--view", "both", "--samples-per-class", "100",
            "--sampling", args.sampling, "--seed", "9182", "--batch-size", "100",
            "--device", args.device, "--output-dir", str(output),
        ]
        if shared_protocol is not None:
            command += ["--protocol", str(shared_protocol)]
        else:
            # The first export creates the protocol consumed by subsequent jobs.
            shared_protocol = output / "protocol.json"
        jobs.append((dfg, drg, output, command))
    return jobs


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    if not args.dry_run:
        # Check every selected weight/output before starting the first export.
        if args.protocol is not None and not args.protocol.is_file():
            raise FileNotFoundError(args.protocol)
        for dfg, drg, output, _ in jobs:
            for checkpoint in (dfg, drg):
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            if output.exists():
                raise FileExistsError(f"Output exists: {output}; choose a different --output-root.")
    for _, _, output, command in jobs:
        print(f"\n{'[dry-run] ' if args.dry_run else ''}Export to: {output}", flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
