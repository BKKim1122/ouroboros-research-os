from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SEEDS = list(range(16200, 16224))
MODES = ["compositional", "packed"]
EXPECTED_PARAM_COUNT = 75102


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    problems = []

    frozen = root / "FROZEN_FILES_SHA256.txt"
    if not frozen.exists():
        problems.append("missing FROZEN_FILES_SHA256.txt")
    else:
        for line in frozen.read_text().splitlines():
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            path = root / rel
            if not path.exists():
                problems.append(f"missing frozen file: {rel}")
            elif sha256(path) != expected:
                problems.append(f"frozen hash mismatch: {rel}")

    configs = []
    devices = set()
    for mode in MODES:
        for seed in SEEDS:
            run = root / "raw" / "V16-E.C1" / f"{mode}_seed{seed}"
            for name in ["checkpoint.pt", "metadata.json", "history.csv", "base_metrics.csv"]:
                if not (run / name).exists():
                    problems.append(f"missing {mode} seed {seed}: {name}")
            if (run / "metadata.json").exists():
                meta = json.loads((run / "metadata.json").read_text())
                if meta.get("version") != "V16-E.C1":
                    problems.append(f"wrong version {mode} seed {seed}")
                if meta.get("parameter_count") != EXPECTED_PARAM_COUNT:
                    problems.append(f"parameter count mismatch {mode} seed {seed}: {meta.get('parameter_count')}")
                configs.append(json.dumps(meta.get("config"), sort_keys=True))
                devices.add(meta.get("device"))
            if (run / "base_metrics.csv").exists():
                df = pd.read_csv(run / "base_metrics.csv")
                expected_modes = {"id", "encoding_ood", "factor_ood", "rule_ood"}
                if set(df.eval_mode) != expected_modes or len(df) != 16:
                    problems.append(f"base metrics shape/conditions mismatch {mode} seed {seed}")

    if configs and len(set(configs)) != 1:
        problems.append("run configs are not identical")
    if len(devices) > 1:
        problems.append(f"mixed backends detected: {sorted(devices)}")

    causal = root / "analysis" / "V16-E.C1" / "causal_metrics_all.csv"
    decisions = root / "analysis" / "V16-E.C1" / "CONFIRMATORY_DECISIONS.csv"
    summary = root / "analysis" / "V16-E.C1" / "CONFIRMATORY_SUMMARY.json"
    for path in [causal, decisions, summary]:
        if not path.exists():
            problems.append(f"missing analysis output: {path.relative_to(root)}")

    status = "PASS" if not problems else "FAIL"
    report = {"audit": status, "problems": problems, "devices": sorted(str(x) for x in devices)}
    out = root / "analysis" / "V16-E.C1" / "AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
