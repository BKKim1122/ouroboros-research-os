from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

SEEDS = list(range(16200, 16224))
MODES = ["compositional", "packed"]


def run_one(root: Path, mode: str, seed: int, device: str, threads: int) -> tuple[str, int, int]:
    run_dir = root / "raw" / "V16-E.C1" / f"{mode}_seed{seed}"
    metrics = run_dir / "base_metrics.csv"
    log = root / "logs" / f"{mode}_seed{seed}.log"
    if metrics.exists():
        return mode, seed, 0
    cmd = [
        sys.executable,
        str(root / "src" / "v16e_c1_confirmatory.py"),
        "--root", str(root),
        "--seed", str(seed),
        "--rule-mode", mode,
        "--steps", "1100",
        "--eval-n", "4000",
        "--threads", str(threads),
        "--device", device,
        "--resume",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=root)
    return mode, seed, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "1")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "2")))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    tasks = [(mode, seed) for mode in MODES for seed in SEEDS]

    if args.device.startswith("cuda") and args.jobs != 1:
        raise SystemExit("Use --jobs 1 for a single CUDA device to avoid run contention.")

    failures = []
    if args.jobs == 1:
        for mode, seed in tasks:
            m, s, code = run_one(root, mode, seed, args.device, args.threads)
            print(f"{m} seed {s}: {'OK' if code == 0 else 'FAIL'}", flush=True)
            if code != 0:
                failures.append((m, s, code))
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, root, mode, seed, args.device, args.threads) for mode, seed in tasks]
            for future in concurrent.futures.as_completed(futures):
                m, s, code = future.result()
                print(f"{m} seed {s}: {'OK' if code == 0 else 'FAIL'}", flush=True)
                if code != 0:
                    failures.append((m, s, code))

    if failures:
        print("Failures:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
