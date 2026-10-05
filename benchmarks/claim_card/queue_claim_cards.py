from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
PYTHON = sys.executable
RUNNER = HERE / "run_claim_cards.py"
PRICE_INPUT = 2.50
PRICE_CACHED = 0.25
PRICE_OUTPUT = 15.00


def spent() -> float:
    total = 0.0
    for path in RUNS.glob("*/*/r*.json"):
        if "superseded" in path.parts:
            continue
        usage = json.loads(path.read_text())["usage"]
        fresh = usage.get("input", 0) - usage.get("cached", 0)
        total += (fresh * PRICE_INPUT + usage.get("cached", 0) * PRICE_CACHED + usage.get("output", 0) * PRICE_OUTPUT) / 1e6
    return total


def done(mode: str, claim: str, rep: int) -> bool:
    return (RUNS / mode / claim / f"r{rep}.json").exists()


def jobs_for(claims: list[str], reps: int) -> list[tuple[str, str, int]]:
    first = [("full", claim, 1) for claim in claims]
    rest = [(mode, claim, rep) for claim in claims for mode in ("full", "none", "frozen") for rep in range(1, reps + 1) if (mode, rep) != ("full", 1)]
    return first + rest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", nargs="+", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)

    pending = [job for job in jobs_for(args.claims, args.reps) if not done(*job)]
    running: dict[tuple[str, str, int], subprocess.Popen] = {}
    attempts: dict[tuple[str, str, int], int] = {}
    stopped = False
    while pending or running:
        for job, process in list(running.items()):
            if process.poll() is not None:
                del running[job]
                if not done(*job) and attempts[job] < 2:
                    pending.append(job)
        if not stopped and spent() >= args.budget:
            stopped = True
            print(f"budget reached at ${spent():.2f}; no new launches", flush=True)
        if stopped:
            pending = []
        for job in list(pending):
            if len(running) >= args.parallel:
                break
            mode, claim, rep = job
            if mode == "frozen" and not done("full", claim, 1):
                card_job = ("full", claim, 1)
                if attempts.get(card_job, 0) >= 2 and card_job not in running and card_job not in pending:
                    pending.remove(job)
                    print(f"drop {mode} {claim} r{rep}: no card", flush=True)
                continue
            pending.remove(job)
            attempts[job] = attempts.get(job, 0) + 1
            log = (args.logs / f"{mode}_{claim}_r{rep}.log").open("a")
            running[job] = subprocess.Popen(
                [PYTHON, "-u", str(RUNNER), "--source", "drop4", "--mode", mode, "--rep", str(rep), "--concurrency", "1", "--claims", claim],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            print(f"launch {mode} {claim} r{rep} (attempt {attempts[job]}, spent ${spent():.2f}, running {len(running)}, pending {len(pending)})", flush=True)
        time.sleep(15)
    print(f"queue finished, spent ${spent():.2f}", flush=True)


if __name__ == "__main__":
    main()
