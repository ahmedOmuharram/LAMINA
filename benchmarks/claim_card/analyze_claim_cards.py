from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
REPEAT_MODES = ("full", "none", "frozen")
EDIT_MODES = ("edit-quantifier", "edit-threshold")
BOOT = 10_000
PRICE_INPUT = 2.50
PRICE_CACHED = 0.25
PRICE_OUTPUT = 15.00


def sign(value: int | None) -> int | None:
    if value is None:
        return None
    return (value > 0) - (value < 0)


def load() -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(RUNS.glob("*/*/r*.json")):
        record = json.loads(path.read_text())
        runs[record["mode"]][record["claim_id"]].append(record)
    return runs


def claim_stats(records: list[dict]) -> dict[str, float]:
    verdicts = [r["verdict"] for r in records if r["verdict"] is not None]
    gold = records[0]["gold"]
    signs = [sign(v) for v in verdicts]
    return {
        "n": len(verdicts),
        "mean": statistics.fmean(verdicts) if verdicts else float("nan"),
        "sd": statistics.pstdev(verdicts) if len(verdicts) > 1 else 0.0,
        "sign_disagree": float(len(set(signs)) > 1),
        "sign_acc": statistics.fmean(s == sign(gold) for s in signs) if signs else float("nan"),
        "abstain": statistics.fmean(v == 0 for v in verdicts) if verdicts else float("nan"),
    }


def boot_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def cost(records: list[dict]) -> float:
    total = 0.0
    for r in records:
        u = r["usage"]
        fresh = u.get("input", 0) - u.get("cached", 0)
        total += (fresh * PRICE_INPUT + u.get("cached", 0) * PRICE_CACHED + u.get("output", 0) * PRICE_OUTPUT) / 1e6
    return total


def main() -> None:
    runs = load()
    print("Repeated runs")
    print(f"{'mode':8} {'claims':>6} {'runs':>5} {'sd':>16} {'sign disagree':>14} {'sign acc':>9} {'abstain':>8} {'cost':>8}")
    per_claim: dict[str, dict[str, dict[str, float]]] = {}
    for mode in REPEAT_MODES:
        stats = {c: claim_stats(rs) for c, rs in runs.get(mode, {}).items()}
        per_claim[mode] = stats
        if not stats:
            continue
        sds = [s["sd"] for s in stats.values()]
        lo, hi = boot_ci(sds)
        all_records = [r for rs in runs[mode].values() for r in rs]
        print(
            f"{mode:8} {len(stats):>6} {sum(s['n'] for s in stats.values()):>5} "
            f"{statistics.fmean(sds):>5.2f} [{lo:.2f},{hi:.2f}] "
            f"{statistics.fmean(s['sign_disagree'] for s in stats.values()):>14.2f} "
            f"{statistics.fmean(s['sign_acc'] for s in stats.values()):>9.2f} "
            f"{statistics.fmean(s['abstain'] for s in stats.values()):>8.2f} "
            f"${cost(all_records):>7.2f}"
        )

    print("\nPer claim (mean, sd)")
    claims = sorted({c for stats in per_claim.values() for c in stats})
    for claim in claims:
        gold = next(rs[0]["gold"] for m in REPEAT_MODES for c, rs in runs.get(m, {}).items() if c == claim)
        cells = []
        for mode in REPEAT_MODES:
            s = per_claim.get(mode, {}).get(claim)
            cells.append(f"{mode}={s['mean']:+.1f}/{s['sd']:.2f}" if s else f"{mode}=-")
        print(f"{claim:28} gold={gold:+d}  " + "  ".join(cells))

    print("\nEdits (frozen mode, vs mean of unedited frozen runs)")
    for mode in EDIT_MODES:
        shifts, below, above = [], 0, 0
        for claim, records in runs.get(mode, {}).items():
            base = [r["verdict"] for r in runs.get("frozen", {}).get(claim, []) if r["verdict"] is not None]
            edited = records[0]["verdict"]
            if not base or edited is None:
                continue
            shifts.append(edited - statistics.fmean(base))
            below += edited < min(base)
            above += edited > max(base)
            print(f"  {mode:16} {claim:28} edited={edited:+d} unedited={base}")
        if shifts:
            lo, hi = boot_ci(shifts)
            print(f"  {mode}: n={len(shifts)} mean shift {statistics.fmean(shifts):+.2f} [{lo:+.2f},{hi:+.2f}] below range {below} above range {above}")


if __name__ == "__main__":
    main()
