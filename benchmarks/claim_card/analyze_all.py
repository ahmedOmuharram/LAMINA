from __future__ import annotations

import itertools
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
REPS = 3
BOOT = 10_000
COMPUTE = {
    "calculate_equilibrium_at_point", "fact_check_microstructure_claim", "sweep_microstructure_claim_over_region",
    "plot_binary_phase_diagram", "plot_composition_temperature", "calculate_phase_fractions_vs_temperature",
    "analyze_phase_fraction_trend", "verify_phase_formation_across_composition", "compute_scheil_solidification",
    "compare_scheil_and_equilibrium", "find_invariant_reactions", "compare_solute_lattice_effects",
    "analyze_solute_lattice_effect", "calculate_solute_lattice_effect", "assess_phase_strength_and_stiffness_claims",
    "calculate_voltage_from_formation_energy", "analyze_lithiation_mechanism", "check_composition_stability",
    "analyze_anode_viability", "estimate_ion_hopping_barrier", "estimate_graphite_intercalation_barrier",
    "estimate_surface_diffusion_barrier",
}


def sign(value: int) -> int:
    return (value > 0) - (value < 0)


def records(mode: str, claim: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RUNS / mode / claim).glob("r*.json"))]


def verdicts(mode: str, claim: str, limit: int | None = REPS) -> list[int]:
    found = [r["verdict"] for r in records(mode, claim) if r["verdict"] is not None]
    return found[:limit] if limit else found


def spread(values: list[int]) -> tuple[float, float]:
    return st.pstdev(values), float(len({sign(v) for v in values}) > 1)


def boot_mean_ci(values: list[float]) -> tuple[float, float, float]:
    rng = random.Random(0)
    means = sorted(st.fmean(rng.choices(values, k=len(values))) for _ in range(BOOT))
    return st.fmean(values), means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def computed(claim: str) -> bool:
    runs = records("frozen", claim)
    return sum(any(t["tool"] in COMPUTE for t in r["tool_calls"]) for r in runs) >= 2


def pairwise_flip(values: list[int]) -> float:
    pairs = list(itertools.combinations(values, 2))
    return st.fmean(a != b for a, b in pairs) if pairs else float("nan")


def main() -> None:
    claims = sorted(p.name for p in (RUNS / "frozen").iterdir() if p.is_dir())
    gold = {c: records("frozen", c)[0]["gold"] for c in claims}
    kind = {c: "computed" if computed(c) else "searched" for c in claims}
    summary: dict = {"claims": len(claims), "computed": sum(k == "computed" for k in kind.values())}

    print(f"{len(claims)} claims, {summary['computed']} computed, first {REPS} runs per condition\n")
    print("Run-to-run spread (mean over claims of per-claim SD and sign disagreement)")
    for group in ("all", "computed", "searched"):
        chosen = [c for c in claims if group == "all" or kind[c] == group]
        line = []
        for mode in ("full", "none", "frozen"):
            sds, dis = zip(*(spread(verdicts(mode, c)) for c in chosen if len(verdicts(mode, c)) >= 2))
            m, lo, hi = boot_mean_ci(list(sds))
            line.append(f"{mode} SD {m:.2f} [{lo:.2f},{hi:.2f}] disagree {st.fmean(dis):.2f}")
            summary[f"{group}_{mode}_sd"] = round(m, 3)
        print(f"  {group:9} n={len(chosen):2}  " + " | ".join(line))

    print("\nFrozen card with computation vs search only (computed claims with search-only runs)")
    pairs = [c for c in claims if kind[c] == "computed" and verdicts("frozen-search", c)]
    diff_sd, rows = [], []
    for c in pairs:
        a, b = verdicts("frozen", c), verdicts("frozen-search", c)
        sa, sb = spread(a), spread(b)
        diff_sd.append(sb[0] - sa[0])
        acc_a = st.fmean(sign(v) == sign(gold[c]) for v in a)
        acc_b = st.fmean(sign(v) == sign(gold[c]) for v in b)
        rows.append((c, gold[c], a, b, sa, sb, acc_a, acc_b))
        print(f"  {c:30} gold {gold[c]:+d}  frozen {a} sd {sa[0]:.2f}   search-only {b} sd {sb[0]:.2f}")
    m, lo, hi = boot_mean_ci(diff_sd)
    print(f"  n={len(pairs)} claims: SD frozen {st.fmean(r[4][0] for r in rows):.2f} vs search-only {st.fmean(r[5][0] for r in rows):.2f}; "
          f"paired difference {m:+.2f} [{lo:+.2f},{hi:+.2f}]; sign disagreement {st.fmean(r[4][1] for r in rows):.2f} vs {st.fmean(r[5][1] for r in rows):.2f}; "
          f"sign accuracy {st.fmean(r[6] for r in rows):.2f} vs {st.fmean(r[7] for r in rows):.2f}; "
          f"search-only SD higher on {sum(d > 0 for d in diff_sd)}, equal on {sum(d == 0 for d in diff_sd)}, lower on {sum(d < 0 for d in diff_sd)}")
    summary["search_only"] = {"n": len(pairs), "sd_frozen": round(st.fmean(r[4][0] for r in rows), 3), "sd_search": round(st.fmean(r[5][0] for r in rows), 3), "diff": [round(m, 3), round(lo, 3), round(hi, 3)]}

    print("\nThesis readings replayed as frozen card edits (GPT-5.4) vs the thesis H3 run (gpt-4o)")
    h3 = json.loads((ROOT / "benchmarks" / "h3" / "runs" / "h3_gpt-4o_policy.json").read_text())["examples"]
    h3_by = {}
    for e in h3:
        if e.get("mapping_id"):
            h3_by.setdefault(e["claim_id"], {})[e["mapping_id"]] = e["predicted_label"]
    fr_now, fr_then, shifts = [], [], []
    for c in claims:
        edits = {p.parent.parent.name.removeprefix("edit-reading-"): json.loads(p.read_text())["verdict"] for p in RUNS.glob(f"edit-reading-*/{c}/r1.json")}
        edits = {k: v for k, v in edits.items() if v is not None}
        if len(edits) < 2:
            continue
        now = [edits[k] for k in sorted(edits)]
        then = [h3_by.get(c, {}).get(k) for k in sorted(edits)]
        base = st.fmean(verdicts("frozen", c, None))
        shifts.extend(abs(v - base) for v in now)
        fr_now.append(pairwise_flip(now))
        if all(t is not None for t in then):
            fr_then.append(pairwise_flip(then))
        print(f"  {c:30} gold {gold[c]:+d}  now {now} (frozen base {base:+.1f})  thesis {then}")
    m, lo, hi = boot_mean_ci(fr_now)
    print(f"  flip rate across readings: GPT-5.4 cards {m:.2f} [{lo:.2f},{hi:.2f}] over {len(fr_now)} claims; thesis gpt-4o {st.fmean(fr_then):.2f} over {len(fr_then)} claims; mean |shift| from frozen base {st.fmean(shifts):.2f}")
    summary["readings"] = {"claims": len(fr_now), "flip_rate_now": round(m, 3), "ci": [round(lo, 3), round(hi, 3)], "flip_rate_thesis": round(st.fmean(fr_then), 3)}

    print("\nSign accuracy against gold (first runs per condition)")
    for mode in ("full", "none", "frozen"):
        acc = [st.fmean(sign(v) == sign(gold[c]) for v in verdicts(mode, c)) for c in claims if verdicts(mode, c)]
        print(f"  {mode:7} {st.fmean(acc):.2f}")
        summary[f"sign_acc_{mode}"] = round(st.fmean(acc), 3)
    (RUNS / "logs" / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
