from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from kani import ChatRole
from openai import AsyncOpenAI

from backend.kani_client import MPKani
from backend.models import BACKBONE_MODEL, CARD_PARSER_MODEL, CARD_PARSER_REASONING_EFFORT
from backend.prompts import KANI_SYSTEM_PROMPT

CLAIMSPY = Path(os.environ.get("CLAIMSPY_DIR", "~/repos/claimspyv2-paper")).expanduser()
RUNS = Path(__file__).resolve().parent / "runs"
CLAIM_IDS = [
    "computational_tools_0003",
    "computational_tools_0016",
    "computational_tools_0017",
    "computational_tools_0025",
    "computational_tools_0028",
    "computational_tools_0036",
]
MODES = ("full", "none", "frozen", "frozen-search", "edit-quantifier", "edit-threshold", "edit")
SEARCH_ONLY = {"search_web"}
VERDICT = re.compile(r"\*{0,2}VERDICT\*{0,2}:\s*\*{0,2}([+-]?\d)\*{0,2}", re.IGNORECASE)
TOOL_RESULT_CHARS = 6000

QUERY = """Evaluate the following materials science claim and provide a feasibility rating.

Claim: {claim}

Rate the claim's feasibility on a scale from -2 to +2:
- -2: Extremely infeasible (clearly contradicted by evidence)
- -1: Likely infeasible (evidence suggests this is unlikely)
- 0: Undecidable (insufficient evidence or ambiguous)
- +1: Likely feasible (evidence suggests this is plausible)
- +2: Extremely feasible (strongly supported by evidence)

Use computational tools to verify the claim if appropriate. Provide your reasoning, then end your response with exactly this format on its own line:
VERDICT: <integer from -2 to 2>"""

CARD_STEP = """

Before using any tool, complete a Claim Card for this claim, following these instructions:

{guideline}

Write only the Claim Card now. Do not call tools and do not give a verdict yet."""

ASSESS_STEP = """Now assess the claim under the Claim Card you wrote, without changing its interpretation, quantifier, threshold, or defaults. Use computational tools to verify the claim if appropriate. Provide your reasoning, then end your response with exactly this format on its own line:
VERDICT: <integer from -2 to 2>"""


def load_drop4() -> dict[str, dict[str, Any]]:
    rows = json.loads((ROOT / "benchmarks" / "data" / "drop4_claims.json").read_text())
    return {row["id"]: {"claim": row["text"], "gold": row["gold_label"]} for row in rows}


def load_claims() -> dict[str, dict[str, Any]]:
    subset = CLAIMSPY / "data" / "claim-card" / "repeat-subset"
    claims = {}
    for line in (subset / "problems.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["problem_id"] in CLAIM_IDS:
            claims[row["problem_id"]] = {"claim": row["claim"]}
    for line in (subset / "gold-standard.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["problem_id"] in claims:
            claims[row["problem_id"]]["gold"] = row["likert_score"]
    return claims


def card_guideline() -> str:
    text = (CLAIMSPY / "claimspy" / "docs" / "materials" / "CLAIM_VERIFICATION_GUIDELINE.md").read_text()
    first = re.search(r"### 1\.1 .*?(?=### 1\.3 )", text, re.S).group(0)
    second = re.search(r"### 1\.6 .*?(?=\n---)", text, re.S).group(0)
    return (first + second).strip()


def parser_prompt() -> str:
    return (CLAIMSPY / "claimspy" / "docs" / "materials" / "PREFLIGHT_PARSE.md").read_text()


def cardmode():
    spec = importlib.util.spec_from_file_location("cardmode", CLAIMSPY / "claimspy" / "cardmode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def threshold_prompt() -> str:
    source = (CLAIMSPY / "experiments" / "claim_card" / "make_counterfactual_cards.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "THRESHOLD_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("THRESHOLD_PROMPT not found")


def usage_of(message) -> dict[str, int]:
    usage = (message.extra or {}).get("openai_usage") or {}
    details = usage.get("input_tokens_details") or {}
    return {
        "input": usage.get("input_tokens", 0),
        "cached": details.get("cached_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }


def add_usage(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


async def parse_card(client: AsyncOpenAI, text: str) -> dict[str, Any] | None:
    response = await client.chat.completions.create(
        model=CARD_PARSER_MODEL,
        messages=[{"role": "system", "content": parser_prompt()}, {"role": "user", "content": text}],
        reasoning_effort=CARD_PARSER_REASONING_EFFORT,
        temperature=0,
        max_completion_tokens=8192,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)["PreFlight"]["ClaimCard"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def assess(kani: MPKani, query: str, record: dict[str, Any]) -> str:
    final = ""
    tag = f"{record['mode']} {record['claim_id']} r{record['rep']}"
    async for message in kani.full_round(query, **kani.get_hyperparams()):
        add_usage(record["usage"], usage_of(message))
        elapsed = round(time.time() - record["started"])
        if message.role == ChatRole.ASSISTANT and message.tool_calls:
            for call in message.tool_calls:
                record["tool_calls"].append({"tool": call.function.name, "arguments": call.function.arguments})
                print(f"  [{tag} {elapsed}s] call {call.function.name} {call.function.arguments[:200]}", flush=True)
        elif message.role == ChatRole.FUNCTION:
            record["tool_results"].append({"tool": message.name, "result": (message.text or "")[:TOOL_RESULT_CHARS]})
            print(f"  [{tag} {elapsed}s] result {message.name} {len(message.text or '')} chars", flush=True)
        elif message.role == ChatRole.ASSISTANT:
            final = message.text or ""
    return final


async def run_one(
    claim_id: str,
    claim: dict[str, Any],
    mode: str,
    rep: int,
    card: dict[str, Any] | None,
    client: AsyncOpenAI,
) -> dict[str, Any]:
    enabled = SEARCH_ONLY if mode == "frozen-search" else None
    kani = MPKani(model=BACKBONE_MODEL, system_prompt=KANI_SYSTEM_PROMPT, enabled_functions=enabled)
    record: dict[str, Any] = {
        "claim_id": claim_id,
        "claim": claim["claim"],
        "gold": claim["gold"],
        "mode": mode,
        "rep": rep,
        "tools_enabled": sorted(enabled) if enabled else "all",
        "model": BACKBONE_MODEL,
        "hyperparams": kani.get_hyperparams(),
        "card_text": None,
        "card": card,
        "tool_calls": [],
        "tool_results": [],
        "usage": {},
        "started": time.time(),
    }
    query = QUERY.format(claim=claim["claim"])
    if mode == "full":
        written = await kani.chat_round(query + CARD_STEP.format(guideline=card_guideline()), include_functions=False, **kani.get_hyperparams())
        add_usage(record["usage"], usage_of(written))
        record["card_text"] = written.text
        record["card"] = await parse_card(client, written.text or "")
        final = await assess(kani, ASSESS_STEP, record)
    elif mode == "none":
        final = await assess(kani, query, record)
    else:
        final = await assess(kani, query + cardmode().frozen_card_instruction(card), record)
    match = VERDICT.search(final)
    record["final_text"] = final
    record["verdict"] = int(match.group(1)) if match else None
    record["seconds"] = round(time.time() - record.pop("started"), 1)
    return record


def run_path(mode: str, claim_id: str, rep: int) -> Path:
    return RUNS / mode / claim_id / f"r{rep}.json"


def frozen_card(claim_id: str) -> dict[str, Any] | None:
    path = run_path("full", claim_id, 1)
    return json.loads(path.read_text())["card"] if path.exists() else None


async def edited_card(mode: str, claim_id: str, claim: dict[str, Any], client: AsyncOpenAI) -> dict[str, Any] | None:
    card = frozen_card(claim_id)
    if card is None:
        return None
    edited = dict(card)
    if mode == "edit-quantifier":
        if card.get("Quantifier") == "universal":
            return None
        edited["Quantifier"] = "universal"
        return edited
    response = await client.chat.completions.create(
        model=BACKBONE_MODEL,
        messages=[
            {
                "role": "user",
                "content": threshold_prompt().format(
                    claim=claim["claim"],
                    threshold=card.get("Threshold"),
                    observable=card.get("ObservableFoM"),
                    units=card.get("Units"),
                ),
            }
        ],
        max_completion_tokens=2000,
    )
    new_threshold = (response.choices[0].message.content or "").strip()
    if not new_threshold or new_threshold == card.get("Threshold"):
        return None
    edited["Threshold"] = new_threshold
    return edited


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--claims", nargs="+", default=CLAIM_IDS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--edit-file", type=Path, default=None)
    parser.add_argument("--source", choices=("claimspy", "drop4"), default="claimspy")
    args = parser.parse_args()

    claims = load_drop4() if args.source == "drop4" else load_claims()
    if args.source == "drop4" and args.claims == CLAIM_IDS:
        args.claims = sorted(claims)
    client = AsyncOpenAI()
    reps = 1 if args.mode.startswith("edit") else args.reps
    rep_numbers = [args.rep] if args.rep is not None else list(range(1, reps + 1))
    semaphore = asyncio.Semaphore(args.concurrency)

    if args.mode == "edit":
        edits = [e for e in json.loads(args.edit_file.read_text()) if e["claim"] in args.claims]

        async def edit_job(edit: dict[str, Any]) -> None:
            mode = f"edit-{edit['name']}"
            path = run_path(mode, edit["claim"], 1)
            base = frozen_card(edit["claim"])
            if path.exists() or base is None:
                return
            card = {**base, **edit["fields"]}
            async with semaphore:
                try:
                    record = await run_one(edit["claim"], claims[edit["claim"]], mode, 1, card, client)
                except Exception as error:
                    print(f"fail {mode} {edit['claim']}: {type(error).__name__}: {error}")
                    return
            record["edit_fields"] = edit["fields"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2))
            print(f"done {mode} {edit['claim']} r1: verdict={record['verdict']} gold={record['gold']} tools={len(record['tool_calls'])} {record['seconds']}s usage={record['usage']}")

        await asyncio.gather(*(edit_job(e) for e in edits))
        return

    async def job(claim_id: str, rep: int) -> None:
        path = run_path(args.mode, claim_id, rep)
        if path.exists():
            return
        card = None
        if args.mode in ("frozen", "frozen-search"):
            card = frozen_card(claim_id)
        elif args.mode.startswith("edit"):
            card = await edited_card(args.mode, claim_id, claims[claim_id], client)
        if args.mode != "full" and args.mode != "none" and card is None:
            print(f"skip {args.mode} {claim_id}: no card")
            return
        async with semaphore:
            try:
                record = await run_one(claim_id, claims[claim_id], args.mode, rep, card, client)
            except Exception as error:
                print(f"fail {args.mode} {claim_id} r{rep}: {type(error).__name__}: {error}")
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        print(f"done {args.mode} {claim_id} r{rep}: verdict={record['verdict']} gold={record['gold']} tools={len(record['tool_calls'])} {record['seconds']}s usage={record['usage']}")

    await asyncio.gather(*(job(claim_id, rep) for claim_id in args.claims for rep in rep_numbers))


if __name__ == "__main__":
    asyncio.run(main())
