from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from openai import OpenAI

from backend.models import BACKBONE_MODEL
from run_claim_cards import frozen_card

PROMPT = """A Claim Card records one interpretation of a scientific claim as fields.

Claim: {claim}

Current Claim Card (JSON):
{card}

A reader interprets the claim this way instead:
{reading}

Return a JSON object holding only the Claim Card fields whose values must change so that the card states this interpretation, each with its new value written in the style of the current card. Leave out every field that already agrees with the interpretation. Do not change Provenance."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    drop4 = {row["id"]: row for row in json.loads((ROOT / "benchmarks" / "data" / "drop4_claims.json").read_text())}
    client = OpenAI()
    edits = []
    for claim_id in args.claims:
        card = frozen_card(claim_id)
        row = drop4[claim_id]
        if card is None or not row.get("ambiguous_mappings"):
            print(f"skip {claim_id}")
            continue
        for mapping in row["ambiguous_mappings"]:
            response = client.chat.completions.create(
                model=BACKBONE_MODEL,
                messages=[{"role": "user", "content": PROMPT.format(claim=row["text"], card=json.dumps(card, indent=2, ensure_ascii=False), reading=mapping["demystified_text"])}],
                response_format={"type": "json_object"},
                reasoning_effort="none",
                max_completion_tokens=2000,
            )
            fields = {key: value for key, value in json.loads(response.choices[0].message.content).items() if key in card and key != "Provenance"}
            edits.append({"name": f"reading-{mapping['mapping_id']}", "claim": claim_id, "reading": mapping["demystified_text"], "fields": fields})
            print(f"{claim_id} {mapping['mapping_id']}: {sorted(fields)}", flush=True)
    args.out.write_text(json.dumps(edits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
