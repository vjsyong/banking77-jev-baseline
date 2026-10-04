#!/usr/bin/env python3
"""One-time generation of frozen CLINC150 intent descriptions (teacher call).

Fix intent descriptions BEFORE search (brief §7). Order = sorted intent names;
option id for the task question = iNNN by this order. Written with sha256.
"""
import json
import sys
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))

from teacher_client import teacher_chat_full  # noqa: E402

OUT = HERE / "data" / "clinc150" / "intent_descriptions_frozen.json"


def main():
    man = json.loads((HERE / "data" / "clinc150" / "samples" / "PREP_MANIFEST.json").read_text())
    intents = man["intents"]
    assert len(intents) == 150
    listing = "\n".join(intents)
    prompt = (
        "For each of the 150 assistant-intent names below, write a SHORT description "
        "(max 12 words) of the user's goal for that intent, as an intent-classification "
        "label. Be concrete and distinguishable from neighboring intents. "
        "Respond ONLY with a JSON object mapping each exact intent name (unchanged) to "
        "its description string.\n\n" + listing)
    resp = teacher_chat_full([{"role": "user", "content": prompt}], model="gpt-6.1-sol",
                             provider="lmuai-pro", temperature=0.0, seed=7, max_tokens=6000)
    text = resp["text"]
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    missing = [i for i in intents if i not in obj or not str(obj[i]).strip()]
    assert not missing, f"missing descriptions: {missing[:10]}"
    ordered = [(i, str(obj[i]).strip()[:120]) for i in intents]
    import hashlib
    payload = json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))
    rec = {"intents": ordered, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
           "model": "gpt-6.1-sol", "temperature": 0.0, "usage": resp.get("usage")}
    OUT.write_text(json.dumps(rec, indent=1))
    print("frozen", len(ordered), "descriptions; sha", rec["sha256"][:16])
    for name, desc in ordered[:5]:
        print(" ", name, "->", desc)


if __name__ == "__main__":
    main()
