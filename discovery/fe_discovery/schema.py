#!/usr/bin/env python3
"""Choice feature-definition schema: canonicalization, hashing, validation.

A definition: {name, question, options: [{id, definition}, ...]}
Rules (brief §6):
  - stable slug name; single short question answerable from the message alone;
  - 3-6 options, each with an explicit definition;
  - an unclear/not stated option must exist (auto-appended if absent; the
    proposal is rejected if that would exceed 6 options);
  - canonical identity = sha256 of the canonical JSON incl. option order
    (a name alone is never cache identity).
"""
import json
import re
import unicodedata
import hashlib

MAX_OPTIONS = 6
MIN_OPTIONS = 3
MAX_QUESTION_CHARS = 400
MAX_OPTION_DEF_CHARS = 300
MAX_NAME_CHARS = 60


class DefinitionError(ValueError):
    pass


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:MAX_NAME_CHARS]


_UNCLEAR_HINT = re.compile(r"unclear|not stated|unknown|none of|cannot|can't|no definite|n/?a$",
                           re.IGNORECASE)
_NEGATIVE_HINT = re.compile(r"^(no|not)[_ ]|none|absent|nothing", re.IGNORECASE)


def _is_unclear_opt(opt: dict) -> bool:
    return bool(_UNCLEAR_HINT.search(opt.get("id", "")) or
                _UNCLEAR_HINT.search(opt.get("definition", "")))


def _is_negative_opt(opt: dict) -> bool:
    return bool(_NEGATIVE_HINT.search(opt.get("id", "")) and not _is_unclear_opt(opt))


def validate_proposal(raw: dict) -> tuple[dict, list[str]]:
    """Normalize/validate a teacher proposal.

    Returns (definition, notes). Raises DefinitionError when invalid.
    Notes record normalizations applied (e.g., unclear option appended).
    """
    notes = []
    if not isinstance(raw, dict):
        raise DefinitionError("proposal is not an object")
    name = _slug(str(raw.get("name", "")))
    if not name:
        raise DefinitionError("empty name after slugging")
    question = str(raw.get("question", "")).strip()
    if not question:
        raise DefinitionError("empty question")
    if len(question) > MAX_QUESTION_CHARS:
        raise DefinitionError(f"question too long ({len(question)} chars)")
    opts_raw = raw.get("options")
    if not isinstance(opts_raw, list) or not (MIN_OPTIONS <= len(opts_raw) <= MAX_OPTIONS + 1):
        raise DefinitionError(f"options must be a list; got {type(opts_raw).__name__} "
                              f"with {len(opts_raw) if isinstance(opts_raw, list) else '-'} items")
    options, seen = [], set()
    for o in opts_raw:
        if not isinstance(o, dict):
            raise DefinitionError("option is not an object")
        oid = _slug(str(o.get("id", "")))
        ode = str(o.get("definition", "")).strip()
        if not oid:
            raise DefinitionError("empty option id")
        if not ode:
            raise DefinitionError(f"empty definition for option '{oid}'")
        if len(ode) > MAX_OPTION_DEF_CHARS:
            raise DefinitionError(f"option '{oid}' definition too long")
        if oid in seen:
            raise DefinitionError(f"duplicate option id '{oid}'")
        seen.add(oid)
        options.append({"id": oid, "definition": ode})
    if not any(_is_unclear_opt(o) for o in options):
        if len(options) >= MAX_OPTIONS:
            raise DefinitionError("no unclear option and already at option cap")
        options.append({"id": "unclear",
                        "definition": "The message does not clearly state a definite category."})
        notes.append("appended standard 'unclear' option")
    if any(_is_unclear_opt(o) for o in options) and any(_is_negative_opt(o) for o in options):
        notes.append("WARNING: both a negative-absence option and an 'unclear' option present; "
                     "audit v3 recommends ONE merged 'not present / cannot be determined' option")
    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        raise DefinitionError(f"after normalization {len(options)} options (allowed 3-6)")
    defn = {"name": name, "question": question, "options": options}
    return defn, notes


def canonical_hash(defn: dict) -> str:
    """sha256 over the canonical compact JSON incl. option order."""
    payload = json.dumps(defn, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def def_from_hash_store_key(defn: dict) -> str:
    """Store slot id: content-addressed, order-sensitive."""
    return f"fe::{canonical_hash(defn)[:20]}"


def columns_of(bank: list[dict]) -> list[tuple[str, str]]:
    """Explicit question-to-column map: (def_hash, option_id) per numeric column."""
    cols = []
    for d in bank:
        for o in d["options"]:
            cols.append((canonical_hash(d), o["id"]))
    return cols


def numeric_width(bank: list[dict]) -> int:
    return sum(len(d["options"]) for d in bank)


# ---- near-label proxy detection (logged for later content analysis) ----
def near_label_flags(defn: dict, intent_labels: list[str]) -> dict:
    """Flag options/defs whose n-grams overlap task intent labels (proxy risk)."""
    label_tokens = [set(re.findall(r"[a-z]+", L.lower())) for L in intent_labels]

    def hits(text: str) -> list[str]:
        toks = set(re.findall(r"[a-z]+", text.lower()))
        out = []
        for L, lt in zip(intent_labels, label_tokens):
            inter = toks & lt
            if len(inter) >= 2 or (len(lt) == 1 and inter == lt):
                out.append(L)
        return out

    flagged = []
    for o in defn["options"]:
        h = hits(o["id"].replace("_", " ") + " " + o["definition"])
        if h:
            flagged.append({"option": o["id"], "overlap": h[:5]})
    return {"flagged": flagged, "is_proxy_suspect": bool(flagged)}


if __name__ == "__main__":
    import sys
    # self-test
    d, notes = validate_proposal({
        "name": "Transaction State!!",
        "question": "What state of the transaction does the message describe?",
        "options": [
            {"id": "pending", "definition": "Awaiting completion or settlement."},
            {"id": "declined", "definition": "Explicitly refused or rejected."},
            {"id": "completed", "definition": "Explicitly completed or posted."},
            {"id": "reversed", "definition": "Explicitly reversed or returned."},
        ]})
    assert d["name"] == "transaction_state" and notes == ["appended standard 'unclear' option"]
    h = canonical_hash(d)
    print("self-test OK", h[:16], columns_of([d]))
    try:
        validate_proposal({"name": "x", "question": "q", "options": []})
        sys.exit("should have raised")
    except DefinitionError as e:
        print("reject-path OK:", e)
