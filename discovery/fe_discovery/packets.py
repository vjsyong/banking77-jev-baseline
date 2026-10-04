#!/usr/bin/env python3
"""Teacher feedback packets for the four arms (brief §8).

U: task + current bank + definition history; no messages, no diagnostics.
R: 8 randomly sampled class-balanced contrastive pairs + predictions + current
   score + the same aggregate diagnostics as E (no error conditioning).
E: 5 failure contrastive pairs (mistaken message + a classmate of the confused
   class) + 3 correctly-classified preservation pairs + predictions + score +
   aggregate diagnostics; concepts must separate failures while preserving
   known distinctions.
F: E-format + measured latency limits, current bank latency, per-question cost
   estimates, and the quality-cost archive + permission for replacements.

All sampling is deterministic (seed, round); class caps with a logged fallback;
fixed text truncation and fixed packet field set. Example ids are recorded.
"""
import json
from collections import Counter, defaultdict

import numpy as np

TEXT_TRUNC = 200
MAX_OUTPUT_TOKENS = 3000


def _trunc(t):
    return t if len(t) <= TEXT_TRUNC else t[:TEXT_TRUNC] + "…"


def _class_norm_errors(y_true, y_pred, classes):
    tot = Counter(y_true)
    err = Counter()
    conf = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            err[t] += 1
            conf[(t, p)] += 1
    rates = {c: err[c] / max(tot[c], 1) for c in classes}
    return rates, conf, tot


def _worst_classes(rates, tot, k=12, min_n=8):
    ranked = sorted(rates.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [(c, round(r, 3), tot[c]) for c, r in ranked if tot[c] >= min_n][:k]
    return out


def _top_confusions(conf, k=12):
    return [[f"{a} -> {b}", n] for (a, b), n in conf.most_common(k)]


def _pair(rng, mist_idx, all_idx_by_class, y_true, y_pred, oof_proba, classes,
          correct_classes_only=True):
    """(mistaken message, distinct contrastive message from the confused class)."""
    i = mist_idx
    true_c, pred_c = y_true[i], y_pred[i]
    cands = [j for j in all_idx_by_class[pred_c] if j != i]
    if correct_classes_only:
        good = [j for j in cands if y_pred[j] == pred_c]
        if good:
            cands = good
    j = int(rng.choice(cands)) if cands else None
    return i, j


def build_packet(arm, seed, round_no, texts, y_true, y_pred, oof_proba, classes,
                 cv_score, limits=None, cost_info=None, archive=None):
    """Return (packet, meta). packet is a dict of string sections for rendering."""
    assert arm in ("U", "R", "E", "F")
    rng = np.random.RandomState(seed * 1000 + round_no)
    n = len(y_true)
    meta = {"arm": arm, "seed": seed, "round": round_no, "example_ids": [],
            "cv_score": round(float(cv_score), 4)}
    idx_by_class = defaultdict(list)
    for i, c in enumerate(y_true):
        idx_by_class[c].append(i)

    def msg_block(i):
        c = int(np.argmax(oof_proba[i]))
        top2 = np.argsort(oof_proba[i])[-2:][::-1]
        return (f'  message: "{_trunc(texts[i])}"\n'
                f'  true intent: {y_true[i]}\n'
                f'  model prediction: {y_pred[i]} '
                f'(p={oof_proba[i][c]:.2f}; runner-up {classes[top2[1]]} '
                f'p={oof_proba[i][top2[1]]:.2f})')

    if arm == "U":
        packet = {"examples": None, "diagnostics": None, "cost": None}
        return packet, meta

    rates, conf, tot = _class_norm_errors(y_true, y_pred, classes)
    diag = ("AGGREGATE DIAGNOSTICS (development folds)\n"
            f"- current cross-validated macro-F1: {cv_score:.4f}\n"
            f"- worst classes (error rate, n): {_worst_classes(rates, tot)}\n"
            f"- most frequent confusions (count): {_top_confusions(conf)}")

    if arm == "R":
        # 8 random class-balanced pairs, NOT conditioned on errors
        pairs, used_c = [], Counter()
        classes_shuffled = sorted(classes)
        rng.shuffle(classes_shuffled)
        for c in classes_shuffled:
            if len(pairs) >= 8:
                break
            if used_c[c] >= 2:
                continue
            members = idx_by_class[c]
            if len(members) < 2:
                continue
            a, b = rng.choice(members, size=2, replace=False)
            pairs.append((int(a), int(b), c))
            used_c[c] += 1
        lines = []
        for k, (a, b, c) in enumerate(pairs, 1):
            lines.append(f"PAIR {k} (class {c})\n{msg_block(a)}\n  contrasting message of the SAME class:\n{msg_block(b)}")
            meta["example_ids"] += [int(a), int(b)]
        packet = {"examples": "EXAMPLE PAIRS (random class-balanced sample)\n" + "\n\n".join(lines),
                  "diagnostics": diag, "cost": None}
        return packet, meta

    # E and F: error-driven
    errors = [i for i in range(n) if y_true[i] != y_pred[i]]
    # 5 failure pairs, class cap <=2, deterministic fallback relaxes the cap
    fail_pairs, used_c = [], Counter()
    order = list(errors)
    rng.shuffle(order)
    for cap in (2, 3, 99):
        for i in order:
            if len(fail_pairs) >= 5:
                break
            if used_c[y_true[i]] >= cap:
                continue
            if (i, y_true[i]) in [(a, y_true[a]) for a, _ in fail_pairs]:
                continue
            j = _pair(rng, i, idx_by_class, y_true, y_pred, None, classes)[1]
            if j is None:
                continue
            fail_pairs.append((i, j))
            used_c[y_true[i]] += 1
        if len(fail_pairs) >= 5:
            break
    meta["fallback_used"] = len(fail_pairs) < 5

    # 3 correctly classified preservation pairs
    correct = [i for i in range(n) if y_true[i] == y_pred[i]]
    rng.shuffle(correct)
    pres_pairs, used_c2 = [], Counter()
    for i in correct:
        if len(pres_pairs) >= 3:
            break
        if used_c2[y_true[i]] >= 1:
            continue
        members = [j for j in idx_by_class[y_true[i]] if j != i]
        if not members:
            continue
        j = int(rng.choice(members))
        pres_pairs.append((i, j))
        used_c2[y_true[i]] += 1

    lines = ["FAILURE PAIRS — the model confuses these; design concepts that separate them:"]
    for k, (i, j) in enumerate(fail_pairs, 1):
        lines.append(f"FAILURE {k}: true {y_true[i]}, predicted {y_pred[i]}\n"
                     f"{msg_block(i)}\n  contrast with a correctly handled message of the confused class:\n{msg_block(j)}")
        meta["example_ids"] += [int(i), int(j)]
    lines.append("\nPRESERVATION CONTROLS — the model already separates these; do not break them:")
    for k, (i, j) in enumerate(pres_pairs, 1):
        lines.append(f"CONTROL {k}\n{msg_block(i)}\n  distinct correct message:\n{msg_block(j)}")
        meta["example_ids"] += [int(i), int(j)]
    examples = "\n\n".join(lines)

    cost = None
    if arm == "F":
        cost_lines = []
        if limits:
            cost_lines.append("SERVING COST — absolute limits for this deployment (batched, ms/text): "
                              + json.dumps(limits))
        if cost_info:
            cost_lines.append(f"current bank measured latency: {cost_info.get('current_latency_ms')} ms/text "
                              f"(questions: {cost_info.get('n_questions')}, numeric width: "
                              f"{cost_info.get('n_columns')})")
            cost_lines.append("per-question cost estimates (question, options, est. ms/text):")
            for q in cost_info.get("per_question", []):
                cost_lines.append(f"  - {q['name']}: {q['n_options']} options, {q['est_ms']} ms")
        if archive:
            cost_lines.append("QUALITY-COST ARCHIVE (entries: definitions, CV macro-F1, ms/text, cumulative discovery cost):")
            for e in archive[-8:]:
                cost_lines.append(f"  - {e}")
        cost_lines.append("You may propose REPLACEMENTS or cheaper alternatives to any current question.")
        cost = "\n".join(cost_lines)

    packet = {"examples": examples, "diagnostics": diag, "cost": cost}
    return packet, meta


def render_prompt(arm, bank_defs, memory_names, packet, task_blurb, slots=12):
    """Assemble the full teacher prompt (single template; arm-specific sections)."""
    parts = [task_blurb]
    parts.append(
        "Each proposal must be a schema-bound categorical question with:\n"
        "- a short snake_case name;\n"
        "- one question answerable from the message alone, about ONE concept;\n"
        "- 3 to 6 options, each with an explicit definition; include an "
        "'unclear' option for when the message does not state a definite category.\n"
        "*Never use the true intent label in the question or options.*\n"
        "Prefer concepts that can distinguish MULTIPLE intents and complement lexical signals."
    )
    cur = "CURRENT SEMANTIC BANK (selected questions):\n"
    if bank_defs:
        for d in bank_defs:
            opts = "; ".join(f"{o['id']}" for o in d["options"])
            cur += f"- {d['name']}: {d['question']} [options: {opts}]\n"
    else:
        cur += "(empty — textual TF-IDF is the only representation so far)\n"
    parts.append(cur)
    if memory_names:
        parts.append("ALREADY PROPOSED NAMES (do not duplicate; revise by proposing a new name):\n"
                     + ", ".join(sorted(set(memory_names))))
    if packet["examples"]:
        parts.append(packet["examples"])
    if packet["diagnostics"]:
        parts.append(packet["diagnostics"])
    if arm == "F" and packet["cost"]:
        parts.append(packet["cost"])
    parts.append(
        f"Propose up to {slots} NEW definitions that would most improve the downstream "
        "macro-F1 if added. Respond with ONLY a JSON array of objects with keys "
        "'name', 'question', 'options' (list of {id, definition})."
    )
    return "\n\n".join(parts)


def task_blurb():
    return (
        "TASK: You are designing interpretable semantic features for a text classifier.\n"
        "Dataset: short user messages mapped to 150 assistant intents (CLINC150).\n"
        "Pipeline: each proposed question is answered for every message by a small local "
        "model (TinyJev). Its per-option probability distribution becomes numeric features "
        "for a logistic regression classifier, which also uses TF-IDF text features.\n"
        "Goal: discover categorical concepts that separate intents the classifier currently "
        "confuses, complementing (not duplicating) lexical signals.\n"
        "CONVENTION (settled in the development instrument audit): include exactly ONE option "
        "covering 'not present or cannot be determined' (e.g., 'unclear'). Do NOT create "
        "separate 'no_X' and 'unclear' options — the extractor cannot reliably distinguish "
        "an explicitly-absent concept from an undeterminable one."
    )
