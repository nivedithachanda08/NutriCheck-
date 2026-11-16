"""
Retrain the Portuguese NER model with SEVEN labels:
SYMPTOM, MEDICATION, TREATMENT, DIAGNOSIS, CONDITION, EXAM + ALLERGY

Usage (from the backend folder):
    python train_ner.py --data path/to/dataset_with_codes.json
    python train_ner.py --data dataset_with_codes.json --epochs 60 --out modeloNER_7_entidades

Why the old model only returned SYMPTOM (things this script fixes):
  1. Overlapping annotations: spaCy NER cannot store two labels on the same words. The usual
     "keep the longest span" rule silently throws away rarer labels. Here the RARER label wins.
  2. Misaligned offsets: char_span(..., strict) returns None and the span vanishes. Here we use
     alignment_mode="expand" and report how many spans were repaired / lost.
  3. Class imbalance: SYMPTOM is the most frequent label, so a small from-scratch model learns to
     predict it and ignore the rest. Here records that contain rare labels are oversampled.
  4. No per-label evaluation: a single overall F1 hides a label with F1 = 0. Here P/R/F is printed
     for every label after training, and a warning is raised for any label under 0.30 F1.

The ONLY dataset-specific part is load_records(); if your JSON layout is different, run with
--inspect to print its structure and adjust TEXT_KEYS / ANNOT_KEYS / LABEL_KEYS below.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from entity_extraction import normalize_label, CLINICAL_LABELS

TEXT_KEYS = ("text", "texto", "record", "registo", "content", "note", "clinical_text")
ANNOT_KEYS = ("entities", "annotations", "labels", "spans", "ents", "entidades", "anotacoes", "anotações", "label")
LABEL_KEYS = ("label", "type", "category", "entity", "tag", "categoria", "classe", "entity_group", "name")
START_KEYS = ("start", "start_offset", "begin", "start_char", "inicio")
END_KEYS = ("end", "end_offset", "stop", "end_char", "fim")
VALUE_KEYS = ("text", "value", "span", "mention", "texto")


# ------------------------------ data loading ------------------------------
def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _find_record_list(data):
    """Return the list of records, wherever it is nested."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [data]
    raise ValueError("Unrecognised JSON root")


def _parse_annotation(a, text):
    """-> (start, end, raw_label) or None."""
    if isinstance(a, (list, tuple)) and len(a) >= 3:
        return int(a[0]), int(a[1]), str(a[2])
    if not isinstance(a, dict):
        return None
    # Label Studio style: {"value": {"start":..,"end":..,"labels":["X"]}}
    inner = a.get("value") if isinstance(a.get("value"), dict) else a
    raw_label = _first(inner, LABEL_KEYS)
    if raw_label is None and isinstance(inner.get("labels"), list) and inner["labels"]:
        raw_label = inner["labels"][0]
    start, end = _first(inner, START_KEYS), _first(inner, END_KEYS)
    if start is None or end is None:
        val = _first(inner, VALUE_KEYS)
        if isinstance(val, str) and val:
            i = text.find(val)
            if i >= 0:
                start, end = i, i + len(val)
    if start is None or end is None or raw_label is None:
        return None
    return int(start), int(end), str(raw_label)


def load_records(path):
    """-> list of (text, [(start, end, canonical_label), ...]) and a Counter of unmapped raw labels."""
    data = json.load(open(path, encoding="utf-8"))
    records, unmapped, skipped = [], Counter(), 0
    for rec in _find_record_list(data):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        text = _first(rec, TEXT_KEYS)
        annots = _first(rec, ANNOT_KEYS)
        if not isinstance(text, str) or not isinstance(annots, list):
            skipped += 1
            continue
        spans = []
        for a in annots:
            parsed = _parse_annotation(a, text)
            if parsed is None:
                continue
            s, e, raw = parsed
            label = normalize_label(raw)
            if label is None:
                unmapped[raw] += 1
                continue
            if 0 <= s < e <= len(text):
                spans.append((s, e, label))
        records.append((text, spans))
    if skipped:
        print(f"[data] skipped {skipped} records without text/annotations")
    return records, unmapped


def resolve_overlaps(spans, label_freq):
    """Keep non-overlapping spans. When two overlap, the RARER label wins (then the longer span)."""
    order = sorted(spans, key=lambda x: (label_freq[x[2]], -(x[1] - x[0]), x[0]))
    kept, dropped = [], Counter()
    for s, e, l in order:
        if any(s < ke and ks < e for ks, ke, _ in kept):
            dropped[l] += 1
        else:
            kept.append((s, e, l))
    return sorted(kept), dropped


def oversample(train, label_freq, target_ratio=0.25, max_repeat=6):
    """Repeat records that contain rare labels so each label reaches ~target_ratio of the top label."""
    top = max(label_freq.values())
    out = []
    for text, spans in train:
        labels = {l for _, _, l in spans}
        rep = 1
        for l in labels:
            if label_freq[l] < top * target_ratio:
                rep = max(rep, min(max_repeat, round(top * target_ratio / max(label_freq[l], 1))))
        out.extend([(text, spans)] * rep)
    return out


# ------------------------------ training ------------------------------
def make_docs(nlp, records):
    from spacy.tokens import Doc  # noqa: F401
    docs, lost = [], Counter()
    for text, spans in records:
        doc = nlp.make_doc(text)
        ents = []
        for s, e, l in spans:
            span = doc.char_span(s, e, label=l, alignment_mode="expand")
            if span is None:
                lost[l] += 1
            else:
                ents.append(span)
        # expansion can create new overlaps -> filter again, rarer labels first is already handled upstream
        from spacy.util import filter_spans
        doc.ents = filter_spans(ents)
        docs.append(doc)
    return docs, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="modeloNER_7_entidades")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true", help="print the JSON structure and exit")
    args = ap.parse_args()

    if args.inspect:
        data = json.load(open(args.data, encoding="utf-8"))
        recs = _find_record_list(data)
        print("records:", len(recs))
        print(json.dumps(recs[0], ensure_ascii=False, indent=2)[:3000])
        return

    records, unmapped = load_records(args.data)
    if not records:
        sys.exit("No usable records. Run with --inspect and adjust TEXT_KEYS / ANNOT_KEYS.")
    raw_freq = Counter(l for _, spans in records for _, _, l in spans)
    print("[data] label counts in raw annotations:", dict(raw_freq))
    if unmapped:
        print("[data] WARNING unmapped raw labels (add them to _LABEL_ALIASES in entity_extraction.py):", dict(unmapped))
    missing = [l for l in (*CLINICAL_LABELS, "ALLERGY") if raw_freq[l] == 0]
    if missing:
        print("[data] WARNING no annotations at all for:", missing)

    resolved, total_dropped = [], Counter()
    for text, spans in records:
        kept, dropped = resolve_overlaps(spans, raw_freq)
        total_dropped.update(dropped)
        resolved.append((text, kept))
    print("[data] spans dropped because of overlap:", dict(total_dropped))

    random.seed(args.seed)
    random.shuffle(resolved)
    n = len(resolved)
    n_dev, n_test = max(1, int(n * 0.1)), max(1, int(n * 0.1))
    test, dev, train = resolved[:n_test], resolved[n_test:n_test + n_dev], resolved[n_test + n_dev:]
    freq = Counter(l for _, spans in train for _, _, l in spans)
    train_os = oversample(train, freq)
    print(f"[data] records train/dev/test = {len(train)}/{len(dev)}/{len(test)}; after oversampling train = {len(train_os)}")

    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding

    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner")
    for label in (*CLINICAL_LABELS, "ALLERGY"):
        ner.add_label(label)

    train_docs, lost_tr = make_docs(nlp, train_os)
    dev_docs, _ = make_docs(nlp, dev)
    test_docs, _ = make_docs(nlp, test)
    if lost_tr:
        print("[data] spans lost in alignment (train):", dict(lost_tr))

    train_ex = [Example(nlp.make_doc(d.text), d) for d in train_docs]
    dev_ex = [Example(nlp.make_doc(d.text), d) for d in dev_docs]
    test_ex = [Example(nlp.make_doc(d.text), d) for d in test_docs]

    nlp.initialize(lambda: train_ex)
    best, best_f, patience = None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_ex)
        losses = {}
        for batch in minibatch(train_ex, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(batch, drop=args.dropout, losses=losses)
        scores = nlp.evaluate(dev_ex)
        f = scores["ents_f"] or 0.0
        print(f"epoch {epoch:02d}  loss={losses.get('ner', 0):.1f}  dev F1={f:.3f}")
        if f > best_f:
            best_f, patience = f, 0
            nlp.to_disk(args.out)
        else:
            patience += 1
            if patience >= 8:
                print("early stop")
                break

    best = spacy.load(args.out)
    scores = best.evaluate(test_ex)
    print("\n=== TEST RESULTS (per label) ===")
    for label, s in sorted(scores["ents_per_type"].items()):
        flag = "   <-- LOW" if s["f"] < 0.30 else ""
        print(f"{label:12s} P={s['p']:.2f} R={s['r']:.2f} F1={s['f']:.2f}{flag}")
    print(f"{'OVERALL':12s} P={scores['ents_p']:.2f} R={scores['ents_r']:.2f} F1={scores['ents_f']:.2f}")
    print(f"\nSaved to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
