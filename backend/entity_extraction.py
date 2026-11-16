"""
Entity post-processing for NutriCheck.

Pure Python (no spaCy import) so it can be unit-tested without the model.

What this module does
---------------------
1. normalize_label()  -> maps any label spelling the model/dataset may use
                         (SINTOMA, Symptom, ALERGIAS_MEDICAMENTOSAS ...) to one of the 7 canonical labels.
2. extract_entities() -> turns spaCy entities into the API dictionary AND runs a rule-based
                         allergy extractor (with negation handling) so "medication allergy"
                         is detected even while the statistical model has no ALLERGY label.
"""
import re
import unicodedata

# Canonical label -> key used in the API / frontend
LABEL_TO_KEY = {
    "SYMPTOM": "symptoms",
    "MEDICATION": "medications",
    "TREATMENT": "treatments",
    "CONDITION": "conditions",
    "DIAGNOSIS": "diagnosis",
    "EXAM": "exams",
    "ALLERGY": "allergies",
}

CLINICAL_LABELS = ("SYMPTOM", "MEDICATION", "TREATMENT", "DIAGNOSIS", "CONDITION", "EXAM")

_LABEL_ALIASES = {
    "SYMPTOM": "SYMPTOM", "SYMPTOMS": "SYMPTOM", "SINTOMA": "SYMPTOM", "SINTOMAS": "SYMPTOM",
    "MEDICATION": "MEDICATION", "MEDICATIONS": "MEDICATION", "MEDICAMENTO": "MEDICATION",
    "MEDICAMENTOS": "MEDICATION", "MEDICACAO": "MEDICATION", "MEDICACOES": "MEDICATION",
    "TREATMENT": "TREATMENT", "TREATMENTS": "TREATMENT", "TRATAMENTO": "TREATMENT", "TRATAMENTOS": "TREATMENT",
    "DIAGNOSIS": "DIAGNOSIS", "DIAGNOSES": "DIAGNOSIS", "DIAGNOSTICO": "DIAGNOSIS", "DIAGNOSTICOS": "DIAGNOSIS",
    "CONDITION": "CONDITION", "CONDITIONS": "CONDITION", "CONDICAO": "CONDITION", "CONDICOES": "CONDITION",
    "EXAM": "EXAM", "EXAMS": "EXAM", "EXAME": "EXAM", "EXAMES": "EXAM", "EXAMINATION": "EXAM",
    "ALLERGY": "ALLERGY", "ALLERGIES": "ALLERGY", "ALERGIA": "ALLERGY", "ALERGIAS": "ALLERGY",
}


def _strip_accents_same_length(text: str) -> str:
    """Lowercase + remove accents WITHOUT changing the string length, so offsets stay valid."""
    out = []
    for ch in text:
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        base = base.lower()
        out.append(base if len(base) == 1 else ch.lower()[:1] or " ")
    return "".join(out)


def normalize_label(label: str):
    """Return the canonical label (one of LABEL_TO_KEY) or None if unknown."""
    if not label:
        return None
    key = _strip_accents_same_length(str(label)).upper()
    key = re.sub(r"[\s\-]+", "_", key).strip("_")
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    if "ALERG" in key or "ALLERG" in key:          # e.g. ALERGIAS_MEDICAMENTOSAS, MEDICATION_ALLERGY
        return "ALLERGY"
    return None


# --------------------------- allergy rules ---------------------------
_TRIGGER = re.compile(
    r"\b(alergi[ac]s?|alergic[oa]s?|hipersensibilidade|reaccao alergica|reacao alergica|anafilax\w*)\b"
)
_NEGATION = re.compile(
    r"\b(sem|nega|negou|negadas?|nao\s+(?:refere|tem|apresenta|conhece|se\s+conhece[mn]?)|desconhece|"
    r"ausencia\s+de|nenhuma|nkda|nao\s+conhecidas?)\b"
)
_NOISE_WORDS = {
    "medicamentosa", "medicamentosas", "medicamentos", "conhecida", "conhecidas", "a", "ao", "aos",
    "as", "os", "o", "de", "do", "da", "dos", "das", "nega", "negadas", "nenhuma", "sem", "nao",
    "refere", "tem", "desconhece", "ha", "registo", "registadas", "documentada", "documentadas",
}
_SEP = re.compile(r"\s*(?:,|;|/|\be\b|\bou\b)\s*")
_LEAD = re.compile(r"^\s*(?:(?:medicamentosas?|conhecidas?)\s*)*[:\-]?\s*")
_ARTICLE = re.compile(r"^(?:(?:a|ao|aos|as|os|o|de|do|da)\s+)+")
_CLAUSE_END = re.compile(r"[.;\n]")


def find_allergies(text: str):
    """
    Rule-based allergy extractor for Portuguese clinical text.
    Returns a list of dicts: {"text", "start", "end", "negated"}.
    A negated entry with text == "" means the record explicitly denies allergies.
    """
    norm = _strip_accents_same_length(text)
    found = []
    for m in _TRIGGER.finditer(norm):
        before = norm[max(0, m.start() - 40): m.start()]
        # only look back inside the same sentence
        before = _CLAUSE_END.split(before)[-1]
        c_start = m.end()
        end_match = _CLAUSE_END.search(norm, c_start)
        c_end = min(end_match.start() if end_match else len(norm), c_start + 140)
        clause = norm[c_start:c_end]
        # blank out parenthetical notes such as "(urticaria e dispneia)" (same length -> offsets stay valid)
        clause = re.sub(r"\([^)]*\)?", lambda m: " " * len(m.group()), clause)

        negated = bool(_NEGATION.search(before)) or bool(_NEGATION.search(clause[:40]))

        lead = _LEAD.match(clause)
        pos = lead.end() if lead else 0
        items = []
        for sep in list(_SEP.finditer(clause, pos)) + [None]:
            seg_end = sep.start() if sep else len(clause)
            seg = clause[pos:seg_end]
            if seg.strip():
                s_off = pos + (len(seg) - len(seg.lstrip()))
                e_off = pos + len(seg.rstrip())
                art = _ARTICLE.match(clause[s_off:e_off])
                if art:
                    s_off += art.end()
                words = re.findall(r"[a-z0-9]+", clause[s_off:e_off])
                if words and len(words) <= 6 and not all(w in _NOISE_WORDS for w in words):
                    items.append((c_start + s_off, c_start + e_off))
            if sep:
                pos = sep.end()
        if negated or not items:
            found.append({"text": "", "start": m.start(), "end": m.end(), "negated": True if negated else False,
                          "trigger_only": not negated})
        if not negated:
            for s, e in items:
                found.append({"text": text[s:e].strip(), "start": s, "end": e, "negated": False})
    return [f for f in found if not f.get("trigger_only")]


def _is_negated_context(text: str, start: int, end: int) -> bool:
    norm = _strip_accents_same_length(text)
    before = _CLAUSE_END.split(norm[max(0, start - 40): start])[-1]
    after = norm[end: end + 30]
    return bool(_NEGATION.search(before)) or bool(_NEGATION.search(after.split(".")[0]))


def _clean(value: str) -> str:
    return " ".join(value.split())


def extract_entities(ents, text: str):
    """
    ents: iterable of objects with .label_, .text, .start_char, .end_char (spaCy doc.ents).
    Returns (entities, allergy_info)

    entities     -> {"symptoms": [...], "medications": [...], ..., "allergies": [...]}
    allergy_info -> {"status": "present" | "denied" | "not_mentioned", "items": [...], "source": [...]}
    """
    entities = {key: [] for key in LABEL_TO_KEY.values()}
    seen = {key: set() for key in LABEL_TO_KEY.values()}
    denied = False
    sources = set()

    def add(key, value):
        value = _clean(value)
        norm = value.casefold()
        if value and norm not in seen[key]:
            seen[key].add(norm)
            entities[key].append(value)
            return True
        return False

    for ent in ents:
        label = normalize_label(ent.label_)
        if label is None:
            continue
        if label == "ALLERGY":
            if _is_negated_context(text, ent.start_char, ent.end_char):
                denied = True
            elif add("allergies", ent.text):
                sources.add("model")
        else:
            add(LABEL_TO_KEY[label], ent.text)

    for hit in find_allergies(text):
        if hit["negated"]:
            denied = True
        elif add("allergies", hit["text"]):
            sources.add("rules")

    if entities["allergies"]:
        status = "present"
    elif denied:
        status = "denied"
    else:
        status = "not_mentioned"

    return entities, {"status": status, "items": list(entities["allergies"]), "source": sorted(sources)}
