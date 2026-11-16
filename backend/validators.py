"""
Input validation for NutriCheck.

Pure Python (no extra dependencies). Two gates:
  1. check_input()   -> is the text usable and written in Portuguese?  (runs BEFORE the NER model)
  2. check_medical() -> does it look like a medical record?            (runs AFTER the NER model)
"""
import re
import unicodedata

# ----------------------------- Settings -----------------------------
MIN_WORDS = 4              # shorter than this is not a medical record
MAX_CHARS = 30_000         # protects the server from huge inputs
MIN_LANG_HITS = 2          # minimum Portuguese-only words needed
MIN_MEDICAL_TERMS = 2      # distinct medical words required...
MIN_ENTITIES = 1           # ...and at least this many entities found by the NER model

MESSAGES = {
    "empty": "Please enter a medical record before analyzing.",
    "too_short": "The text is too short to be a medical record. Please paste a complete medical record written in Portuguese.",
    "too_long": f"The text is too long (maximum {MAX_CHARS:,} characters).",
    "not_portuguese": "This text does not appear to be written in Portuguese. NutriCheck only accepts medical records in Portuguese.",
    "not_medical": "This text does not look like a medical record. Please enter a clinical record (symptoms, medications, treatments, diagnoses, exams...) written in Portuguese.",
}


def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'vómitos' == 'vomitos' (your dataset mixes both)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str):
    return re.findall(r"[a-z]+", _normalize(text))


# ------------------------- Language detection -------------------------
_PT = set("""
o os as e de do da dos das em no na nos nas um uma uns umas para com nao que ao aos foi foram ser
tem teve tinha ha sem pelo pela pelos pelas ou mais ja esta estao sao se por como mas seu sua seus suas
ele ela eles elas apos ate sobre entre desde ainda muito tambem quando onde isto isso aquele
doente doentes dor dores febre vomitos tosse tratamento exames analises medicacao internamento
urgencia servico enfermaria antecedentes historia queixa queixas sintomas hipertensao doenca
realizou realizado apresenta apresentou refere referiu sexo masculino feminino anos filha filho
""".split())

_ES = set("""
el los las y en con del una unos unas para por que lo al su sus es son fue fueron ha han
como pero sin sobre entre desde tambien cuando donde este esta estos estas
paciente pacientes dolor fiebre tos tratamiento examenes analisis medicacion ingreso urgencias
servicio antecedentes historia sintomas hipertension enfermedad realizo presenta refiere sexo
masculino femenino anos hija hijo
""".split())

_EN = set("""
the and of to in is are was were with for at on by from he she his her this that these those
has have had been not no patient patients pain fever cough treatment tests medication admission
emergency department history symptoms hypertension disease male female years old daughter son
presented complains reports denies
""".split())

# Words shared by several languages carry no information -> drop them from every list.
_PT, _ES, _EN = _PT - _ES - _EN, _ES - _PT - _EN, _EN - _PT - _ES


def detect_language_scores(text: str) -> dict:
    toks = _tokens(text)
    return {
        "pt": sum(t in _PT for t in toks),
        "es": sum(t in _ES for t in toks),
        "en": sum(t in _EN for t in toks),
        "n_tokens": len(toks),
    }


def is_portuguese(text: str) -> bool:
    s = detect_language_scores(text)
    return s["pt"] >= MIN_LANG_HITS and s["pt"] > s["es"] and s["pt"] > s["en"]


# --------------------------- Medical check ---------------------------
# Short words must match exactly; long stems match as a prefix (so "hipertens" hits "hipertensao").
_MEDICAL_EXACT = {
    "dor", "dores", "tosse", "mg", "ecg", "rx", "tac", "rm", "dose", "sinais", "edema", "febre",
    "queixa", "queixas", "urgencia", "sutura", "gripe", "asma", "avc", "dpoc", "ic", "hta", "dm",
}
_MEDICAL_PREFIX = (
    "doent", "pacient", "diagnos", "tratament", "terapeut", "medicac", "medicament", "antibiot",
    "analges", "exame", "analis", "ecograf", "radiograf", "tomograf", "ressonanc", "eletrocard",
    "electrocard", "hemograma", "glicemi", "hipertens", "diabet", "insulin", "metformin", "alergi",
    "internament", "enfermar", "hospital", "clinic", "sintoma", "cirurg", "cardio", "pulmon",
    "renal", "hepat", "gastr", "abdomin", "toraci", "cefale", "dispne", "infec", "inflam",
    "doenca", "patolog", "antecedent", "admissao", "comprimid", "prescri", "posologia", "vomit",
    "febr", "arterial", "anemia", "colesterol", "neoplas", "tumor", "fratura", "pneumon",
    "bronqu", "oncolog", "neurolog", "ortoped", "pediatr", "obstetr", "ginecolog", "farmac",
    "ecocardio", "endoscop", "bioquim", "hemoglobin", "leucocit", "plaquet", "creatinin",
)


def medical_terms_found(text: str) -> set:
    found = set()
    for t in _tokens(text):
        if t in _MEDICAL_EXACT or t.startswith(_MEDICAL_PREFIX):
            found.add(t)
    return found


# ------------------------------ Gates ------------------------------
def check_input(text: str):
    """Returns None if OK, otherwise (error_code, message). Runs before the NER model."""
    if not text or not text.strip():
        return "empty", MESSAGES["empty"]
    if len(text) > MAX_CHARS:
        return "too_long", MESSAGES["too_long"]
    if len(_tokens(text)) < MIN_WORDS:
        return "too_short", MESSAGES["too_short"]
    if not is_portuguese(text):
        return "not_portuguese", MESSAGES["not_portuguese"]
    return None


def check_medical(text: str, n_entities: int):
    """Returns None if OK, otherwise (error_code, message). Runs after the NER model."""
    if n_entities >= MIN_ENTITIES and len(medical_terms_found(text)) >= MIN_MEDICAL_TERMS:
        return None
    return "not_medical", MESSAGES["not_medical"]
