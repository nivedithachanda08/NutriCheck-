from pathlib import Path

import joblib
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from entity_extraction import CLINICAL_LABELS, LABEL_TO_KEY, extract_entities
from translate_pt_en import translate_entities
from validators import check_input, check_medical

app = FastAPI(title="NutriCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------- NER MODEL -----------------------------
# Prefer the retrained 7-label model (made by train_ner.py); fall back to the original 6-label one.
# You can force a folder with:  set NER_MODEL_DIR=modeloNER_7_entidades   (Windows)
import os

_candidates = [os.environ.get("NER_MODEL_DIR"), "modeloNER_7_entidades", "modeloNER_6_entidades"]
NER_MODEL_PATH = next(
    (BASE_DIR / c for c in _candidates if c and (BASE_DIR / c).exists()), None
)
if NER_MODEL_PATH is None:
    raise RuntimeError(f"No NER model folder found in {BASE_DIR} (tried {[c for c in _candidates if c]})")

nlp = spacy.load(str(NER_MODEL_PATH))

# Which labels does the loaded model actually know?  (ALLERGY is also covered by rules, see entity_extraction.py)
MODEL_LABELS = sorted(nlp.get_pipe("ner").labels)
MISSING_LABELS = [l for l in (*CLINICAL_LABELS, "ALLERGY") if l not in MODEL_LABELS]
print(f"[NER] loaded {NER_MODEL_PATH.name}; labels = {MODEL_LABELS}")
if MISSING_LABELS:
    print(f"[NER] WARNING: model has no {MISSING_LABELS}. ALLERGY falls back to rules; others need retraining (train_ner.py).")

# -------------------------- ML MODEL --------------------------
# These two files were produced from the same Logistic Regression pipeline
# used in the training notebook.
TFIDF_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
ML_MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"

if not TFIDF_PATH.exists():
    raise RuntimeError(f"TF-IDF vectorizer not found: {TFIDF_PATH}")
if not ML_MODEL_PATH.exists():
    raise RuntimeError(f"Logistic Regression model not found: {ML_MODEL_PATH}")

tfidf = joblib.load(TFIDF_PATH)
model = joblib.load(ML_MODEL_PATH)

# The trained model was fitted on 3,179 TF-IDF features.
if len(tfidf.vocabulary_) != model.n_features_in_:
    raise RuntimeError(
        "TF-IDF vectorizer and Logistic Regression model are incompatible: "
        f"vectorizer={len(tfidf.vocabulary_)} features, "
        f"model={model.n_features_in_} features."
    )


class MedicalText(BaseModel):
    text: str


def reject(code: str, message: str):
    raise HTTPException(
        status_code=422,
        detail={"error": code, "message": message},
    )


def run_ner(text: str):
    """Run the NER model, then merge in the rule-based allergy extractor.
    Returns (entities, allergy_info). entities has 7 keys: the six clinical ones + 'allergies'."""
    doc = nlp(text)
    return extract_entities(doc.ents, text)


def build_ml_text(entities: dict) -> str:
    """
    Reproduce the exact text construction used by the training notebook:

        symptom + medication + treatment + diagnosis + condition + exam

    Each NER category is joined into one text string before TF-IDF.

    NOTE: 'allergies' is intentionally NOT included. The saved TF-IDF/Logistic Regression were trained on
    the six clinical columns only, and the allergy target was built from the allergy annotations, so
    feeding allergy text back in would leak the answer (and change the feature space).
    """
    columns = (
        "symptoms",
        "medications",
        "treatments",
        "diagnosis",
        "conditions",
        "exams",
    )

    parts = []
    for column in columns:
        values = entities.get(column, [])
        parts.append(" ".join(values))

    return " ".join(parts).strip()


def run_allergy_model(entities: dict):
    """Run the saved TF-IDF + Logistic Regression model."""
    ml_text = build_ml_text(entities)

    # The NER gate ensures at least one entity exists, but keep this guard
    # so the endpoint never predicts on an empty feature string.
    if not ml_text:
        reject(
            "no_features",
            "No medical entities were extracted, so a medication-allergy prediction cannot be made.",
        )

    features = tfidf.transform([ml_text])
    prediction = int(model.predict(features)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0, 1])

    return prediction, probability, ml_text


@app.get("/")
def home():
    return {
        "message": "NutriCheck Backend is running",
        "status": "OK",
        "models": {
            "ner": "Portuguese 6-entity spaCy NER",
            "classifier": "Logistic Regression + TF-IDF",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ner_loaded": nlp is not None,
        "tfidf_loaded": tfidf is not None,
        "classifier_loaded": model is not None,
        "ner_model_folder": NER_MODEL_PATH.name,
        "ner_labels": MODEL_LABELS,
        "ner_labels_missing": MISSING_LABELS,
    }


@app.post("/predict")
def predict(data: MedicalText):
    text = data.text

    # ---------------------- GATE 1 ----------------------
    # Reject empty, too-short, non-Portuguese input BEFORE NER/ML.
    problem = check_input(text)
    if problem:
        reject(*problem)

    # ----------------------- NER ------------------------
    entities, allergy_info = run_ner(text)
    n_entities = sum(len(values) for values in entities.values())

    # ---------------------- GATE 2 ----------------------
    # Reject Portuguese text that does not look like a medical record.
    problem = check_medical(text, n_entities)
    if problem:
        reject(*problem)

    # ----------------------- ML --------------------------
    medication_allergy, probability, ml_text = run_allergy_model(entities)

    # ------------------- ENGLISH OUTPUT (display only) -------------------
    # The classifier above used the ORIGINAL Portuguese entities (it was trained on Portuguese text).
    # What the user sees is translated to English; the Portuguese originals stay in "entities_pt".
    entities_en, untranslated = translate_entities(entities)
    allergy_info_en = dict(allergy_info, items=list(entities_en["allergies"]))

    return {
        "entities": entities_en,
        "entities_pt": entities,
        "untranslated_terms": [{"field": k, "term": t} for k, t in untranslated],
        "allergy_info": allergy_info_en,   # status: present / denied / not_mentioned (from the record itself)
        "medication_allergy": medication_allergy,
        "medication_allergy_label": (
            "Medication Allergy" if medication_allergy == 1 else "No Medication Allergy"
        ),
        "medication_allergy_probability": probability,
        "model_input": ml_text,
    }
