# NutriCheck — Connected NER + Logistic Regression

This version connects the existing frontend to the trained Portuguese NER model and the trained Logistic Regression medication-allergy classifier.

## Pipeline

Portuguese medical record → Portuguese/medical validation → 6-entity NER → entity text → saved TF-IDF → Logistic Regression → medication-allergy result.

## Backend

Open a terminal inside `backend`:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

Useful checks:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Frontend

Keep the backend running, then open `frontend/index.html` in the browser (or use VS Code Live Server).

## Important behavior

The `/predict` endpoint does **not** predict for:

- empty input
- very short input
- non-Portuguese input
- Portuguese text that does not look like a medical record

For valid Portuguese medical records, the endpoint runs the NER model first and then sends the six extracted entity fields to the same TF-IDF + Logistic Regression setup used for training.

The project includes the saved `tfidf_vectorizer.pkl` and `logistic_regression_model.pkl` files.

## Update: all six entities + medication allergy

Files added/changed:

- `backend/entity_extraction.py` (new) – label normalisation, merge logic, rule-based **allergy** extractor with negation ("sem alergias conhecidas").
- `backend/train_ner.py` (new) – retrains the NER model with **7 labels** (6 clinical + `ALLERGY`) and prints P/R/F1 **per label**.
- `backend/main.py` – loads `modeloNER_7_entidades` if present (else the old 6-label model), returns `allergy_info`, `/health` now lists the labels the loaded model really has.
- `frontend/` – new "Allergies (medication)" card.

Retrain (the real fix for the model returning only symptoms):

```powershell
cd backend
python train_ner.py --inspect --data path\to\dataset_with_codes.json   # check the JSON layout first
python train_ner.py --data path\to\dataset_with_codes.json
python -m uvicorn main:app --reload                                    # picks modeloNER_7_entidades automatically
```

Open `http://127.0.0.1:8000/health` and confirm `ner_labels_missing` is `[]`.

## Update: extracted information is shown in English

- `backend/translate_pt_en.py` (new) translates every extracted entity (and the allergies) to English before it is returned.
- `/predict` returns `entities` in **English**, the Portuguese originals in `entities_pt`, and `untranslated_terms` (words the glossary did not know – add them to `PHRASES` / `WORDS`).
- The Logistic Regression still receives the **Portuguese** entities (it was trained on Portuguese text), so its prediction is unchanged.
- Better translation quality (optional, offline after one download): `pip install argostranslate`, install the pt→en package, then `set NUTRICHECK_USE_ARGOS=1`. See the header of `translate_pt_en.py`.
