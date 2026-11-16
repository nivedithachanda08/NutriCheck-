"""
Portuguese -> English translation of the EXTRACTED entities (display only).

How it works (no internet needed):
  1. exact phrase lookup in PHRASES   ("dor abdominal" -> "abdominal pain")
  2. optional offline neural translation with Argos Translate (only if you enable it, see below)
  3. word-by-word glossary fallback with noun/adjective re-ordering
  Anything the glossary does not know is kept as written and reported in `untranslated`,
  so you can add it to PHRASES / WORDS. Drug names that are identical in both languages need no entry.

The classifier (TF-IDF + Logistic Regression) was trained on Portuguese text, so main.py keeps feeding it
the ORIGINAL Portuguese entities; only what is shown to the user is translated.

Optional better quality (offline after a one-time download):
    pip install argostranslate
    python -c "import argostranslate.package as p; p.update_package_index(); a=[x for x in p.get_available_packages() if x.from_code=='pt' and x.to_code=='en'][0]; p.install_from_path(a.download())"
    set NUTRICHECK_USE_ARGOS=1
"""
import os
import re
import unicodedata

# ----------------------------- phrases (normalised: lowercase, no accents) -----------------------------
PHRASES = {
    # symptoms
    "dor abdominal": "abdominal pain", "dor toracica": "chest pain", "dor no peito": "chest pain",
    "dor de cabeca": "headache", "dor de garganta": "sore throat", "dor lombar": "lower back pain",
    "dor de costas": "back pain", "dor de dentes": "toothache", "dor de ouvidos": "earache",
    "dores abdominais": "abdominal pain", "dores musculares": "muscle aches", "dores articulares": "joint pain",
    "falta de ar": "shortness of breath", "tosse seca": "dry cough", "tosse produtiva": "productive cough",
    "perda de peso": "weight loss", "aumento de peso": "weight gain", "perda de apetite": "loss of appetite",
    "perda de consciencia": "loss of consciousness", "perda de forca": "loss of strength",
    "mal estar": "malaise", "mal-estar": "malaise", "mal estar geral": "general malaise",
    "nauseas e vomitos": "nausea and vomiting", "febre alta": "high fever", "febre baixa": "low-grade fever",
    "edema dos membros inferiores": "lower limb edema", "pernas inchadas": "swollen legs",
    "batimentos cardiacos acelerados": "rapid heartbeat", "pressao alta": "high blood pressure",
    "tensao arterial alta": "high blood pressure", "visao turva": "blurred vision",
    "dificuldade respiratoria": "breathing difficulty", "dificuldade em respirar": "difficulty breathing",
    "sangue nas fezes": "blood in stool", "sangue na urina": "blood in urine", "dor ao urinar": "pain on urination",
    # conditions / diagnoses
    "hipertensao arterial": "hypertension", "hipertensao": "hypertension", "hta": "hypertension",
    "diabetes mellitus": "diabetes mellitus", "diabetes mellitus tipo 2": "type 2 diabetes mellitus",
    "diabetes mellitus tipo 1": "type 1 diabetes mellitus", "diabetes tipo 2": "type 2 diabetes",
    "diabetes tipo 1": "type 1 diabetes", "dm": "diabetes mellitus", "dm2": "type 2 diabetes mellitus",
    "insuficiencia cardiaca": "heart failure", "insuficiencia cardiaca cronica": "chronic heart failure", "insuficiencia renal": "renal failure",
    "insuficiencia renal cronica": "chronic renal failure", "insuficiencia respiratoria": "respiratory failure",
    "doenca renal cronica": "chronic kidney disease",
    "doenca pulmonar obstrutiva cronica": "chronic obstructive pulmonary disease", "dpoc": "COPD",
    "acidente vascular cerebral": "stroke", "avc": "stroke", "enfarte agudo do miocardio": "acute myocardial infarction",
    "enfarte do miocardio": "myocardial infarction", "fibrilhacao auricular": "atrial fibrillation",
    "infecao urinaria": "urinary tract infection", "infecao respiratoria": "respiratory infection",
    "infecao das vias respiratorias": "respiratory tract infection", "litiase renal": "kidney stones",
    "choque septico": "septic shock", "cancro da mama": "breast cancer", "cancro do pulmao": "lung cancer",
    "cancro do colon": "colon cancer", "refluxo gastroesofagico": "gastroesophageal reflux",
    "doenca de crohn": "Crohn's disease", "esclerose multipla": "multiple sclerosis",
    # exams
    "analises": "blood tests", "analises clinicas": "laboratory tests", "analises sanguineas": "blood tests",
    "exames laboratoriais": "laboratory tests", "exame laboratorial": "laboratory test",
    "exames complementares": "complementary tests", "exame objetivo": "physical examination",
    "exame neurologico": "neurological examination", "exame fisico": "physical examination",
    "hemograma": "complete blood count", "hemograma completo": "complete blood count",
    "ecg": "ECG", "eletrocardiograma": "electrocardiogram", "electrocardiograma": "electrocardiogram",
    "raio x": "X-ray", "raio-x": "X-ray", "rx": "X-ray", "radiografia": "X-ray",
    "radiografia do torax": "chest X-ray", "rx do torax": "chest X-ray", "rx torax": "chest X-ray",
    "tac": "CT scan", "tc": "CT scan", "tomografia computorizada": "CT scan", "tomografia computadorizada": "CT scan",
    "tac cerebral": "brain CT scan", "tac toraco-abdomino-pelvica": "chest-abdomen-pelvis CT scan",
    "ressonancia magnetica": "MRI", "rm": "MRI", "ecografia": "ultrasound", "ecografia abdominal": "abdominal ultrasound",
    "ecocardiograma": "echocardiogram", "endoscopia": "endoscopy", "endoscopia digestiva alta": "upper endoscopy",
    "colonoscopia": "colonoscopy", "gasimetria": "blood gas analysis", "gasometria": "blood gas analysis",
    "glicemia": "blood glucose", "glicemia capilar": "capillary blood glucose", "sumaria de urina": "urinalysis",
    "urina tipo ii": "urinalysis", "cultura de urina": "urine culture", "urocultura": "urine culture",
    "hemoculturas": "blood cultures", "proteina c reativa": "C-reactive protein", "pcr": "C-reactive protein",
    "auscultacao cardiopulmonar": "cardiopulmonary auscultation", "bioquimica": "blood chemistry",
    "funcao renal": "renal function", "funcao hepatica": "liver function", "ionograma": "electrolytes panel",
    "eletroencefalograma": "electroencephalogram", "espirometria": "spirometry", "mamografia": "mammography",
    # treatments
    "cirurgia": "surgery", "fisioterapia": "physiotherapy", "quimioterapia": "chemotherapy",
    "radioterapia": "radiotherapy", "hemodialise": "hemodialysis", "dialise": "dialysis",
    "oxigenoterapia": "oxygen therapy", "antibioterapia": "antibiotic therapy", "antibiotico": "antibiotic",
    "antibioticos": "antibiotics", "analgesico": "analgesic", "analgesicos": "analgesics",
    "anti-inflamatorio": "anti-inflammatory", "anti inflamatorio": "anti-inflammatory",
    "antipiretico": "antipyretic", "anticoagulante": "anticoagulant", "anticoagulacao": "anticoagulation",
    "anti-hipertensor": "antihypertensive", "anti-hipertensivo": "antihypertensive", "diuretico": "diuretic",
    "broncodilatador": "bronchodilator", "broncodilatadores": "bronchodilators", "corticoide": "corticosteroid",
    "corticoides": "corticosteroids", "corticosteroide": "corticosteroid", "insulinoterapia": "insulin therapy",
    "fluidoterapia": "fluid therapy", "soroterapia": "fluid therapy", "hidratacao": "hydration",
    "sutura": "suture", "internamento": "hospital admission", "dieta": "diet", "repouso": "rest",
    "drenagem": "drainage", "transfusao": "transfusion", "reabilitacao": "rehabilitation",
    "apendicectomia": "appendectomy", "colecistectomia": "cholecystectomy", "analgesia": "analgesia",
    "ventilacao mecanica": "mechanical ventilation", "ventilacao nao invasiva": "non-invasive ventilation",
    "vacinacao": "vaccination", "cuidados intensivos": "intensive care", "terapia de substituicao": "replacement therapy",
    # allergens / classes
    "contraste iodado": "iodinated contrast", "acido acetilsalicilico": "acetylsalicylic acid (aspirin)",
    "aas": "aspirin (ASA)", "anti-inflamatorios nao esteroides": "non-steroidal anti-inflammatory drugs (NSAIDs)",
    "aines": "NSAIDs", "sulfamidas": "sulfonamides", "beta-lactamicos": "beta-lactams", "latex": "latex",
    "frutos secos": "tree nuts", "marisco": "shellfish", "mariscos": "shellfish", "ovo": "egg", "leite": "milk",
}

# ----------------------------- words: normalised -> (english, part-of-speech) -----------------------------
# pos: n = noun, a = adjective (PT puts adjectives after the noun, EN before), None = other
WORDS = {
    # nouns - symptoms
    "dor": ("pain", "n"), "dores": ("pain", "n"), "febre": ("fever", "n"), "tosse": ("cough", "n"),
    "vomitos": ("vomiting", "n"), "vomito": ("vomiting", "n"), "nauseas": ("nausea", "n"), "diarreia": ("diarrhea", "n"),
    "obstipacao": ("constipation", "n"), "cansaco": ("fatigue", "n"), "astenia": ("asthenia", "n"),
    "cefaleia": ("headache", "n"), "cefaleias": ("headache", "n"), "tonturas": ("dizziness", "n"), "tontura": ("dizziness", "n"),
    "vertigens": ("vertigo", "n"), "dispneia": ("dyspnea", "n"), "edema": ("edema", "n"), "edemas": ("edema", "n"),
    "prurido": ("itching", "n"), "exantema": ("rash", "n"), "erupcao": ("rash", "n"), "urticaria": ("hives", "n"),
    "dormencia": ("numbness", "n"), "palpitacoes": ("palpitations", "n"), "calafrios": ("chills", "n"),
    "sudorese": ("sweating", "n"), "anorexia": ("anorexia", "n"), "insonia": ("insomnia", "n"), "sonolencia": ("drowsiness", "n"),
    "confusao": ("confusion", "n"), "convulsoes": ("seizures", "n"), "sincope": ("syncope", "n"), "hemoptises": ("hemoptysis", "n"),
    "odinofagia": ("odynophagia", "n"), "disfagia": ("dysphagia", "n"), "epistaxis": ("nosebleed", "n"),
    "queixas": ("complaints", "n"), "queixa": ("complaint", "n"), "inchaço": ("swelling", "n"), "inchaco": ("swelling", "n"),
    "fraqueza": ("weakness", "n"), "rigidez": ("stiffness", "n"), "congestao": ("congestion", "n"),
    # nouns - conditions / procedures
    "infecao": ("infection", "n"), "insuficiencia": ("failure", "n"), "doenca": ("disease", "n"), "fratura": ("fracture", "n"),
    "cancro": ("cancer", "n"), "neoplasia": ("neoplasm", "n"), "tumor": ("tumor", "n"), "gastrite": ("gastritis", "n"),
    "pneumonia": ("pneumonia", "n"), "bronquite": ("bronchitis", "n"), "asma": ("asthma", "n"), "anemia": ("anemia", "n"),
    "obesidade": ("obesity", "n"), "dislipidemia": ("dyslipidemia", "n"), "depressao": ("depression", "n"),
    "ansiedade": ("anxiety", "n"), "demencia": ("dementia", "n"), "epilepsia": ("epilepsy", "n"), "artrose": ("osteoarthritis", "n"),
    "artrite": ("arthritis", "n"), "osteoporose": ("osteoporosis", "n"), "gripe": ("influenza", "n"), "tuberculose": ("tuberculosis", "n"),
    "apendicite": ("appendicitis", "n"), "colecistite": ("cholecystitis", "n"), "pancreatite": ("pancreatitis", "n"),
    "hepatite": ("hepatitis", "n"), "cirrose": ("cirrhosis", "n"), "hipotiroidismo": ("hypothyroidism", "n"),
    "hipertiroidismo": ("hyperthyroidism", "n"), "sepsis": ("sepsis", "n"), "rinite": ("rhinitis", "n"), "sinusite": ("sinusitis", "n"),
    "otite": ("otitis", "n"), "amigdalite": ("tonsillitis", "n"), "faringite": ("pharyngitis", "n"), "dermatite": ("dermatitis", "n"),
    "colite": ("colitis", "n"), "gota": ("gout", "n"), "enxaqueca": ("migraine", "n"), "hemorragia": ("hemorrhage", "n"),
    "tratamento": ("treatment", "n"), "terapia": ("therapy", "n"), "terapeutica": ("therapy", "n"), "exame": ("examination", "n"),
    "exames": ("tests", "n"), "analise": ("test", "n"), "alergia": ("allergy", "n"), "alergias": ("allergies", "n"),
    "reacao": ("reaction", "n"), "pressao": ("pressure", "n"), "tensao": ("pressure", "n"), "cirurgia": ("surgery", "n"),
    "nodulo": ("nodule", "n"), "quisto": ("cyst", "n"), "ulcera": ("ulcer", "n"), "lesao": ("lesion", "n"),
    "inflamacao": ("inflammation", "n"), "estenose": ("stenosis", "n"), "trombose": ("thrombosis", "n"), "embolia": ("embolism", "n"),
    # adjectives
    "abdominal": ("abdominal", "a"), "toracica": ("chest", "a"), "toracico": ("chest", "a"), "lombar": ("lumbar", "a"),
    "aguda": ("acute", "a"), "agudo": ("acute", "a"), "cronica": ("chronic", "a"), "cronico": ("chronic", "a"),
    "intensa": ("severe", "a"), "intenso": ("severe", "a"), "forte": ("severe", "a"), "ligeira": ("mild", "a"), "ligeiro": ("mild", "a"),
    "moderada": ("moderate", "a"), "moderado": ("moderate", "a"), "grave": ("severe", "a"), "persistente": ("persistent", "a"),
    "generalizada": ("generalized", "a"), "generalizado": ("generalized", "a"), "arterial": ("arterial", "a"),
    "pulmonar": ("pulmonary", "a"), "renal": ("renal", "a"), "cardiaca": ("cardiac", "a"), "cardiaco": ("cardiac", "a"),
    "hepatica": ("hepatic", "a"), "hepatico": ("hepatic", "a"), "urinaria": ("urinary", "a"), "urinario": ("urinary", "a"),
    "respiratoria": ("respiratory", "a"), "respiratorio": ("respiratory", "a"), "gastrica": ("gastric", "a"),
    "intestinal": ("intestinal", "a"), "cerebral": ("cerebral", "a"), "vascular": ("vascular", "a"), "sanguinea": ("blood", "a"),
    "sanguineas": ("blood", "a"), "sanguineo": ("blood", "a"), "laboratoriais": ("laboratory", "a"), "laboratorial": ("laboratory", "a"),
    "bacteriana": ("bacterial", "a"), "viral": ("viral", "a"), "alergica": ("allergic", "a"), "alergico": ("allergic", "a"),
    "esquerdo": ("left", "a"), "esquerda": ("left", "a"), "direito": ("right", "a"), "direita": ("right", "a"),
    "inferior": ("lower", "a"), "inferiores": ("lower", "a"), "superior": ("upper", "a"), "superiores": ("upper", "a"),
    "seca": ("dry", "a"), "alta": ("high", "a"), "alto": ("high", "a"), "baixa": ("low", "a"), "baixo": ("low", "a"),
    "medicamentosa": ("drug", "a"), "medicamentosas": ("drug", "a"), "diabetica": ("diabetic", "a"), "neurologico": ("neurological", "a"),
    "completo": ("complete", "a"), "completa": ("complete", "a"), "cardiopulmonar": ("cardiopulmonary", "a"),
    # body parts and others (no reordering)
    "peito": ("chest", "n"), "cabeca": ("head", "n"), "garganta": ("throat", "n"), "estomago": ("stomach", "n"),
    "joelho": ("knee", "n"), "ombro": ("shoulder", "n"), "pescoco": ("neck", "n"), "costas": ("back", "n"),
    "braco": ("arm", "n"), "perna": ("leg", "n"), "pernas": ("legs", "n"), "membros": ("limbs", "n"), "membro": ("limb", "n"),
    "coracao": ("heart", "n"), "pulmao": ("lung", "n"), "figado": ("liver", "n"), "rim": ("kidney", "n"), "rins": ("kidneys", "n"),
    "tipo": ("type", None), "de": ("of", None), "do": ("of the", None), "da": ("of the", None), "dos": ("of the", None),
    "das": ("of the", None), "no": ("in the", None), "na": ("in the", None), "com": ("with", None), "sem": ("without", None),
    "e": ("and", None), "ou": ("or", None), "por": ("due to", None), "em": ("in", None), "a": ("to", None), "ao": ("to the", None),
    # drugs written differently in English
    "amoxicilina": ("amoxicillin", None), "ampicilina": ("ampicillin", None), "penicilina": ("penicillin", None),
    "penicilinas": ("penicillins", None), "ibuprofeno": ("ibuprofen", None), "metformina": ("metformin", None),
    "insulina": ("insulin", None), "omeprazol": ("omeprazole", None), "pantoprazol": ("pantoprazole", None),
    "morfina": ("morphine", None), "furosemida": ("furosemide", None), "atorvastatina": ("atorvastatin", None),
    "sinvastatina": ("simvastatin", None), "varfarina": ("warfarin", None), "amlodipina": ("amlodipine", None),
    "digoxina": ("digoxin", None), "prednisolona": ("prednisolone", None), "prednisona": ("prednisone", None),
    "dexametasona": ("dexamethasone", None), "claritromicina": ("clarithromycin", None), "azitromicina": ("azithromycin", None),
    "ciprofloxacina": ("ciprofloxacin", None), "ceftriaxona": ("ceftriaxone", None), "gentamicina": ("gentamicin", None),
    "vancomicina": ("vancomycin", None), "metronidazol": ("metronidazole", None), "fluconazol": ("fluconazole", None),
    "aciclovir": ("acyclovir", None), "sertralina": ("sertraline", None), "fluoxetina": ("fluoxetine", None),
    "levotiroxina": ("levothyroxine", None), "heparina": ("heparin", None), "enoxaparina": ("enoxaparin", None),
    "cefalosporinas": ("cephalosporins", None), "cefalexina": ("cephalexin", None), "tetraciclina": ("tetracycline", None),
    "eritromicina": ("erythromycin", None), "sulfametoxazol": ("sulfamethoxazole", None), "carbamazepina": ("carbamazepine", None),
    "fenitoina": ("phenytoin", None), "diclofenac": ("diclofenac", None), "naproxeno": ("naproxen", None),
    "paracetamol": ("paracetamol", None), "tramadol": ("tramadol", None), "aspirina": ("aspirin", None),
    "bisoprolol": ("bisoprolol", None), "captopril": ("captopril", None), "losartan": ("losartan", None),
    "lorazepam": ("lorazepam", None), "diazepam": ("diazepam", None), "salbutamol": ("salbutamol", None),
    "contraste": ("contrast", None), "iodado": ("iodinated", None), "sulfamidas": ("sulfonamides", None),
    "mg": ("mg", None), "ml": ("ml", None),
}

_PUNCT = ".,;:()"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split()).strip(_PUNCT + " ")


def _glossary_translate(term: str):
    """Word-by-word with noun+adjective re-ordering. Returns (english, complete: bool)."""
    original_tokens = term.split()
    parts, complete = [], True
    for tok in original_tokens:
        core = tok.strip(_PUNCT)
        key = _norm(core)
        if not key:
            continue
        if key in WORDS:
            parts.append(list(WORDS[key]))
        elif re.fullmatch(r"[\d.,/%\-]+", key) or (core.isupper() and len(core) <= 6):
            parts.append([core, None])                      # numbers / acronyms stay as they are
        else:
            parts.append([core, None])
            complete = False                                # unknown word kept in Portuguese
    # re-order "noun adjective(+)" -> "adjective(+) noun"
    i, out = 0, []
    while i < len(parts):
        if parts[i][1] == "n":
            j = i + 1
            while j < len(parts) and parts[j][1] == "a":
                j += 1
            if j > i + 1:
                out.extend(p[0] for p in reversed(parts[i + 1:j]))
                out.append(parts[i][0])
                i = j
                continue
        out.append(parts[i][0])
        i += 1
    return " ".join(out), complete


def _argos(text: str):
    if os.environ.get("NUTRICHECK_USE_ARGOS") != "1":
        return None
    try:
        import argostranslate.translate as t
        out = (t.translate(text, "pt", "en") or "").strip()
        return out if out and out.casefold() != text.casefold() else None
    except Exception:
        return None


def translate_term(term: str):
    """Returns (english_text, complete)."""
    clean = " ".join(str(term).split())
    if not clean:
        return "", True
    key = _norm(clean)
    if key in PHRASES:
        return PHRASES[key], True
    neural = _argos(clean)
    if neural:
        return neural, True
    return _glossary_translate(clean)


def translate_entities(entities: dict):
    """
    entities: {"symptoms": [pt, ...], ...}
    Returns (english_entities, untranslated) where untranslated = [(key, original_term), ...]
    """
    out, untranslated = {}, []
    for key, values in entities.items():
        seen, items = set(), []
        for value in values:
            english, ok = translate_term(value)
            if not ok:
                untranslated.append((key, value))
            norm = english.casefold()
            if english and norm not in seen:
                seen.add(norm)
                items.append(english)
        out[key] = items
    return out, untranslated
