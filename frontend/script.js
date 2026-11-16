const API_URL = "https://nutricheck-ehca.onrender.com/predict";

const textarea = document.getElementById("medicalText");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const errorBox = document.getElementById("errorBox");
const resultStatus = document.getElementById("resultStatus");

const fields = {
  symptoms: document.getElementById("symptoms"),
  medications: document.getElementById("medications"),
  treatments: document.getElementById("treatments"),
  conditions: document.getElementById("conditions"),
  diagnosis: document.getElementById("diagnosis"),
  exams: document.getElementById("exams")
};

const allergyResult = document.getElementById("allergyResult");
const allergyConfidence = document.getElementById("allergyConfidence");

textarea.addEventListener("input", () => {
  charCount.textContent = `${textarea.value.length} characters`;
});

sampleBtn.addEventListener("click", () => {
  textarea.value =
    "O paciente apresenta dor abdominal e febre. Foi diagnosticado com gastrite. " +
    "Recebeu tratamento com antibiótico e iniciou amoxicilina. Foram realizados exames laboratoriais.";
  textarea.dispatchEvent(new Event("input"));
});

function displayValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None detected";
  if (value === null || value === undefined || value === "") return "None detected";
  return String(value);
}

function setResults(data) {
  const extracted = data.entities || data.extracted_information || data;

  fields.symptoms.textContent =
    displayValue(extracted.symptoms ?? extracted.symptom);
  fields.medications.textContent =
    displayValue(extracted.medications ?? extracted.medication);
  fields.treatments.textContent =
    displayValue(extracted.treatments ?? extracted.treatment);
  fields.conditions.textContent =
    displayValue(extracted.conditions ?? extracted.condition);
  fields.diagnosis.textContent =
    displayValue(extracted.diagnosis ?? extracted.diagnoses);
  fields.exams.textContent =
    displayValue(extracted.exams ?? extracted.examinations ?? extracted.exam);

  const allergy =
    data.medication_allergy ??
    data.allergy_prediction ??
    data.prediction ??
    data.medicationAllergy;

  if (allergy !== undefined && allergy !== null) {
    const positive =
      allergy === 1 || allergy === "1" ||
      String(allergy).toLowerCase().includes("positive") ||
      String(allergy).toLowerCase().includes("allergy");

    allergyResult.textContent = positive ? "Allergy detected" : "No allergy detected";
    allergyResult.className = `risk-pill ${positive ? "positive" : "negative"}`;

    const probability = data.medication_allergy_probability;
    if (typeof probability === "number") {
      allergyConfidence.textContent = `Model probability: ${(probability * 100).toFixed(1)}%`;
    } else {
      allergyConfidence.textContent = "";
    }
  } else {
    allergyResult.textContent = "Not returned by API";
    allergyResult.className = "risk-pill";
    allergyConfidence.textContent = "";
  }

  resultStatus.textContent = "Analysis complete";
  resultStatus.classList.add("ready");
}

function clearResults() {
  Object.values(fields).forEach((el) => { el.textContent = "—"; });
  allergyResult.textContent = "Not analyzed";
  allergyResult.className = "risk-pill";
  allergyConfidence.textContent = "";
  resultStatus.classList.remove("ready");
}

async function analyzeRecord() {
  const text = textarea.value.trim();

  errorBox.style.display = "none";
  errorBox.textContent = "";

  if (!text) {
    errorBox.textContent = "Please enter a medical record before analyzing.";
    errorBox.style.display = "block";
    textarea.focus();
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="btn-icon">◌</span> Processing...';

  resultStatus.textContent = "Processing";
  resultStatus.classList.remove("ready");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      // The backend rejected the input (not Portuguese / not a medical record / too short...)
      let message = `The server returned an error (${response.status}).`;
      try {
        const err = await response.json();
        if (err.detail && typeof err.detail === "object" && err.detail.message) {
          message = err.detail.message;
        }
      } catch (_) { /* response was not JSON */ }

      clearResults();
      errorBox.textContent = message;   // textContent: never render server text as HTML
      errorBox.style.display = "block";
      resultStatus.textContent = "Input rejected";
      return;
    }

    const data = await response.json();
    setResults(data);
  } catch (error) {
    console.error(error);
    clearResults();
    errorBox.innerHTML =
      `Could not connect to the FastAPI backend.<br>` +
      `Make sure the server is running at <b>${API_URL}</b> and that the endpoint accepts JSON with a <b>text</b> field.`;
    errorBox.style.display = "block";
    resultStatus.textContent = "Connection error";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML =
      '<span class="btn-icon">✦</span> Analyze Medical Record <span class="arrow">→</span>';
  }
}

analyzeBtn.addEventListener("click", analyzeRecord);
