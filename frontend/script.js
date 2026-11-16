/**
 * AI Healthcare Assistance System - Client Application
 * Handles symptom catalog loading, multi-select tag interaction,
 * form validation, API communication, and result rendering.
 */

// Application State
const state = {
  symptomCatalog: [],
  categories: {},
  selectedSymptoms: new Set(),
  activeCategory: 'all',
  searchQuery: '',
};

// Preset Scenarios for Rapid Demonstrations & Interview Testing
const PRESETS = {
  cold: {
    age: 28,
    gender: 'Male',
    location: 'New York',
    symptoms: ['runny_nose', 'sneezing', 'sore_throat', 'cough', 'nasal_congestion']
  },
  asthma: {
    age: 24,
    gender: 'Female',
    location: 'Chicago',
    symptoms: ['shortness_of_breath', 'wheezing', 'chest_tightness', 'cough']
  },
  migraine: {
    age: 32,
    gender: 'Female',
    location: 'Austin',
    symptoms: ['throbbing_headache', 'nausea', 'sensitivity_to_light', 'dizziness']
  },
  gerd: {
    age: 45,
    gender: 'Male',
    location: 'San Francisco',
    symptoms: ['heartburn', 'chest_pain', 'burning_stomach_pain', 'nausea']
  }
};

// DOM Elements
const dom = {
  systemStatusBadge: document.getElementById('systemStatusBadge'),
  form: document.getElementById('predictionForm'),
  patientAge: document.getElementById('patientAge'),
  patientLocation: document.getElementById('patientLocation'),
  ageError: document.getElementById('ageError'),
  locationError: document.getElementById('locationError'),
  symptomsError: document.getElementById('symptomsError'),
  selectedSymptomsTray: document.getElementById('selectedSymptomsTray'),
  selectedCounter: document.getElementById('selectedCounter'),
  symptomSearchInput: document.getElementById('symptomSearchInput'),
  clearSearchBtn: document.getElementById('clearSearchBtn'),
  categoryPillsContainer: document.getElementById('categoryPillsContainer'),
  symptomsBank: document.getElementById('symptomsBank'),
  predictSubmitBtn: document.getElementById('predictSubmitBtn'),
  resetFormBtn: document.getElementById('resetFormBtn'),
  emptyStateCard: document.getElementById('emptyStateCard'),
  loadingCard: document.getElementById('loadingCard'),
  errorCard: document.getElementById('errorCard'),
  errorMessage: document.getElementById('errorMessage'),
  dismissErrorBtn: document.getElementById('dismissErrorBtn'),
  resultsContainer: document.getElementById('resultsContainer'),
  predictedDiseaseName: document.getElementById('predictedDiseaseName'),
  confidenceValue: document.getElementById('confidenceValue'),
  confidenceBarFill: document.getElementById('confidenceBarFill'),
  confidencePercentageText: document.getElementById('confidencePercentageText'),
  locationBadge: document.getElementById('locationBadge'),
  differentialsToggle: document.getElementById('differentialsToggle'),
  differentialsList: document.getElementById('differentialsList'),
  featureBarsContainer: document.getElementById('featureBarsContainer'),
  specialistTitle: document.getElementById('specialistTitle'),
  specialistScope: document.getElementById('specialistScope'),
  specialistUrgency: document.getElementById('specialistUrgency'),
  specialistAction: document.getElementById('specialistAction'),
  doctorsSummaryText: document.getElementById('doctorsSummaryText'),
  doctorsGrid: document.getElementById('doctorsGrid')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  checkSystemHealth();
  fetchSymptomCatalog();
  setupEventListeners();
});

// 1. Check System Health & Model Status
async function checkSystemHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.model_loaded) {
      dom.systemStatusBadge.innerHTML = `
        <span class="status-dot"></span>
        <span class="status-text">Model Active (${data.total_diseases} Conditions)</span>
      `;
    } else {
      dom.systemStatusBadge.innerHTML = `
        <span class="status-dot" style="background:#f59e0b;"></span>
        <span class="status-text">Model Degraded</span>
      `;
    }
  } catch (err) {
    dom.systemStatusBadge.innerHTML = `
      <span class="status-dot" style="background:#ef4444;"></span>
      <span class="status-text">Backend Offline</span>
    `;
  }
}

// 2. Fetch Available Symptom Catalog
async function fetchSymptomCatalog() {
  try {
    const res = await fetch('/symptoms');
    if (!res.ok) throw new Error('Could not load symptom catalog');
    const data = await res.json();
    
    state.symptomCatalog = data.all_symptoms || [];
    state.categories = data.categories || {};
    
    renderCategoryPills();
    renderSymptomBank();
  } catch (err) {
    dom.symptomsBank.innerHTML = `
      <div class="field-error" style="padding:1rem; text-align:center;">
        Failed to load symptoms: ${err.message}. Please verify the backend is running.
      </div>
    `;
  }
}

// 3. Render Category Tabs
function renderCategoryPills() {
  const categories = Object.keys(state.categories);
  let html = `<button type="button" class="cat-pill active" data-category="all">All (${state.symptomCatalog.length})</button>`;
  
  categories.forEach(cat => {
    const count = state.categories[cat].length;
    html += `<button type="button" class="cat-pill" data-category="${cat}">${cat} (${count})</button>`;
  });
  
  dom.categoryPillsContainer.innerHTML = html;
  
  // Attach category click listeners
  dom.categoryPillsContainer.querySelectorAll('.cat-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      dom.categoryPillsContainer.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeCategory = btn.dataset.category;
      renderSymptomBank();
    });
  });
}

// 4. Render Symptom Chips in the Selection Bank
function renderSymptomBank() {
  let list = [];
  
  if (state.activeCategory === 'all') {
    list = state.symptomCatalog;
  } else if (state.categories[state.activeCategory]) {
    list = state.categories[state.activeCategory];
  }
  
  // Filter by search query if present
  if (state.searchQuery) {
    const q = state.searchQuery.toLowerCase();
    list = list.filter(item => 
      item.label.toLowerCase().includes(q) || item.id.toLowerCase().includes(q)
    );
  }
  
  if (list.length === 0) {
    dom.symptomsBank.innerHTML = `<div class="loading-symptoms">No symptoms match "${state.searchQuery}".</div>`;
    return;
  }
  
  dom.symptomsBank.innerHTML = list.map(item => {
    const isSelected = state.selectedSymptoms.has(item.id);
    return `
      <button type="button" 
              class="symptom-chip ${isSelected ? 'selected' : ''}" 
              data-id="${item.id}"
              aria-pressed="${isSelected}">
        ${isSelected ? '&#10003; ' : '+ '}${item.label}
      </button>
    `;
  }).join('');
  
  // Attach click listeners to chips
  dom.symptomsBank.querySelectorAll('.symptom-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      toggleSymptom(chip.dataset.id);
    });
  });
}

// 5. Toggle Symptom Selection
function toggleSymptom(symptomId) {
  if (state.selectedSymptoms.has(symptomId)) {
    state.selectedSymptoms.delete(symptomId);
  } else {
    state.selectedSymptoms.add(symptomId);
  }
  
  dom.symptomsError.textContent = '';
  updateSelectedTray();
  renderSymptomBank();
}

// 6. Update Active Selected Symptoms Tray
function updateSelectedTray() {
  const count = state.selectedSymptoms.size;
  dom.selectedCounter.textContent = `${count} selected`;
  
  if (count === 0) {
    dom.selectedSymptomsTray.innerHTML = `
      <span class="placeholder-tag">No symptoms selected yet. Pick from the categories below.</span>
    `;
    return;
  }
  
  const tagsHtml = Array.from(state.selectedSymptoms).map(id => {
    const found = state.symptomCatalog.find(s => s.id === id);
    const label = found ? found.label : id.replace(/_/g, ' ');
    return `
      <span class="active-symptom-tag">
        ${label}
        <button type="button" class="tag-remove-btn" data-id="${id}" title="Remove symptom" aria-label="Remove ${label}">&times;</button>
      </span>
    `;
  }).join('');
  
  dom.selectedSymptomsTray.innerHTML = tagsHtml;
  
  // Attach remove handlers
  dom.selectedSymptomsTray.querySelectorAll('.tag-remove-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSymptom(btn.dataset.id);
    });
  });
}

// 7. Setup Event Listeners
function setupEventListeners() {
  // Search input filtering
  dom.symptomSearchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value.trim();
    dom.clearSearchBtn.style.display = state.searchQuery ? 'block' : 'none';
    renderSymptomBank();
  });
  
  dom.clearSearchBtn.addEventListener('click', () => {
    dom.symptomSearchInput.value = '';
    state.searchQuery = '';
    dom.clearSearchBtn.style.display = 'none';
    renderSymptomBank();
    dom.symptomSearchInput.focus();
  });
  
  // Quick location chips
  document.querySelectorAll('.chip-loc').forEach(btn => {
    btn.addEventListener('click', () => {
      dom.patientLocation.value = btn.dataset.city;
      dom.locationError.textContent = '';
    });
  });
  
  // Presets
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      applyPreset(presetKey);
    });
  });
  
  // Reset Button
  dom.resetFormBtn.addEventListener('click', () => {
    resetForm();
  });
  
  // Differentials Accordion Toggle
  dom.differentialsToggle.addEventListener('click', () => {
    const isVisible = dom.differentialsList.style.display === 'flex';
    dom.differentialsList.style.display = isVisible ? 'none' : 'flex';
    dom.differentialsToggle.classList.toggle('active', !isVisible);
  });
  
  // Dismiss error
  dom.dismissErrorBtn.addEventListener('click', () => {
    dom.errorCard.style.display = 'none';
    dom.emptyStateCard.style.display = 'block';
  });
  
  // Form Submit
  dom.form.addEventListener('submit', handlePredictionSubmit);
}

// 8. Apply Preset Scenario
function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  
  dom.patientAge.value = p.age;
  dom.patientLocation.value = p.location;
  
  const genderRadio = dom.form.querySelector(`input[name="gender"][value="${p.gender}"]`);
  if (genderRadio) genderRadio.checked = true;
  
  state.selectedSymptoms = new Set(p.symptoms);
  updateSelectedTray();
  renderSymptomBank();
  
  // Clear errors
  dom.ageError.textContent = '';
  dom.locationError.textContent = '';
  dom.symptomsError.textContent = '';
}

// 9. Reset Form
function resetForm() {
  dom.patientAge.value = 28;
  dom.patientLocation.value = 'New York';
  const defaultGender = dom.form.querySelector('input[name="gender"][value="Male"]');
  if (defaultGender) defaultGender.checked = true;
  
  state.selectedSymptoms.clear();
  state.searchQuery = '';
  dom.symptomSearchInput.value = '';
  dom.clearSearchBtn.style.display = 'none';
  
  updateSelectedTray();
  renderSymptomBank();
  
  dom.ageError.textContent = '';
  dom.locationError.textContent = '';
  dom.symptomsError.textContent = '';
  
  dom.resultsContainer.style.display = 'none';
  dom.loadingCard.style.display = 'none';
  dom.errorCard.style.display = 'none';
  dom.emptyStateCard.style.display = 'block';
}

// 10. Form Validation & Submission
async function handlePredictionSubmit(e) {
  e.preventDefault();
  
  // Reset field errors
  dom.ageError.textContent = '';
  dom.locationError.textContent = '';
  dom.symptomsError.textContent = '';
  
  const age = parseInt(dom.patientAge.value, 10);
  const location = dom.patientLocation.value.trim();
  const genderRadio = dom.form.querySelector('input[name="gender"]:checked');
  const gender = genderRadio ? genderRadio.value : 'Male';
  const symptoms = Array.from(state.selectedSymptoms);
  
  let hasError = false;
  
  if (isNaN(age) || age < 1 || age > 120) {
    dom.ageError.textContent = 'Please enter a valid age between 1 and 120.';
    hasError = true;
  }
  
  if (!location || location.length < 2) {
    dom.locationError.textContent = 'Please enter a city or metropolitan location.';
    hasError = true;
  }
  
  if (symptoms.length === 0) {
    dom.symptomsError.textContent = 'Please select at least one symptom to evaluate.';
    hasError = true;
  }
  
  if (hasError) return;
  
  // Show Loading State
  setLoadingState(true);
  
  try {
    const payload = { age, gender, symptoms, location };
    
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Inference request failed');
    }
    
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoadingState(false);
  }
}

// 11. Loading UI Transition
function setLoadingState(isLoading) {
  const btnText = dom.predictSubmitBtn.querySelector('.btn-text');
  const btnSpinner = dom.predictSubmitBtn.querySelector('.btn-spinner');
  
  if (isLoading) {
    dom.emptyStateCard.style.display = 'none';
    dom.resultsContainer.style.display = 'none';
    dom.errorCard.style.display = 'none';
    dom.loadingCard.style.display = 'block';
    
    dom.predictSubmitBtn.disabled = true;
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline-flex';
  } else {
    dom.loadingCard.style.display = 'none';
    dom.predictSubmitBtn.disabled = false;
    btnText.style.display = 'inline-flex';
    btnSpinner.style.display = 'none';
  }
}

// 12. Display Error Alert
function showError(message) {
  dom.emptyStateCard.style.display = 'none';
  dom.resultsContainer.style.display = 'none';
  dom.errorCard.style.display = 'block';
  dom.errorMessage.textContent = message;
}

// 13. Render Structured Prediction Results
function renderResults(data) {
  // 1. Probable Condition & Statistical Confidence
  dom.predictedDiseaseName.textContent = data.predicted_disease;
  dom.confidenceValue.textContent = data.confidence_percentage;
  dom.confidenceBarFill.style.width = data.confidence_percentage;
  dom.confidencePercentageText.textContent = `${data.confidence_percentage} statistical confidence`;
  dom.locationBadge.textContent = `Location: ${data.location_queried}`;
  
  // 2. Differential Diagnoses
  if (data.differential_diagnoses && data.differential_diagnoses.length > 0) {
    dom.differentialsSection.style.display = 'block';
    dom.differentialsList.innerHTML = data.differential_diagnoses.map(diff => `
      <div class="differential-item">
        <span class="diff-disease">${diff.disease}</span>
        <span class="diff-prob">${diff.confidence_percentage} probability</span>
      </div>
    `).join('');
  } else {
    dom.differentialsSection.style.display = 'none';
  }
  
  // 3. Explainability Feature Importance
  if (data.key_contributing_features && data.key_contributing_features.length > 0) {
    dom.featureBarsContainer.innerHTML = data.key_contributing_features.map(f => `
      <div class="feature-bar-row">
        <span class="feature-name" title="${f.symptom_display}">${f.symptom_display}</span>
        <div class="feature-track">
          <div class="feature-fill" style="width: ${f.importance_percentage}"></div>
        </div>
        <span class="feature-percent">${f.importance_percentage}</span>
      </div>
    `).join('');
  } else {
    dom.featureBarsContainer.innerHTML = `<div class="placeholder-tag">No feature importance breakdown available.</div>`;
  }
  
  // 4. Specialist Recommendation
  const spec = data.specialist_recommendation;
  dom.specialistTitle.textContent = spec.specialist_type;
  dom.specialistScope.textContent = spec.clinical_focus;
  dom.specialistUrgency.textContent = spec.urgency_guidance;
  dom.specialistAction.textContent = spec.recommended_action;
  
  // 5. Doctor Directory Filtered Results
  const isExact = data.is_location_exact_match;
  if (isExact) {
    dom.doctorsSummaryText.innerHTML = `Showing verified specialists located in <strong>${data.location_queried}</strong>.`;
  } else {
    dom.doctorsSummaryText.innerHTML = `No local doctors found directly in "${data.location_queried}". Showing <strong>telehealth and regional specialists</strong> available for virtual consultation:`;
  }
  
  if (data.recommended_doctors && data.recommended_doctors.length > 0) {
    dom.doctorsGrid.innerHTML = data.recommended_doctors.map(doc => `
      <div class="doctor-item-card">
        <div class="doc-top-row">
          <h5 class="doc-name">${doc.name}</h5>
          ${doc.telehealth_available ? '<span class="doc-badge-pill">Telehealth Available</span>' : ''}
        </div>
        <div class="doc-spec">${doc.specialization}</div>
        <div class="doc-clinic">&#127973; ${doc.hospital_clinic} &bull; ${doc.city}, ${doc.state_country}</div>
        <div class="doc-info-chips">
          <span class="doc-chip">&#9733; ${doc.rating} / 5.0</span>
          <span class="doc-chip">&#128197; ${doc.experience_years} yrs exp</span>
          <span class="doc-chip">&#9742; ${doc.contact}</span>
          <span class="doc-chip">&#9200; ${doc.consultation_hours}</span>
        </div>
      </div>
    `).join('');
  } else {
    dom.doctorsGrid.innerHTML = `<div class="placeholder-tag">No sample doctor records available for this specialty.</div>`;
  }
  
  // Show Results Container
  dom.emptyStateCard.style.display = 'none';
  dom.loadingCard.style.display = 'none';
  dom.errorCard.style.display = 'none';
  dom.resultsContainer.style.display = 'flex';
  
  // Smooth scroll into view on smaller displays
  if (window.innerWidth < 960) {
    dom.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}
