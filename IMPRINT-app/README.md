```markdown
# IMPRINT Clinical Decision Support 🩺

## Overview
IMPRINT is a clinical decision support tool designed to estimate the risk of malignancy in uterine masses.  
The **Extended model** has been updated with the final coefficients derived from logistic regression (statsmodels), ensuring calibrated outputs and performance metrics consistent with the published manuscript (AUC 0.865, Brier score 0.148, Hosmer–Lemeshow p=0.889, AIC 201.1).

## Model
- **Current version:** Extended (updated with final coefficients)
- **Model file:** `models/imprint_extended.pkl`
- **Predictors included:**
  - **Ultrasound**
    - Menopause (binary)
    - Irregular margins (binary)
    - Color score (ordinal, z-score)
    - Maximum diameter (mm, z-score)
    - Cystic areas (binary)
  - **Systemic Inflammatory Markers (SIMs)**
    - NLR (log + z-score)
    - SII (log + z-score)
    - PIV (log + z-score)

## Input
The interface requires the following inputs:

- **Clinical**
  - Age, AUB, Pain, Abdominal distress (captured but not included in final predictors)
- **Ultrasound**
  - Maximum diameter (mm)
  - Irregular margins (checkbox)
  - Cystic areas (checkbox)
  - Color score (slider 1–4)
  - Cooked aspect, Shadows absence (captured but not included in final predictors)
- **Laboratory (absolute counts, x10^9/L)**
  - Platelets
  - Neutrophils
  - Lymphocytes
  - Monocytes
  - Eosinophils

From these values, the app automatically derives systemic inflammatory markers (NLR, PLR, MLR, ELR, SII, SIRI, LMR, PIV) and applies manuscript‑consistent transformations (conditional log‑transform + z‑score standardization).

## Output
The Extended model provides:
- **Calibrated probability of malignancy** (continuous value between 0 and 1)
- **Risk class (band)** based on decision thresholds:
  - Low risk (<10%)
  - Intermediate risk (10–30%)
  - High risk (≥30%)
- **Clinical guidance** linked to each risk band:
  - Low: consider conservative management
  - Intermediate: consider MRI or referral
  - High: refer to sarcoma center

## Developer Mode
When developer mode is enabled, the app displays:
- Predictors actually used by the model
- The transformed input row (model-ready features)
- Derived SIMs
- Variables captured but not included in predictors
- Raw probability and classification outputs

## Performance
- Apparent AUC: 0.865  
- Bootstrap AUC (95% CI): 0.816–0.910  
- Brier score: 0.148  
- Hosmer–Lemeshow p-value: 0.889  
- AIC: 201.1  

---

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Enter clinical, ultrasound, and lab data in the interface.
4. Click **Estimate risk** to obtain the calibrated probability and risk class.
```