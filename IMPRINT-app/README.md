# 🏥 IMPRINT Risk Calculator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Validated-success)

## 📋 Overview
**IMPRINT (Imaging and Markers for Preoperative Risk Assessment in Uterine Tumors)** is a clinical decision support tool designed to assist clinicians in the preoperative discrimination between benign leiomyomas and uterine sarcomas (including STUMP).

This application implements the **IMPRINT Extended Model**, a multivariable logistic regression algorithm that integrates standardized ultrasound features with systemic inflammatory markers (SIMs) to refine risk stratification.

> **Key Feature:** Unlike traditional algorithms, IMPRINT uses continuous variables (standardized via Z-scores) rather than dichotomous cut-offs, preserving the biological continuum of risk.

---

## 🧠 Models Implemented

The application computes risks using three distinct algorithms simultaneously for comparison:

| Model | Type | Predictors | Best Use Case |
| :--- | :--- | :--- | :--- |
| **🚀 IMPRINT Extended** | **Combined** (US + Blood) | Menopause, Irreg. Margins, CS, Diameter, Cystic Areas, **PIV**, **NLR**, **MLR** | **Recommended.** Best calibration and discrimination (AUC 0.864). |
| **🔹 IMPRINT Core** | Ultrasound Only | Menopause, Irreg. Margins, CS, Diameter, Cystic Areas | Use when laboratory blood tests are unavailable. |
| **🌙 MYLUNAR** | External Benchmark | Age, Diameter >8cm, Irreg. Margins, CS=4, Shadows | Included for external validation and comparison purposes. |

---

## ⚙️ Methodology & Architecture

The model coefficients are **hard-coded** within `app.py`, derived from the final analysis of the training cohort (N=204). This ensures the app is standalone and does not rely on external dependency files.

### Data Standardization
To maintain consistency with the published manuscript, the app applies a rigorous preprocessing pipeline to raw inputs:

1.  **Log-Transformation:** Applied to skewed variables (Max Diameter, NLR, MLR, PIV) to normalize distribution.
2.  **Z-Score Standardization:** $z = \frac{x - \mu}{\sigma}$  
    *Inputs are standardized using the population parameters (Mean/SD) defined in the study's **Supplementary Table 1**.*

### Performance Metrics (Extended Model)
* **AUC:** 0.864 (95% CI: 0.813–0.908)
* **Brier Score:** 0.150 (High calibration accuracy)
* **Calibration:** Hosmer–Lemeshow p=0.872
* **AIC:** 202.5

---

## 📝 Input Requirements

To generate a prediction, the interface requires the following data points:

### 1. Clinical Data
* **Age** (Years)
* **Menopausal Status** (Pre/Post)

### 2. Ultrasound Features
* **Max Lesion Diameter** (mm, continuous)
* **Color Score** (Subjective semiquantitative score 1–4)
* **Irregular Margins** (Yes/No)
* **Cystic Areas** (Yes/No)
* **Acoustic Shadows** (Required only for MYLUNAR comparison)

### 3. Laboratory Data (Complete Blood Count)
*Enter absolute counts (e.g., 4500/µL or 4.5 x10³/µL):*
* **Neutrophils**
* **Lymphocytes**
* **Monocytes**
* **Platelets**

*The app automatically calculates derived markers (NLR, MLR, PIV).*

---

## 🚦 Risk Classification

Patients are stratified into three clinical management bands:

| Risk Band | Probability | Clinical Suggestion |
| :--- | :--- | :--- |
| **🟢 Low Risk** | **< 10%** | Consider conservative management or ultrasound follow-up. |
| **🟡 Intermediate**| **10 – 50%** | Consider second-level imaging (MRI) or referral to expert centers. |
| **🔴 High Risk** | **> 50%** | Planned oncologic surgery in a referral sarcoma center is advised. |

---

## 💻 Installation & Usage

This app is built with **Streamlit**. To run it locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/IMPRINT-app.git](https://github.com/your-username/IMPRINT-app.git)
    cd IMPRINT-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```

---

## ⚖️ Disclaimer

**For Research and Educational Purposes Only.**

This tool is a statistical model derived from retrospective data. It is **not** intended to replace clinical judgment, pathological diagnosis, or official guidelines. The authors assume no responsibility for medical decisions made based on this tool.

_Derived from the IMPRINT Study (Paratore et al., 2025)._
