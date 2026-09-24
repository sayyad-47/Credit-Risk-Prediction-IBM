# 🏦 Credit Risk & Loan Default Prediction

> **AICTE / IBM SkillsBuild Academic Internship — Capstone Project**  
> Author: **Sayyad Malik**  
> Technology Stack: Python · Scikit-Learn · Pandas · Streamlit · Plotly

---

## 📌 Project Overview

This project builds an end-to-end **Credit Risk & Loan Default Prediction** system that ingests the German Credit Dataset, trains a Gradient Boosting classifier, and serves interactive visualisations and real-time predictions through a professional Streamlit web application.

The system enables financial institutions to:
- Monitor portfolio-level risk KPIs at a glance
- Drill into default patterns across customer segments
- Score individual loan applicants in real time with model-driven underwriting recommendations

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **Name** | German Credit Dataset (Cleaned) |
| **File** | `german_credit_dataset_cleaned.csv` |
| **Source** | [UCI ML Repository – Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| **Records** | 1,000 customers |
| **Features** | 21 attributes (demographic, financial, loan-related) |
| **Target** | `Risk` — `No Risk` (good credit) / `Risk` (default) |

### Key Feature Descriptions

| Column | Description |
|---|---|
| `CheckingStatus` | Balance in the checking account (`less_0`, `0_to_200`, `greater_200`, `no_checking`) |
| `LoanDuration` | Duration of the loan in months |
| `CreditHistory` | Past credit behaviour |
| `LoanPurpose` | Purpose for which the loan is taken |
| `LoanAmount_INR` | Loan amount in Indian Rupees |
| `ExistingSavings` | Level of savings / bonds held |
| `EmploymentDuration` | Years of current employment |
| `Age` | Applicant's age in years |
| `Housing` | Housing situation (`own`, `free`, `rent`) |
| `Job` | Job classification (`skilled`, `unskilled`, `management_self-employed`) |
| `Risk` | **Target variable** — `No Risk` or `Risk` |

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML / Data | Scikit-Learn, Pandas, NumPy |
| Frontend | Streamlit 1.35+ |
| Charts | Plotly Express (interactive, responsive) |
| Report Gen | python-docx |
| Encoding | LabelEncoder (Scikit-Learn) |
| Model | Gradient Boosting Classifier |

---

## 🗂️ Project Structure

```
.
├── sayyadmalik_CreditRiskPrediction.py   # Main Streamlit application
├── generate_report.py                    # Auto-generates project DOCX report
├── german_credit_dataset_cleaned.csv     # Source dataset
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── sayyadmalik_ProjectReport.docx        # Generated report (run generate_report.py)
```

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites
Ensure you have **Python 3.11+** installed.

### 2. Clone / Download the project
```bash
# If hosted on GitHub
git clone https://github.com/<your-username>/credit-risk-prediction.git
cd credit-risk-prediction
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit application
```bash
streamlit run sayyadmalik_CreditRiskPrediction.py
```
The app will open automatically at **http://localhost:8501**.

### 6. Generate the Project Report (DOCX)
```bash
python generate_report.py
```
This creates `sayyadmalik_ProjectReport.docx` in the same directory.

---

## 📊 Application Pages

### Page 1 — Executive Overview
- Portfolio KPIs: Total Customers, Default Rate, Avg Loan Duration, Avg Amount, Avg Age
- Donut chart: Overall Risk Distribution
- Bar chart: Default Rate by Loan Duration Band
- Histogram: Loan Amount Distribution by Risk
- Box plot: Age Distribution across Risk Classes

### Page 2 — Portfolio Analysis
- Bar charts: Default Rate by Housing, Loan Purpose, Checking Status, Savings Level
- Grouped bar: Loan Count by Employment Duration & Risk
- Bubble scatter: Loan Amount vs Duration (sized by Age)

### Page 3 — Customer Risk Analysis
- Input form with 20 borrower attributes (sliders, dropdowns, number inputs)
- Gauge chart displaying real-time default probability
- Dynamic callout cards:
  - 🚨 `st.error` — High Risk (probability > 35%)
  - ✅ `st.success` — Low Risk (probability ≤ 35%)
  - 📋 `st.info` — Tailored underwriter action instructions
- Top-12 Feature Importance bar chart

### Page 4 — Model Performance
- Accuracy, AUC-ROC, and F1-Score scorecards
- Confusion Matrix heatmap
- Precision / Recall / F1 grouped bar chart
- Full feature importance ranking

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| n_estimators | 200 |
| learning_rate | 0.08 |
| max_depth | 4 |
| subsample | 0.85 |
| Train/Test Split | 80% / 20% (stratified) |
| Encoding | LabelEncoder per categorical column |
| Decision Threshold | 35% default probability |

---

## 🎨 Design Principles

- **Light theme only** — white background, `#f4f4f4` surfaces, `#0f62fe` IBM Blue accents
- **No dark mode** — professional, presentation-ready UI
- **Wide layout** (`st.set_page_config(layout="wide")`)
- **Responsive Plotly charts** — all labels rotated, margins tuned to prevent overlapping
- **IBM Design Language** — typography, colour palette, spacing

---

## 📜 License

This project is submitted as part of the **AICTE / IBM SkillsBuild Academic Internship** programme. All rights reserved by the author.

---

*Built with ❤️ using Python, Streamlit, and IBM SkillsBuild resources.*
