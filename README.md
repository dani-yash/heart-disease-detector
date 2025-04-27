# Heart Disease Detection

A comprehensive machine learning pipeline to predict the presence of heart disease using clinical features. This project covers data profiling, exploratory analysis, feature engineering, model training and evaluation, hyperparameter tuning, fairness assessment, and ensemble learning.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Highlights](#key-highlights)
3. [Dataset](#dataset)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage](#usage)
8. [Key Steps](#key-steps)
9. [Modeling & Evaluation](#modeling--evaluation)
10. [Fairness Analysis](#fairness-analysis)
11. [Ensemble Learning & ROC](#ensemble-learning--roc)

---

## Project Overview
This notebook-driven project implements a heart disease detector using classical ML algorithms and ensemble methods. It demonstrates a full workflow from data profiling through model stacking and ROC curve analysis.

## Key Highlights
- **Dataset**: Cleveland Heart Disease (1,190 samples, 12 features)
- **Profiling**: Automated exploratory reports with `ydata-profiling`
- **Baseline Models**: Logistic Regression, K-Nearest Neighbors
- **Safety**: Class-weighted logistic regression to reduce false negatives
- **Tuning**: 5-fold cross-validation and `GridSearchCV` for hyperparameters
- **Fairness**: Separate evaluation by gender groups
- **Ensemble**: StackingCVClassifier combining multiple base learners
- **Visualization**: Correlation heatmaps and ROC curve with AUC

## Dataset
The Cleveland dataset includes the following clinical features:

| Feature               | Description                         |
|-----------------------|-------------------------------------|
| age                   | Age in years                        |
| sex                   | (0 = female, 1 = male)              |
| chest pain type       | (1–4) categorical                   |
| resting blood pressure| Resting BP (mm Hg)                  |
| cholesterol           | Serum cholesterol (mg/dl)           |
| fasting blood sugar   | > 120 mg/dl (0 = false; 1 = true)   |
| resting ECG           | (0–2) categories                    |
| max heart rate        | Maximum heart rate achieved         |
| exercise angina       | (0 = no; 1 = yes)                   |
| oldpeak               | ST depression induced by exercise   |
| ST slope              | Peak exercise ST slope (1–3)        |
| target                | Heart disease (0 = no; 1 = yes)     |

## Prerequisites
- Python 3.8+ installed from [python.org](https://www.python.org)
- `pip` package manager
- (Optional) Virtual environment tool: `venv` or `conda`

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-detector.git
   cd heart-disease-detector
   ```
2. **Create and activate a virtual environment** (optional)
   ```bash
   python3 -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\\Scripts\\activate
   ```
3. **Install required packages**
   ```bash
   pip install \
     ydata-profiling \
     pandas numpy scikit-learn matplotlib seaborn pyspark mlxtend
   ```

## Project Structure
```text
heart-disease-detector/
├── data/                      # Raw dataset CSV
│   └── Heart_Disease_Dataset.csv
├── notebooks/                 # Jupyter notebooks
│   └── heart_disease_detector.ipynb
└── README.md                  # Project README
```

## Usage
1. **Ensure data is available**
   - Place `Heart_Disease_Dataset.csv` in the `data/` folder.
2. **Launch the notebook**
   ```bash
   jupyter notebook notebooks/heart_disease_detector.ipynb
   ```
3. **Execute all cells** to reproduce profiling, modeling, and evaluation.

## Key Steps
1. **Data Profiling** with `ydata-profiling`
2. **EDA**: histograms, count plots, correlation heatmap
3. **Preprocessing**: train/test split; `StandardScaler` on numeric features
4. **Report**: compute Accuracy, Precision, Recall, F1, FNR

## Modeling & Evaluation
- Train **Logistic Regression** and **KNN** classifiers
- **Safe LR** with `class_weight='balanced'`
- Generate confusion matrices and classification metrics
- **Cross-Validation** for performance stability

## Fairness Analysis
- Split test set by `sex` feature
- Evaluate performance separately for male vs. female
- Highlight False Negative Rate differences

## Ensemble Learning & ROC
- Build **StackingCVClassifier** with base learners: LR, KNN, SVM, Decision Tree, RF, NB
- Meta-classifier: Logistic Regression
- **3-fold CV** on base models
- Plot **ROC Curve** and compute AUC
```

