
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
1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/heart-disease-detector.git
   cd heart-disease-detector
   ```

2. **Setup environment**
   - **Create virtual environment**
     ```bash
     python3 -m venv venv
     ```
   - **Activate environment**
     - macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
     - Windows:
       ```bash
       venv\Scripts\activate
       ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```text
heart-disease-detector/
├── data/                      # Raw CSV
├── notebooks/                 # Jupyter/Colab notebooks
│   └── heart_disease_detector.ipynb
├── requirements.txt           # Python libs
└── README.md                  # Project README
```

## Usage

1. **Load data in Colab**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp /content/drive/MyDrive/HW2/HW2_Data.zip /content/
   !unzip /content/HW2_Data.zip -d /content/
   ```
2. **Run the notebook**:
   ```bash
   jupyter notebook notebooks/heart_disease_detector.ipynb
   ```
3. **Execute all cells** sequentially to reproduce results.

## Key Steps

1. **Data Profiling** with `ydata-profiling`
2. **EDA**: histograms, count plots, correlation heatmap
3. **Preprocessing**: train/test split; `StandardScaler` on numeric
4. **Custom Report**: compute Accuracy, Precision, Recall, F1, FNR

## Modeling & Evaluation

- Train **Logistic Regression** and **KNN** classifiers
- **Safe LR** with `class_weight='balanced'`
- Compute confusion matrices
- **Cross-Validation** for performance stability

## Fairness Analysis

- Split test set by `sex` feature
- Evaluate metrics separately for male vs. female
- Discuss False Negative Rate differences

## Ensemble Learning & ROC

- Build **StackingCVClassifier** with multiple base learners
- Meta-classifier: Logistic Regression
- 3-fold cross-validation
- **ROC Curve** plotting and AUC calculation
