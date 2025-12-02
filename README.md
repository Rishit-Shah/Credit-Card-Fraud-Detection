# Credit Card Fraud Detection Model — EECS 4412

This repository contains the full implementation, analysis, and results of our **Credit Card Fraud Detection** project completed for **EECS 4412 – Data Mining**.  
Our objective was to build an **end-to-end machine learning pipeline** for fraud detection while handling extreme class imbalance and implementing one model **from scratch** using NumPy.

---

## Project Overview

Credit card fraud detection is a highly challenging problem due to:

- **Severe class imbalance** (<0.2% fraud)
- **Non-linear feature relationships**
- **Anonymized PCA-transformed features (V1–V28)**
- **High financial cost of false negatives and false positives**

This project includes:

- Extensive **Phase 1 Exploratory Data Analysis (EDA)**
- Full **preprocessing pipeline** (scaling, duplicate removal, imputation)
- A complete **scratch Logistic Regression implementation**
- Additional models:
  - Class-weighted Scikit-learn Logistic Regression
  - Random Forest Classifier
- Detailed evaluation using **precision, recall, F1-score**, and accuracy
- Final report + video presentation

---

## Phase 1: Exploratory Data Analysis (EDA)

Key insights:

### Class Imbalance  
Fraud represents only **0.17%** of all transactions → accuracy alone is misleading.

### Amount Distribution  
- Right-skewed  
- Most purchases are small  

### Time Feature  
- Displays periodic patterns  
- Potential daily cycles

### PCA Features  
- Strong correlation blocks  
- t-SNE reveals non-linear clusters of fraud

EDA was performed using:  
`matplotlib`, `seaborn`, and `pandas`.

---

## Preprocessing Pipeline

Our data preprocessing includes:

- **Duplicate removal**  
  → Over 1,000 duplicates removed to avoid leakage
- **Mean imputation** for missing values
- **Feature scaling**  
  - `Time` → MinMax scaling  
  - `Amount` → Standardization  
- **Stratified 80/20 split**

The preprocessing ensures stable gradient descent and fair model evaluation.

---

## Models Implemented

### 1. Logistic Regression (Scratch Implementation)
Implemented fully using **NumPy**, including:

- Sigmoid activation  
- Binary cross-entropy loss  
- Gradient descent updates  
- Weight & bias initialization  
- Loss history tracking  

This meets the course requirement for a model implemented *without* ML libraries.

---

### 2. Scikit-learn Logistic Regression
- Includes `class_weight='balanced'`
- Much higher recall, very low precision
- Useful for detecting all possible fraud cases

---

### 3. Random Forest Classifier
- Non-linear ensemble method  
- Best overall model  
- Captures interactions between PCA features

---

## Results Summary

| Model | Precision (Fraud) | Recall (Fraud) | F1 Score | Accuracy |
|------|-------------------|----------------|----------|----------|
| **Scratch Logistic Regression** | 0.789 | 0.316 | 0.451 | 99.87% |
| **Sklearn Logistic Regression** | 0.06 | 0.87 | 0.11 | 98% |
| **Random Forest Classifier** | **0.99** | **0.69** | **0.81** | **100%** |

### Best Model: **Random Forest**
Reason:
- Handles severe imbalance  
- Captures non-linearities  
- Best trade-off between precision & recall  

---

## How to Run the Project

### Option 1 — Run in Jupyter Notebook (Recommended)
Open: creditcardfraud.ipynb

Run all cells top-to-bottom.

---

### Option 2 — Manual Steps

1. **Install dependencies**

pip install numpy pandas scikit-learn matplotlib seaborn

2. **Open or run notebook**


jupyter notebook Credit_Card_Fraud.ipynb


3. **Run each block:**
- Load dataset  
- Perform EDA  
- Preprocess  
- Train scratch LR  
- Train sklearn LR / Random Forest  
- Evaluate results  

---

## System Design (Summary)

### Components:
- Data loading module  
- Preprocessing section  
- Scratch logistic regression class  
- Model training & evaluation  
- Visualization tools  

### Data Structures:
- Pandas DataFrame for raw data  
- NumPy arrays for model input  
- Dictionaries for metrics  

### Control Flow:
1. Load  
2. Clean  
3. Scale  
4. Split  
5. Train  
6. Evaluate  
7. Visualize  
8. Report  

---

## Limitations

- No SMOTE or oversampling  
- Limited hyperparameter tuning  
- Scratch LR lacks regularization  
- Not optimized for streaming data  

---

## Future Work

- SMOTE / ADASYN oversampling  
- Cost-sensitive learning  
- XGBoost, LightGBM  
- Autoencoders for anomaly detection  
- Real-time fraud detection pipeline  

---

## Team

**Rishit Shah**  
- EDA visualizations, scaling, duplicate removal  

**Sagar Saha**  
- Dataset description, preprocessing support, training sklearn models  

**Pratham Patel**  
- Phase 1 challenges, scratch logistic regression assisted, report writing  

**Miguel De La Cruz**  
- Dataset insights, scratch model implementation  

---

## License
This project is for academic purposes under York University EECS 4412.  
Reuse requires instructor permission.

---

## Acknowledgements

- Kaggle Credit Card Fraud Detection Dataset  
- Scikit-learn  
- NumPy  
- York University  
- OpenAI ChatGPT (referenced as an AI-assisted tool in project report)




