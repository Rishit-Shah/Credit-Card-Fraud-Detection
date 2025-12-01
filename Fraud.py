# Rishit
# EDA (Summary statistics)
import pandas as pd

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display dataset info
print("\nDataset Info:")
print(df.info())

# Check shape
print("Shape of dataset:", df.shape)

# Class distribution (Fraud vs Non-Fraud)
print("\nClass Distribution:")
print(df['Class'].value_counts())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Number of fraud and non-fraud transactions
fraud_counts = df['Class'].value_counts()
num_non_fraud = fraud_counts[0]
num_fraud = fraud_counts[1]

print(f"Number of Non-Fraud transactions in current datasest: {num_non_fraud}")
print(f"Number of Fraud transactions in current datasest: {num_fraud}")


# Summary statistic

selected_features = ['Time', 'Amount', 'V1', 'V2']
summary_stats = df[selected_features].describe()

print("\n Summary Statistics:")
print(summary_stats)


#Rishit

#EDA Visuals

import seaborn as sns
import matplotlib.pyplot as plt

#  Visualization 1: Histogram of Transaction Amount
plt.figure(figsize=(6,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.tight_layout()
print("Histogram")
plt.show()
print( )

# Visualization 2: Distribution of Transaction Time
plt.figure(figsize=(6,4))
sns.kdeplot(df['Time'], fill=True)  # fill=True makes it a filled curve
plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds)')
plt.ylabel('Density')
plt.tight_layout()
print("Distribution")
plt.show()
print( )


# Visualization 3: Bar chart of Fraud vs Non-Fraud transactions
fraud_pct = df['Class'].value_counts(normalize=True) * 100
fraud_pct.plot(kind='bar')
plt.title('Fraud vs Non-Fraud Transaction Percentage')
plt.ylabel('Percentage (%)')
print("Bar chart")
plt.show()
print( )


# Visualization 4: correlation chart of Fraud vs Non-Fraud transactions
sns.scatterplot(x='Amount', y='Class', data=df, alpha=0.5)
plt.title('Correlation between Transaction Amount and Fraud Class')
plt.xlabel('Transaction Amount')
plt.ylabel('Fraud (1) / Non-Fraud (0)')
print("Correlation Chart")
plt.tight_layout()
plt.show()



#PHASE 2 STARTS FROM HERE
#Rishit
# Model from scratch,
# Data Preprocessing:

#Removing duplicates function

def remove_duplicates(df):
    bef = len(df)
    df_clean = df.drop_duplicates()
    aft = len(df_clean)
    print(f"Removed {bef - aft} duplicate rows.")
    return df_clean

df=remove_duplicates(df)

#Rishit
#Removing missing values

df = df.dropna(subset=['Class'])
print("Dropped rows with missing Class label.")

#seperating numerical and categorical columns
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

#Rishit
#impying mean imputation for numerical values to handle continuous variables

def impute_missing_mean(df, num_cols):
    """
    Replaces missing values in numeric columns with the mean of each column.
    """
    for col in num_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    print("Mean imputation applied to numeric columns.")
    return df

df = impute_missing_mean(df, num_cols)
print("Missing values after imputation:")
print(df.isnull().sum())

# print("Missing values BEFORE:")
# print(missing_cols)

# print("Missing values AFTER:")
# print(df.isnull().sum().sum())

#Rishit
#Scaling the data

#1) scaling the time using mix-max scaler method

def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

#2) scaling amount using z-score
def z_score_scale(series):
    mean_val = series.mean()
    std_val = series.std()
    return (series - mean_val) / std_val

df['Time'] = min_max_scale(df['Time'])
df['Amount'] = z_score_scale(df['Amount'])

print(df[['Time', 'Amount']].describe())

# sagar
# Train Random Forest and Logistic Regression

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#preprocess the dataset

def impute_missing_mean(df, num_cols):
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

def z_score_scale(series):
    return (series - series.mean()) / series.std()

df = df.drop_duplicates()
df = df.dropna(subset=['Class'])
num_cols = df.select_dtypes(include='number').columns.tolist()
df = impute_missing_mean(df, num_cols)
df['Time'] = min_max_scale(df['Time'])
df['Amount'] = z_score_scale(df['Amount'])

# split the dataset

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train logistic regression model

log_reg = LogisticRegression(max_iter=500, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))

print("\n")

# train random forest model

rf = RandomForestClassifier(
    n_estimators=40,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))

# Franco Miguel
# Logistic Regression implemented from scratch by using gradient descent method.
# Trains and evaluates the from-scratch Logistic Regression model
# Binary classification labels: 0 and 1.


#SCRATCH MODEL IMPLEMANTAION

class ScratchLogisticRegression:


    def __init__(self, lr=0.01, num_epochs=1000, verbose=False):
        self.lr = lr # learning rate: controls size of each update step is during gradient descent.
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function.
        Purpose: Converts any real number into a probability between 0 and 1.
        Formula: sigmoid(z) = 1 / (1 + e^-z)
        """
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        """
        Loss function (binary cross entropy)
        Formula: L = -1/m * Σ [y*log(p) + (1-y)*log(1-p)]

        - y_true : true label (0 or 1)
        - y_pred : probability label (between 0 and 1)
        - eps is for log(0), that's a numerical error.
        """
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        m = y_true.shape[0] # number of samples

        # Cross-entropy formula
        loss = - (1.0 / m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
        """
        Learning: update w, b by using gradient descent.
        X: shape (m, d) // m = number of samples and d = number of features
        y: shape (m,)
        The model learns:
        - w : weight vector
        - b : bias term

        Learning steps:
        1. Computing predictions
        2. Measuring loss
        3. Calculating gradients
        4. Updating w and b
        5. Repeating for many epochs
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # m = number of samples, d = number of features
        m, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.num_epochs):
            z = X.dot(self.w) + self.b           # linear combo (m,)
            y_hat = self._sigmoid(z)             # predicted probabilities

            #loss
            loss = self._binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            #gradients
            error = y_hat - y
            dw = (1.0 / m) * X.T.dot(error)
            db = (1.0 / m) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # print the progress for every 100 epochs if verbose=True
            if self.verbose and (epoch % 100 == 0 or epoch == self.num_epochs - 1):
                print(f"Epoch {epoch:4d}  Loss = {loss:.6f}")

    def predict_probability(self, X):

        X = np.array(X, dtype=float)
        z = X.dot(self.w) + self.b       # output of the linear model
        return self._sigmoid(z)             # converts to probability

    def predict(self, X, threshold=0.5):
        probability = self.predict_probability(X)
        return (probability >= threshold).astype(int)

#TRAINING AND EVALUATING SCRATCH MODEL

print("\nTraining Logistic Regression (scratch)...")

scratch_learningrate = ScratchLogisticRegression(
    lr=0.01,
    num_epochs=1000,
    verbose=True   # set false if its too noisy
)
# learning: the model adjusts weights using the training training data
scratch_learningrate .fit(X_train, y_train)

# testing: model makes the predictions on unseen test data
y_pred_scratch = scratch_learningrate .predict(X_test, threshold=0.5)

# print regression results
print("\nScratch Logistic Regression Results\n")
print("F1-score :", f1_score(y_test,y_pred_scratch,zero_division=0))
print("Accuracy :", accuracy_score(y_test,y_pred_scratch))
print("Recall   :", recall_score(y_test,y_pred_scratch,zero_division=0))
print("Precision:", precision_score(y_test,y_pred_scratch,zero_division=0))

# print classification report
print("\nClassification report:\n")
print(classification_report(y_test,y_pred_scratch,zero_division=0))
