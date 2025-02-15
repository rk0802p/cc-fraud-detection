# Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent transactions using machine learning techniques. Due to the highly imbalanced nature of fraud detection datasets, special preprocessing steps and evaluation metrics are used to ensure accurate model performance.

## Dataset Information

The dataset used in this study contains anonymized transaction details, including time, amount, and engineered features. The target variable, `Class`, indicates whether a transaction is fraudulent (`1`) or legitimate (`0`).

## Project Phases

The project is divided into six phases:

### 1. Load Data

```python
import pandas as pd
# Load the dataset
df = pd.read_csv('creditcard.csv')
# Inspect dataset
df.info()
df.head()
```

### 2. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Class distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()
```

### 3. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Feature Engineering
df['Hour'] = df['Time'] / 3600 % 24
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split data
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training

Three different models were trained and evaluated:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
```

### 5. Model Evaluation

Each model was evaluated using classification metrics:

```# Select the best performing model (assuming XGBoost here)
best_model = XGBClassifier(eval_metric='logloss', random_state=42)
best_model.fit(x_train_final, y_train_final)

# Test Set Prediction
y_test_pred = best_model.predict(x_test)
y_test_prob = best_model.predict_proba(x_test)[:, 1]
```

### 6. Deployment & Conclusion

- Summarize findings and key takeaways from the model performance.
- Discuss potential improvements and real-world applications.

## Implementation Details

- The project is implemented in Python using libraries such as:
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn` for visualization
  - `scikit-learn`, `xgboost` for machine learning

## How to Use

1. Clone the repository and navigate to the project folder.
   ```sh
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install required dependencies using:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to process data and train models.
   ```sh
   jupyter notebook
   ```

## Requirements

- Python 3.8+
- Jupyter Notebook
- Required libraries:
  ```sh
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
  ```

## Results

- The best-performing model achieved high recall, ensuring most fraudulent transactions were detected.
- Further work can be done on feature engineering and advanced deep learning models.

---

**Note:** The dataset can be downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data"
