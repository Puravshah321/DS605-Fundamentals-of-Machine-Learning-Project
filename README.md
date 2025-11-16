# 💸 Money Laundering Detection – FUSEX  
A Machine Learning Project on Imbalanced Classification, CO₂ Efficiency, Sampling Strategy Evaluation & Model Optimization

---

## 👥 Collaborators  
**Team FUSEX**  
- Mannan Aggrawal (202518013)  
- Purav Shah (202518020)  
- Jay Salot (202518029)  
- Neel Shah (202518044)  

---

## 📂 Dataset  
IBM Anti-Money Laundering (AML) Transactions Dataset  

🔗 **Kaggle Link:**  
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data?select=HI-Small_Trans.csv  

---

# 📌 Project Overview  
Money laundering detection is a rare-event classification problem. Fraudulent transactions represent **<1%** of the dataset → making it **extremely imbalanced**.

### Objectives:
- Build a scalable AML detection pipeline  
- Handle imbalance using different sampling techniques  
- Train, compare & tune multiple ML models  
- Reduce false negatives (Type-II errors)  
- Measure and compare **CO₂ emissions**  
- Select the best model based on accuracy, recall & carbon-efficiency  

---

# 🔍 Exploratory Data Analysis (EDA)

## **Univariate Analysis**
- Transaction amount highly **right-skewed**  
- Categorical features show **very high cardinality**  
- Fraud transactions had **higher mean values**  
- Multiple numeric features showed **class separation**  

## **Bivariate Analysis**
- Fraud correlated strongly with higher transaction amounts  
- Specific customer segments exhibited higher fraud rates  
- Correlation heatmap revealed strong numeric relationships  
- Bivariate trends favored **tree-based models** like LightGBM  

---

# ⚖️ Sampling Techniques Tried

## **1. Random Undersampling (RUS)**  
✓ Fastest  
✓ No synthetic noise  
✓ Works best with tree-based models  
✓ Best confusion matrix stability  
→ **Chosen as final method**

## **2. SMOTE (Numeric Only) + NN Categorical Matching + KMeans**
✗ Introduced synthetic noise  
✗ Very heavy computation  
✗ Lower F1  

## **3. Full SMOTE**  
✗ Unrealistic synthetic samples  
✗ Overlapping boundaries  
✗ Poor generalization  

---

# ⭐ Why We Selected Random Undersampling (RUS)

### ✔️ Highest recall + lowest false negatives  
### ✔️ Cleanest decision boundaries  
### ✔️ Best model performance with tree algorithms  
### ✔️ Lowest CO₂ emissions  
### ✔️ Fastest training time  
### ✔️ Most stable confusion matrix  

---

# ⚙️ Model Training Pipeline

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.3, random_state=42
)

ohe = Pipeline([
    ("Encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

transformer = ColumnTransformer([
    ("OneHot", ohe, cat)
])

model = Pipeline([
    ("Transformer", transformer),
    ("Estimator", XGBClassifier())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

# 🤖 Models Trained

We trained 9 ML models:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- GradientBoosting  
- AdaBoost  
- KNN  
- SVM  
- XGBoost  
- LightGBM  

---

# 📊 Performance Comparison Table (Before Tuning)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Type I Error | Type II Error |
|-------|----------|-----------|--------|------|----------|----------------|----------------|
| Logistic Regression | ~0.78 | Low | Moderate | Low | Moderate | High | High |
| Decision Tree | ~0.82 | Moderate | Moderate | Moderate | Moderate | Medium | Medium |
| Random Forest | ~0.86 | Good | Moderate | Good | Good | Medium | Medium |
| GradientBoosting | **0.88** | High | **0.97** | **0.90** | High | Low | Low |
| AdaBoost | ~0.84 | Moderate | Moderate | Moderate | Moderate | Medium | Medium |
| KNN | ~0.80 | Low | Low | Low | Low | High | High |
| SVM | 0.89 | **0.82** | **0.99** | **0.90** | High | Medium | **Very Low** |
| XGBoost | 0.87 | Good | High | High | High | Low | Low |
| LightGBM | **0.90** | **0.85** | **0.98** | **0.91** | **Highest** | **Lowest** | Very Low |

---

# 🏆 Top 3 Models After Hyperparameter Tuning

| Rank | Model | F1 Score | Recall | Accuracy | CO₂ Emissions | Notes |
|------|--------|----------|--------|----------|----------------|--------|
| **1** | **LightGBM (Tuned)** | **0.91** | **0.98** | **0.90** | ⭐ Lowest | Best overall performer |
| **2** | **GradientBoosting (Tuned)** | **0.90** | **0.97** | **0.88** | Moderate | Very stable & strong recall |
| **3** | **SVM (Linear Kernel, Tuned)** | **0.90** | **0.99** | **0.89** | ❗ Highest | Excellent recall but too costly |

---

# 📸 Confusion Matrices (Tuned Models)

### **SVM – Tuned**
<img width="534" src="https://github.com/user-attachments/assets/c31209d1-87ce-4848-8872-4f5237186cfc">

### **GradientBoosting – Tuned**
<img width="534" src="https://github.com/user-attachments/assets/072d0963-afb1-4b48-9cb7-845ca549791e">

### **LightGBM – Tuned**
<img width="534" src="https://github.com/user-attachments/assets/02704b14-c089-4a62-9b34-a4364a4eb16a">

---

# 🌱 CO₂ Emission Comparison

<img width="684" src="https://github.com/user-attachments/assets/b77c99c3-8640-4ad6-95bc-89e7e5689960">

### Insights:
- **LightGBM = Lowest CO₂**  
- GradientBoosting moderate  
- **SVM = Highest CO₂** → Not ideal for production  

---

# 🔧 Threshold Tuning (Reducing FN)

### Recommended Threshold: **0.38**

### Confusion Matrix at 0.38:
```
[[1257  329]
 [  36 1485]]
```

### Key Benefits:
- FN reduced from **45 → 36**  
- Huge recall improvement  
- Slight FP increase acceptable in AML  

---

# 🔥 Hyperparameter Tuning

### Example (SVM):
```
Best Params:
{'clf__kernel': 'linear', 'clf__C': 0.1}
```

Tuning improved:
- F1 Score  
- Recall  
- Model stability  

---

# 🧩 Difficulties We Faced & Solutions

### 1️⃣ Extreme Imbalance  
✅ Solution: Switched from SMOTE → RUS  

### 2️⃣ Categorical SMOTE Complexity  
❌ Too slow, too noisy → Dropped  

### 3️⃣ High CO₂ Usage in SVM  
❌ Not suitable for production  

### 4️⃣ Threshold Optimization  
✅ Full sweep performed → Found 0.38  

### 5️⃣ High Cardinality  
✅ OneHotEncoder(drop="first") used  

---

# 🆕 Novelty of Our Approach

- **CO₂-aware ML model selection**  
- Custom **Safe-SMOTE** implementation  
- Full **threshold sweep** for minimizing Type-II errors  
- Unified, reusable ML pipeline  

---

# 🏁 Final Conclusion

### 🟩 **Best Sampling Method → Random Undersampling (RUS)**  
### 🟩 **Best Overall Model → LightGBM (91% F1, 98% Recall, Lowest CO₂)**  
### 🟨 **Runner-Up → GradientBoosting (Strong recall, stable performance)**  
### 🟧 **High Recall but Not Selected → SVM (Too high CO₂ emission)**  
### ❌ **Avoid → SMOTE + KMeans (Noisy, slow, bad real-world stability)**  

**LightGBM was chosen as the final model because it delivered:**  
✓ Best F1 Score (0.91)  
✓ Extremely high recall (0.98)  
✓ Lowest false negatives  
✓ Lowest CO₂ emissions  
✓ Fastest inference → suitable for real-time AML systems  

---
