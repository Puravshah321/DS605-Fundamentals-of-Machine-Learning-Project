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
Money laundering detection is a rare-event classification problem. Fraudulent transactions represent **less than 1%** of the dataset, making it **highly imbalanced**.

### Our main objectives were:  
- Build a scalable AML detection pipeline  
- Handle imbalance using multiple sampling techniques  
- Train, compare, and tune ML models  
- Reduce false negatives (Type-II errors)  
- Measure **CO₂ emissions** for green AI  
- Build a production-ready AML solution  

---

# 🔍 Exploratory Data Analysis (EDA)

## **Univariate Analysis**
We examined each feature individually to understand:
- Skewness & outliers  
- High-cardinality categorical columns  
- Distribution differences between fraud & non-fraud  
- Missing values  

### Key Insights:
- Transaction amounts are **heavily right-skewed**  
- Many categorical columns have **large cardinality**  
- Fraud transactions slightly differ in numeric summaries  
- Several important features show strong class separation  

---

## **Bivariate Analysis**
We explored:
- Relationship of amount vs fraud  
- Customer segment vs fraud  
- Frequency of categorical levels for fraudulent users  
- Pairwise correlations among numerics  

### Insights:
- Clear numeric separation between laundering and normal transactions  
- Certain categorical segments are more fraud-prone  
- Tree-based models (GradientBoosting, LightGBM) seemed ideal  

---

# ⚖️ Sampling Techniques Tried

## **1. Random Undersampling (RUS)**  
- Removes majority samples  
- Fastest method  
- Produces clean, balanced datasets  
- Minimal noise → high performance  
- **Final chosen method**

---

## **2. SMOTE (Numeric Only) + Categorical Matching + KMeans**  
We built a **custom Safe-SMOTE pipeline**:

### Steps:
- Apply SMOTE on numeric columns only  
- Use Nearest Neighbors to assign categorical values to synthetic points  
- Use StandardScaler → KMeans for clustering  
- Merge numeric + categorical + cluster labels  

### Drawbacks:
- Introduced noise  
- Heavy computation time  
- Lower performance  

---

## **3. Full SMOTE**
- Synthetic points not realistic  
- Increased overlapping & model confusion  
- Poor generalization  

---

# ⭐ Why We Selected Random Undersampling (RUS)

### ✔️ Best recall & lowest false negatives  
### ✔️ No synthetic noise  
### ✔️ Very fast & CO₂ efficient  
### ✔️ Works extremely well with LightGBM & GradientBoosting  
### ✔️ Best confusion matrix stability  

---

# ⚙️ Model Training Pipeline

We used a shared preprocessing + estimator pipeline:

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

Ensures consistent preprocessing across all models.

---

# 🤖 Models Trained

We trained and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- GradientBoosting  
- AdaBoost  
- KNN  
- SVM  
- XGBoost  
- LightGBM  

### Metrics calculated:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Type I (False Positive) rate  
- Type II (False Negative) rate  
- CO₂ Emissions  

---

# 🏆 Top 3 Models (After Hyperparameter Tuning)

1. **LightGBM** – best overall + lowest CO₂  
2. **GradientBoosting** – excellent recall & F1  
3. **SVM (Linear)** – high recall but high CO₂  

---

# 📊 Confusion Matrices (Tuned Models)

### **SVM – Tuned**
<img width="534" height="437" alt="image" src="https://github.com/user-attachments/assets/c31209d1-87ce-4848-8872-4f5237186cfc" />


### **GradientBoosting – Tuned**
<img width="534" height="437" alt="image" src="https://github.com/user-attachments/assets/072d0963-afb1-4b48-9cb7-845ca549791e" />


### **LightGBM – Tuned**
<img width="534" height="437" alt="image" src="https://github.com/user-attachments/assets/02704b14-c089-4a62-9b34-a4364a4eb16a" />


---

# 🌱 CO₂ Emission Comparison

<img width="684" height="384" alt="image" src="https://github.com/user-attachments/assets/b77c99c3-8640-4ad6-95bc-89e7e5689960" />


### Insights:
- **LightGBM → Most eco-friendly**  
- GradientBoosting → Moderate CO₂  
- **SVM → Highest CO₂ footprint**  

---

# 🔧 Threshold Tuning

A threshold sweep (0.01 → 0.99) was used to minimize false negatives.

### **Recommended Threshold: 0.38**

### Confusion Matrix at 0.38:
```
[[1257  329]
 [  36 1485]]
```

### Benefits:
- False negatives reduced **from 45 → 36**  
- Higher recall  
- Better fraud capturing capability  

---

# 🔥 Hyperparameter Tuning

We used **RandomizedSearchCV (25 iterations)** for the top 3 models.

### Example (SVM):
```
Best Params:
{'clf__kernel': 'linear', 'clf__C': 0.1}
```

Significantly improved:
- F1  
- Recall  
- Boundary stability  

---

# 🧩 Difficulties We Faced & Solutions

### ❌ 1. Extreme Imbalance  
SMOTE added noise → switched to RUS.

### ❌ 2. Complex Categorical SMOTE  
Too slow + unwanted noise → dropped.

### ❌ 3. High CO₂ in SVM  
Still used for evaluation, not final pipeline.

### ❌ 4. Threshold Selection  
Solved using threshold vs. FN sweep.

### ❌ 5. High Cardinality  
Handled using OneHotEncoder(drop="first").

---

# 🆕 Novelty of Our Approach

### 🌟 CO₂-aware ML model selection  
### 🌟 Custom Safe-SMOTE pipeline (numeric SMOTE + NN categorical merge)  
### 🌟 Threshold tuning to reduce Type-II error (critical in AML)  
### 🌟 Unified modeling pipeline with consistent encoding  

---

# 🏁 Final Conclusion

### ✅ Best Sampling Method → **Random Undersampling**  
### ✅ Best Model → **LightGBM**  
### ✅ Best CO₂ Efficiency → **LightGBM**  
### ✅ Best F1/Recall → **GradientBoosting**  
### ❌ Avoid → SMOTE + KMeans (too noisy & expensive)  

The final pipeline is **accurate, carbon-efficient, stable, and production-ready for AML detection**.

---
