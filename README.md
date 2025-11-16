# FUSEX – Sampling Technique Comparison

## Collaborators
- Mannan Aggrawal  
- Purav Shah  
- Jay Salot  
- Neel Shah  

---

## Project Overview
This repository compares three techniques used for handling imbalanced datasets:

- **Random Sampling**
- **SMOTE**
- **Cluster-Based Sampling**

The objective is to identify which method gives the most stable and accurate performance with models like XGBoost and RandomForest.

---

## Why Random Sampling Performed Best

### 1. Preserves True Data Structure
Random sampling keeps real patterns unchanged, helping the model learn clean, natural boundaries.

### 2. SMOTE Added Synthetic Noise
SMOTE generated artificial minority points that often fell in overlapping or noisy regions, lowering model accuracy.

### 3. Cluster Sampling Distorted Boundaries
Cluster-based sampling mixed class distributions within clusters, affecting class separation and balancing.

### 4. Works Better With Tree-Based Models
XGBoost and RandomForest handle imbalance well, and simple random sampling complements their learning behavior more effectively.

### 5. Faster and More Stable
Random Sampling is computationally light and avoids the instability introduced by synthetic generation or clustering.

---

## Conclusion
Random Sampling delivered the most reliable performance because it maintains natural data structure, avoids noise, and works smoothly with tree-based models.

