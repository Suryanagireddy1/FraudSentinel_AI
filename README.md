# 🛡️ FraudSentinel AI — Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Deployed-black?style=flat&logo=flask)
![Render](https://img.shields.io/badge/Render-Live-brightgreen?style=flat&logo=render)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat)

> A machine learning web application that detects fraudulent financial transactions in real time using classification algorithms, deployed live on Render.

🔗 **Live App:** [fraudsentinel-ai.onrender.com](https://fraudsentinel-ai.onrender.com)  
📁 **GitHub:** [github.com/Suryanagireddy1/FraudSentinel_AI](https://github.com/Suryanagireddy1/FraudSentinel_AI)

---

## 📌 Project Overview

Financial fraud costs businesses billions every year. FraudSentinel AI is a machine learning system that analyzes transaction data and predicts whether a transaction is **fraudulent or legitimate** — in real time through a web interface.

This project covers the full ML pipeline:
- Data preprocessing & cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training & evaluation
- Hyperparameter tuning
- Web deployment

---

## 🎯 Problem Statement

Given a set of transaction features (amount, time, anonymized PCA features), predict whether a transaction is:
- `0` — Legitimate
- `1` — Fraudulent

The dataset is highly imbalanced (fraudulent transactions are < 1% of total), making this a challenging real-world classification problem.

---

## 🗂️ Project Structure

```
FraudSentinel_AI/
│
├── app.py                  # Flask web application
├── model.py                # Model training script
├── fraud_model.pkl         # Trained model (saved)
├── scaler.pkl              # Feature scaler
│
├── data/
│   └── creditcard.csv      # Dataset
│
├── notebooks/
│   └── fraud_detection.ipynb  # EDA & model development
│
├── templates/
│   └── index.html          # Web UI
│
├── static/
│   └── style.css           # Styling
│
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Library | Scikit-learn |
| Algorithms | Random Forest, XGBoost, Logistic Regression |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Deployment | Render |
| Model Saving | Pickle |

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- Handled missing values and duplicate records
- Scaled `Amount` and `Time` features using StandardScaler
- Analyzed class distribution (highly imbalanced dataset)

### 2. Exploratory Data Analysis
- Correlation heatmap of all features
- Distribution plots for fraudulent vs legitimate transactions
- Transaction amount analysis by class

### 3. Feature Engineering
- Normalized skewed features
- Applied SMOTE to handle class imbalance
- Selected top features based on feature importance scores

### 4. Model Training & Evaluation

| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| Logistic Regression | 97.8% | 0.96 | 0.84 |
| Random Forest | 99.2% | 0.98 | 0.87 |
| **XGBoost (Final)** | **99.5%** | **0.99** | **0.88** |

### 5. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Tuned: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Suryanagireddy1/FraudSentinel_AI.git
cd FraudSentinel_AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🌐 Live Demo

The app is deployed on **Render** and accessible at:  
👉 [https://fraudsentinel-ai.onrender.com](https://fraudsentinel-ai.onrender.com)

Enter transaction details in the form and get an instant prediction — **Fraudulent** or **Legitimate**.

---

## 📊 Key Results

- ✅ **99.5% accuracy** on test data
- ✅ **AUC-ROC: 0.99** — excellent discrimination between classes
- ✅ **F1-Score: 0.88** on the minority (fraud) class
- ✅ Minimized false negatives — critical in fraud detection

---

## 📈 Future Improvements

- [ ] Add real-time transaction streaming using Kafka
- [ ] Integrate SHAP explainability dashboard
- [ ] Add user authentication for the web app
- [ ] Retrain model periodically with new transaction data
- [ ] Add email alert system for detected fraud

---

## 👨‍💻 Author

**Surya Nagireddy**  
MSc Computer Science — Dravidian University  
📧 suryanagireddy7564@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/surya-nagireddy-568728245)  
🐙 [GitHub](https://github.com/Suryanagireddy1)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this project helpful, please give it a star!**
