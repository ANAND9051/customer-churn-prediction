# 🛡️ End-to-End Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-red)

## 📊 Project Overview
This project is an end-to-end Machine Learning solution designed to predict customer churn for a Telecom provider. Churn prediction is a critical business task—retaining a customer is **5x cheaper** than acquiring a new one.

### Key Highlights:
- **Class Imbalance Handling:** Used **SMOTE** to balance a dataset with only 26.5% churners.
- **Model Interpretability:** Identified that **Tenure**, **Electronic Checks**, and **Month-to-Month contracts** are the primary drivers of churn.
- **Real-Time Deployment:** Built a **Streamlit Web App** for proactive customer retention.

---

## 🏗️ Project Architecture
```text
customer-churn-prediction/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory Data Analysis (EDA) & Visualizations
├── src/                # Modular logic
│   ├── preprocess.py   # Cleaning, encoding, and scaling
│   ├── train.py        # Model training and SMOTE balancing
│   └── interpret.py    # Feature importance analysis
├── app/                # Streamlit UI
└── models/             # Saved model artifacts (.pkl)
```

---

## 🚀 How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App:
```bash
streamlit run app/main.py
```

---

## 📈 Insights Found
- **Payment Friction:** Customers using **Electronic Checks** have a significantly higher churn risk.
- **Loyalty Program:** Customers with **Two-Year contracts** are the most stable, suggesting that locking in customers with multi-year discounts is an effective strategy.

---

## 🛠️ Tech Stack
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Deployment:** Streamlit
- **Environment Management:** uv
