# 🧠 Personality Finder

## 📌 Project Overview  
Personality Finder is a machine learning project designed to predict a person's personality type based on input values.  
The system includes preprocessing, feature selection, model training, evaluation, and real-time prediction using saved model files.

---

## 🛠️ Technologies Used  
- Python  
- Machine Learning  
- Scikit-Learn  
- NumPy & Pandas  
- Logistic Regression Model  

---

## 🔍 Project Workflow  
1. Data Collection  
2. Data Cleaning & Preprocessing  
3. Feature Scaling & Selection  
4. Model Training  
5. Model Saving (`.pkl` files)  
6. Predicting New Input Values  

---

## 🤖 Machine Learning Model  

| Model | Status | Accuracy |
|--------|--------|----------|
| Logistic Regression | ✔ Final Selected Model | *98%* |

---

## 📂 Project Files  

| File Name | Description |
|-----------|------------|
| `personality_finder.py` | Script used for prediction |
| `PersonalityModel__LogisticReg.ipynb` | Model training file |
| `personality_model.pkl` | Trained ML model used for predictions |
| `scaler.pkl` | Standard scaler for preprocessing |
| `Selected_features.pkl` | Stored selected feature names |
| `personality_synthetic_dataset.csv` | Dataset used for model training |
| `requirements.txt` | Required dependencies |

---

## ▶️ How to Run  

1. Install dependencies  
   ```bash
   pip install -r requirements.txt
## 🎯 Uses

This system can be used in multiple real-world applications such as:

- HR Screening to understand candidates' behavior patterns
- Chatbot Personalization for tailoring conversations
- Educational Tools to recommend learning styles
- Game Character Personalization based on user traits
- Research & Behavioral Analytics for psychology studies

## 🚀 Future Enhancements
Planned improvements for system expansion:
- Add Advanced Models (Decision Trees, Random Forest, or Deep Learning)
- Use Real-World Personality Datasets (MBTI, Big Five, etc.)
- Build a User-Friendly Interface for Inputs
- Develop a Mobile App Version

---

## 🖥 App Preview

### 🧠 Streamlit UI – Input Page

<p align="center">
  <img src="<img width="1918" height="1017" alt="Screenshot 2025-11-28 154114" src="https://github.com/user-attachments/assets/6c12674c-fa06-44d3-b7b2-ef1eb4eb7d92" />
" alt="Personality Finder Input Page" width="600">
</p>

Users adjust the slider-based personality feature values.

---

### 🎯 Prediction Page

<p align="center">
  <img src="<img width="1919" height="912" alt="Screenshot 2025-11-28 154152" src="https://github.com/user-attachments/assets/eef0fb05-f0de-40b5-840a-4ac0a3851ef5" />
" alt="Prediction Result Page" width="600">
</p>

Once submitted, the model predicts the personality type and displays a short description.



---
