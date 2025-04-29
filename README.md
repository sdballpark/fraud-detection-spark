# 🏦 Fraud Detection with PySpark and Explainable AI

This project builds a machine learning model using PySpark MLlib to predict fraudulent transactions.  
It addresses class imbalance using SMOTE and evaluates performance with AUC-ROC, confusion matrix, and accuracy metrics.

Includes a ready-to-deploy Streamlit app for real-time fraud detection.

## 📂 Project Structure
- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for EDA and Modeling
- `src/` - Data preprocessing, model training, inference modules
- `app/` - Streamlit web application

## 🚀 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Launch Jupyter and run EDA/Modeling notebooks:
```bash
jupyter notebook notebooks/1_EDA.ipynb
jupyter notebook notebooks/2_Modeling.ipynb
```
3. Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## 🛠 Technologies Used
- PySpark MLlib
- Imbalanced-learn (SMOTE workaround)
- Streamlit
- Pandas, Numpy, Matplotlib

## 🔥 Future Work
- Implement full hyperparameter tuning (GridSearchCV-style equivalent in PySpark)
- Expand model training to include Gradient Boosted Trees and Stacking Ensembles
- Introduce logging and monitoring pipelines for model health tracking
- Add SHAP explainability integration (custom workaround for Spark models)
- Deploy Streamlit app to a cloud platform (AWS EC2 or Azure WebApp)
