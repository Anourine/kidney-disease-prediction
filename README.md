# 🩺 Kidney Disease Prediction System

A machine learning-powered web application that predicts the risk of kidney disease based on various medical parameters.

## 🎯 Features

- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Machine Learning Prediction**: Uses trained Logistic Regression model
- **Real-time Results**: Instant predictions with confidence scores
- **Medical Parameter Input**: Comprehensive form for patient data
- **Risk Assessment**: Clear HIGH/LOW risk categorization

## 🔬 Model Performance

- **Algorithm**: Logistic Regression (Best performing model)
- **Accuracy**: ~95%+
- **Features**: 42+ medical parameters
- **Cross-validation**: 5-fold CV implemented

## 🚀 Live Demo

[View Live App]([https://your-app-name.streamlit.app](https://kidney-disease-prediction-34uyhbxv2skdqpyjhrkfnp.streamlit.app/)) *(Update this link after deployment)*

## 📊 Dataset

The model is trained on a comprehensive kidney disease dataset with features including:
- Blood pressure measurements
- Laboratory test results (creatinine, glucose, etc.)
- Urine analysis parameters
- Medical history indicators
- Demographic information

### Model Training Pipeline
- Data preprocessing with missing value imputation
- Feature scaling using StandardScaler
- SMOTE for handling class imbalance
- Grid search for hyperparameter tuning
- Cross-validation for model evaluation

### Models Compared
- Logistic Regression ✅ (Best)
- Decision Tree
- Random Forest
- XGBoost

## ⚕️ Medical Disclaimer

This application is for educational and informational purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
