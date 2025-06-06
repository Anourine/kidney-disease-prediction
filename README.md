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

[View Live App](https://your-app-name.streamlit.app) *(Update this link after deployment)*

## 📊 Dataset

The model is trained on a comprehensive kidney disease dataset with features including:
- Blood pressure measurements
- Laboratory test results (creatinine, glucose, etc.)
- Urine analysis parameters
- Medical history indicators
- Demographic information

## 🛠️ Installation & Setup

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kidney-disease-prediction.git
cd kidney-disease-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

### Deploy on Streamlit Cloud

1. Fork/clone this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically!

## 📁 Project Structure

```
kidney-disease-prediction/
│
├── streamlit_app.py          # Main Streamlit application
├── kidney_model_pipeline.pkl # Trained ML model
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── data/                    # Dataset files (optional)
```

## 🔧 Technical Details

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
