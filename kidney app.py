import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_paths = [
            'kidney_model_pipeline.pkl',
            'models/kidney_model_pipeline.pkl',
            'best_model_logistic_regression.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                st.success(f"Model loaded from: {path}")
                return model
        
        st.error("Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_feature_names():
    try:
        feature_file_paths = [
            'feature_names.pkl',
            'feature_names.txt',
            'features.pkl'
        ]
        
        for path in feature_file_paths:
            if os.path.exists(path):
                if path.endswith('.pkl'):
                    feature_names = joblib.load(path)
                    st.info(f"Feature names loaded from: {path}")
                    return feature_names
        
        st.warning("Feature names file not found. Using default names.")
        return ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    except Exception as e:
        st.error(f"Error loading feature names: {str(e)}")
        return []

def main():
    st.title("🩺 Kidney Disease Prediction System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    feature_names = load_feature_names()
    
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Patient Information")
        
        with st.form("prediction_form"):
            # Basic Information
            st.markdown("**Basic Information**")
            age = st.number_input("Age of the patient", min_value=0, max_value=120, value=50)
            bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=200, value=120)
            
            # Laboratory Tests
            st.markdown("**Laboratory Tests**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                specific_gravity = st.number_input("Specific gravity of urine", min_value=1.0, max_value=1.5, value=1.02, step=0.001)
                albumin = st.selectbox("Albumin in urine", [0, 1, 2, 3, 4, 5])
                sugar = st.selectbox("Sugar in urine", [0, 1, 2, 3, 4, 5])
                rbc = st.selectbox("Red blood cells in urine", ["normal", "abnormal"])
                
            with col_b:
                pus_cell = st.selectbox("Pus cells in urine", ["normal", "abnormal"])
                bacteria = st.selectbox("Bacteria in urine", ["present", "notpresent"])
                bg_random = st.number_input("Random blood glucose level (mg/dl)", min_value=50, max_value=500, value=120)
                blood_urea = st.number_input("Blood Urea (mg/dl)", min_value=10, max_value=200, value=30)
            
            # More tests
            st.markdown("**Additional Tests**")
            col_c, col_d = st.columns(2)
            
            with col_c:
                serum_creatinine = st.number_input("Serum Creatinine (mg/dl)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
                sodium_level = st.number_input("Sodium level (mEq/L)", min_value=100, max_value=200, value=140)
                potassium_level = st.number_input("Potassium level (mEq/L)", min_value=2.0, max_value=10.0, value=4.0, step=0.1)
                
            with col_d:
                hemoglobin = st.number_input("Hemoglobin level (gms)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
                pcv = st.number_input("Packed cell volume (%)", min_value=20, max_value=60, value=40)
                wbc_count = st.number_input("White blood cell count (cells/cumm)", min_value=2000, max_value=20000, value=8000)
            
            # Medical History
            st.markdown("**Medical History**")
            col_e, col_f = st.columns(2)
            
            with col_e:
                hypertension = st.selectbox("Hypertension", ["yes", "no"])
                diabetes = st.selectbox("Diabetes Mellitus", ["yes", "no"])
                
            with col_f:
                cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
                appetite = st.selectbox("Appetite", ["good", "poor"])
            
            # Submit button
            submitted = st.form_submit_button("🔬 Predict Kidney Disease", use_container_width=True)
            
            if submitted:
                try:
                    # Convert categorical variables to numerical
                    rbc_encoded = 1 if rbc == "abnormal" else 0
                    pus_cell_encoded = 1 if pus_cell == "abnormal" else 0
                    bacteria_encoded = 1 if bacteria == "present" else 0
                    hypertension_encoded = 1 if hypertension == "yes" else 0
                    diabetes_encoded = 1 if diabetes == "yes" else 0
                    cad_encoded = 1 if cad == "yes" else 0
                    appetite_encoded = 1 if appetite == "poor" else 0
                    
                    # Create input array
                    input_data = np.array([[
                        age, bp, specific_gravity, albumin, sugar, rbc_encoded,
                        pus_cell_encoded, 0, bacteria_encoded, bg_random, blood_urea,
                        serum_creatinine, sodium_level, potassium_level, hemoglobin, pcv, wbc_count,
                        0, hypertension_encoded, diabetes_encoded, cad_encoded, appetite_encoded,
                        0, 0
                    ]])
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    
                    # Display results
                    with col2:
                        st.subheader("🔬 Prediction Results")
                        
                        if prediction == 1:
                            st.error("🚨 **HIGH RISK** of Kidney Disease")
                            st.markdown(f"**Confidence:** {probability[1]:.2%}")
                        else:
                            st.success("✅ **LOW RISK** of Kidney Disease")
                            st.markdown(f"**Confidence:** {probability[0]:.2%}")
                        
                        # Probability breakdown
                        st.markdown("**Probability Breakdown:**")
                        st.write(f"• Low Risk: {probability[0]:.2%}")
                        st.write(f"• High Risk: {probability[1]:.2%}")
                        
                        st.markdown("---")
                        st.markdown("⚠️ **Medical Disclaimer:**")
                        st.caption("This prediction is for informational purposes only and should not replace professional medical diagnosis.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check that all fields are filled correctly.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        This machine learning model predicts the risk of kidney disease based on various medical parameters.
        
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: Multiple medical parameters
        - Purpose: Risk assessment tool
        
        **Key Indicators:**
        - Blood pressure
        - Serum creatinine levels
        - Blood glucose levels
        - Protein in urine
        - Medical history
        """)

if __name__ == "__main__":
    main()