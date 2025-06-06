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

# Load the model and feature names
@st.cache_resource
def load_model():
    try:
        # Try different possible paths
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
        # Try to load feature names from file first
        feature_file_paths = [
            'feature_names.pkl',
            'feature_names.txt',
            'features.pkl',
            'features.txt'
        ]
        
        for path in feature_file_paths:
            if os.path.exists(path):
                if path.endswith('.pkl'):
                    feature_names = joblib.load(path)
                    st.info(f"Feature names loaded from: {path}")
                    return feature_names
                elif path.endswith('.txt'):
                    with open(path, 'r') as f:
                        feature_names = [line.strip() for line in f.readlines()]
                    st.info(f"Feature names loaded from: {path}")
                    return feature_names
        
        # Fallback to hardcoded feature names if file not found
        st.warning("Feature names file not found. Using default feature names.")
        feature_names = [
            'Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine', 
            'Albumin in urine', 'Sugar in urine', 'Red blood cells in urine', 
            'Pus cells in urine', 'Pus cell clumps in urine', 'Bacteria in urine',
            'Random blood glucose level (mg/dl)', 'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)',
            'Sodium level (mEq/L)', 'Potassium level (mEq/L)', 'Hemoglobin level (gms)', 
            'Packed cell volume (%)', 'White blood cell count (cells/cumm)', 
            'Red blood cell count (millions/cumm)', 'Hypertension (yes/no)', 
            'Diabetes mellitus (yes/no)', 'Coronary artery disease (yes/no)', 
            'Appetite (good/poor)', 'Pedal edema (yes/no)', 'Anemia (yes/no)', 
            'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio', 
            'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level', 
            'Parathyroid hormone (PTH) level', 'Serum calcium level', 'Serum phosphate level', 
            'Family history of chronic kidney disease', 'Smoking status', 'Body Mass Index (BMI)', 
            'Physical activity level', 'Duration of diabetes mellitus (years)', 
            'Duration of hypertension (years)', 'Cystatin C level', 
            'Urinary sediment microscopy results', 'C-reactive protein (CRP) level', 
            'Interleukin-6 (IL-6) level'
        ]
        return feature_names
    except Exception as e:
        st.error(f"Error loading feature names: {str(e)}")
        return []

# Main app
def main():
    st.title("🩺 Kidney Disease Prediction System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    feature_names = load_feature_names()
    
    if model is None:
        st.stop()
    
    # Debug information
    st.write(f"Model loaded. Expected features: {len(feature_names)}")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Patient Information")
        
        # Create input fields
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
                # Prepare input data
                try:
                    # Convert categorical variables to numerical
                    rbc_encoded = 1 if rbc == "abnormal" else 0
                    pus_cell_encoded = 1 if pus_cell == "abnormal" else 0
                    bacteria_encoded = 1 if bacteria == "present" else 0
                    hypertension_encoded = 1 if hypertension == "yes" else 0
                    diabetes_encoded = 1 if diabetes == "yes" else 0
                    cad_encoded = 1 if cad == "yes" else 0
                    appetite_encoded = 1 if appetite == "poor" else 0
                    
                    # Create input array with ALL 42 features (fill missing ones with default values)
                    input_data = np.array([[
                        age,                    # Age of the patient
                        bp,                     # Blood pressure (mm/Hg)
                        specific_gravity,       # Specific gravity of urine
                        albumin,               # Albumin in urine
                        sugar,                 # Sugar in urine
                        rbc_encoded,           # Red blood cells in urine
                        pus_cell_encoded,      # Pus cells in urine
                        0,                     # Pus cell clumps in urine (default)
                        bacteria_encoded,      # Bacteria in urine
                        bg_random,             # Random blood glucose level (mg/dl)
                        blood_urea,            # Blood urea (mg/dl)
                        serum_creatinine,      # Serum creatinine (mg/dl)
                        sodium_level,          # Sodium level (mEq/L)
                        potassium_level,       # Potassium level (mEq/L)
                        hemoglobin,            # Hemoglobin level (gms)
                        pcv,                   # Packed cell volume (%)
                        wbc_count,             # White blood cell count (cells/cumm)
                        5.0,                   # Red blood cell count (millions/cumm) - default
                        hypertension_encoded,  # Hypertension (yes/no)
                        diabetes_encoded,      # Diabetes mellitus (yes/no)
                        cad_encoded,           # Coronary artery disease (yes/no)
                        appetite_encoded,      # Appetite (good/poor)
                        0,                     # Pedal edema (yes/no) - default
                        0,                     # Anemia (yes/no) - default
                        90,                    # Estimated Glomerular Filtration Rate (eGFR) - default
                        0.2,                   # Urine protein-to-creatinine ratio - default
                        2000,                  # Urine output (ml/day) - default
                        4.0,                   # Serum albumin level - default
                        200,                   # Cholesterol level - default
                        50,                    # Parathyroid hormone (PTH) level - default
                        10,                    # Serum calcium level - default
                        4,                     # Serum phosphate level - default
                        0,                     # Family history of chronic kidney disease - default
                        0,                     # Smoking status - default
                        25,                    # Body Mass Index (BMI) - default
                        1,                     # Physical activity level - default
                        0 if diabetes == "no" else 5,  # Duration of diabetes mellitus (years)
                        0 if hypertension == "no" else 5,  # Duration of hypertension (years)
                        1.2,                   # Cystatin C level - default
                        0,                     # Urinary sediment microscopy results - default
                        3,                     # C-reactive protein (CRP) level - default
                        5                      # Interleukin-6 (IL-6) level - default
                    ]])
                    
                    st.write(f"Input shape: {input_data.shape}")  # Debug info
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    
                    # Display results in the second column
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
                        
                        # Additional information
                        st.markdown("---")
                        st.markdown("⚠️ **Medical Disclaimer:**")
                        st.caption("This prediction is for informational purposes only and should not replace professional medical diagnosis. Please consult with a healthcare provider for proper medical evaluation.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check that all fields are filled correctly.")
                    # Debug information
                    st.write(f"Error details: {e}")
                    st.write(f"Model type: {type(model)}")
                    try:
                        if hasattr(model, 'n_features_in_'):
                            st.write(f"Model expects {model.n_features_in_} features")
                    except:
                        pass
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        This machine learning model predicts the risk of kidney disease based on various medical parameters.
        
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: 42 medical parameters
        - Accuracy: ~95%+ (based on your results)
        
        **Key Indicators:**
        - Blood pressure
        - Serum creatinine levels
        - Blood glucose levels
        - Protein in urine
        - Medical history
        """)
        
        st.markdown("---")
        st.markdown("**🔬 How to Use:**")
        st.markdown("""
        1. Fill in the patient information
        2. Click 'Predict Kidney Disease'
        3. Review the results
        4. Consult with healthcare providers
        """)

if __name__ == "__main__":
    main()