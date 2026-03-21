import os
import streamlit as st
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Multi-Disease Prediction System",
                   layout="wide",
                   page_icon="🩺")

# Get absolute paths to load models correctly regardless of where script is run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_models():
    """Loads all models and scalers into memory once."""
    try:
        diabetes_model = joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
        diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'))
        
        heart_model = joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl'))
        heart_scaler = joblib.load(os.path.join(MODELS_DIR, 'heart_scaler.pkl'))
        
        parkinsons_model = joblib.load(os.path.join(MODELS_DIR, 'parkinsons_model.pkl'))
        parkinsons_scaler = joblib.load(os.path.join(MODELS_DIR, 'parkinsons_scaler.pkl'))
        
        return {
            'diabetes': (diabetes_model, diabetes_scaler),
            'heart': (heart_model, heart_scaler),
            'parkinsons': (parkinsons_model, parkinsons_scaler)
        }
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have run 'python src/train.py' first.")
        return None

models = load_models()

def predict(model, scaler, input_data):
    """Formats inputs, scales them, and returns a prediction."""
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    return prediction[0]

# --- Sidebar for Navigation ---
with st.sidebar:
    st.title('🩺 Disease Prediction')
    selected = st.selectbox('Select a Disease Module',
                            ['Diabetes Prediction',
                             'Heart Disease Prediction',
                             "Parkinson's Prediction"])

if models is None:
    st.stop()

# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value='0')
    with col2:
        Glucose = st.text_input('Glucose Level', value='100')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', value='70')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', value='20')
    with col2:
        Insulin = st.text_input('Insulin Level', value='79')
    with col3:
        BMI = st.text_input('BMI value', value='25.0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value='0.5')
    with col2:
        Age = st.text_input('Age of the Person', value='30')
        
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            model, scaler = models['diabetes']
            if predict(model, scaler, user_input) == 1:
                diab_diagnosis = 'The person is highly likely to be Diabetic'
                st.error(diab_diagnosis)
            else:
                diab_diagnosis = 'The person is likely Non-Diabetic'
                st.success(diab_diagnosis)
        except ValueError:
            st.warning("Please enter valid numerical values.")

# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age', value='50')
    with col2:
        Sex = st.selectbox('Sex', options=['M', 'F'])
    with col3:
        ChestPainType = st.selectbox('Chest Pain Type', options=['ATA', 'NAP', 'ASY', 'TA'])
    with col1:
        RestingBP = st.text_input('Resting Blood Pressure', value='120')
    with col2:
        Cholesterol = st.text_input('Serum Cholestoral in mg/dl', value='200')
    with col3:
        FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    with col1:
        RestingECG = st.selectbox('Resting Electrocardiographic Results', options=['Normal', 'ST', 'LVH'])
    with col2:
        MaxHR = st.text_input('Maximum Heart Rate Achieved', value='150')
    with col3:
        ExerciseAngina = st.selectbox('Exercise Induced Angina', options=['N', 'Y'])
    with col1:
        Oldpeak = st.text_input('ST depression (Oldpeak)', value='0.0')
    with col2:
        ST_Slope = st.selectbox('Slope of the peak exercise ST segment', options=['Up', 'Flat', 'Down'])
        
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        try:
            sex_map = {'M': 1, 'F': 0}
            cp_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
            restecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
            exang_map = {'N': 0, 'Y': 1}
            slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
            
            user_input = [
                float(Age), sex_map[Sex], cp_map[ChestPainType], float(RestingBP), float(Cholesterol), 
                int(FastingBS), restecg_map[RestingECG], float(MaxHR), exang_map[ExerciseAngina], 
                float(Oldpeak), slope_map[ST_Slope]
            ]
            
            model, scaler = models['heart']
            if predict(model, scaler, user_input) == 1:
                heart_diagnosis = 'The person is likely to have Heart Disease'
                st.error(heart_diagnosis)
            else:
                heart_diagnosis = 'The person is likely to NOT have Heart Disease'
                st.success(heart_diagnosis)
        except ValueError:
            st.warning("Please enter valid numerical values.")

# --- Parkinsons Prediction Page ---
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    
    st.markdown("Please enter the voice measurement features below:")
    
    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    user_inputs = []
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            # Setting default value using typical values slightly above 0 to make it easier to test
            val = st.text_input(feature, value='0.005')
            user_inputs.append(val)
            
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        try:
            user_input_floats = [float(x) for x in user_inputs]
            
            model, scaler = models['parkinsons']
            if predict(model, scaler, user_input_floats) == 1:
                parkinsons_diagnosis = "The person has Parkinson's Disease indicators"
                st.error(parkinsons_diagnosis)
            else:
                parkinsons_diagnosis = "The person does not show Parkinson's Disease indicators"
                st.success(parkinsons_diagnosis)
        except ValueError:
            st.warning("Please enter valid numerical values.")
