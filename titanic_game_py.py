import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="🚢 Titanic Survival Game",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load your trained model and scalers
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('titanic_model.pkl')
        scaler_a = joblib.load('scaler.age')
        scaler_f = joblib.load('scaler.fare') 
        scaler_p = joblib.load('scaler.parch')
        scaler_s = joblib.load('scaler.sibsp')
        encoder = joblib.load('encoder_titanic.pkl')
        return model, scaler_a, scaler_f, scaler_p, scaler_s, encoder
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure all model files are in the same directory as this script.")
        return None, None, None, None, None, None

# Your actual prediction function
def predict_survival(input_data, model, scaler_a, scaler_f, scaler_p, scaler_s, encoder):
    """
    Your original prediction function adapted for Streamlit
    """
    if None in [model, scaler_a, scaler_f, scaler_p, scaler_s, encoder]:
        return "Model not loaded"
    
    try:
        df = pd.DataFrame([input_data])
        
        # Scale numeric columns
        df[['age']] = scaler_a.transform(df[['age']])
        df[['fare']] = scaler_f.transform(df[['fare']])
        df[['parch']] = scaler_p.transform(df[['parch']])
        df[['sibsp']] = scaler_s.transform(df[['sibsp']])
        
        # Encode categorical columns properly
        encoded_embarked = encoder.transform(df[['embarked']])
        encoded_embarked_df = pd.DataFrame(
            encoded_embarked, 
            columns=encoder.get_feature_names_out(['embarked'])
        )
        
        # Drop original 'embarked' and add encoded columns
        df = df.drop('embarked', axis=1)
        df = pd.concat([df.reset_index(drop=True), encoded_embarked_df.reset_index(drop=True)], axis=1)
        
        # Make sure columns are in the same order as training
        final_features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'family_size', 'is_alone'] + list(encoder.get_feature_names_out(['embarked']))
        
        # Predict
        

        prediction = model.predict(df[final_features])
        return "Survived" if prediction[0] == 1 else "Did not survive"
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error in prediction"

# Custom CSS for game-like styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P:wght@400&display=swap');
    
    /* Background and main styling */
    .stApp {
        background: linear-gradient(180deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        color: white;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .main-title {
        font-family: 'Press Start 2P', monospace;
        text-align: center;
        color: #81d4fa;
        font-size: 2rem;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: shipSway 6s ease-in-out infinite;
    }
    
    @keyframes shipSway {
        0%, 100% { transform: rotate(-1deg) translateY(0px); }
        25% { transform: rotate(0.5deg) translateY(-3px); }
        50% { transform: rotate(1deg) translateY(-5px); }
        75% { transform: rotate(-0.5deg) translateY(-2px); }
    }
    
    /* Captain styling */
    .captain {
        text-align: center;
        font-size: 4rem;
        margin: 20px 0;
        animation: captainBob 3s ease-in-out infinite;
    }
    
    @keyframes captainBob {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-8px) scale(1.02); }
    }
    
    /* Form container */
    .form-container {
        background: rgba(0, 0, 0, 0.3);
        border: 3px solid #4fc3f7;
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        backdrop-filter: blur(5px);
    }
    
    /* Captain quote */
    .captain-quote {
        font-family: 'Press Start 2P', monospace;
        text-align: center;
        color: #81d4fa;
        font-size: 0.8rem;
        margin: 20px 0;
        padding: 15px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        border: 2px solid #29b6f6;
    }
    
    /* Success result */
    .success-result {
        background: linear-gradient(135deg, #4caf50, #8bc34a);
        border: 3px solid #66bb6a;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        animation: survivedGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes survivedGlow {
        0% { 
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
            transform: scale(1);
        }
        100% { 
            box-shadow: 0 0 40px rgba(76, 175, 80, 0.8);
            transform: scale(1.02);
        }
    }
    
    /* Failure result */
    .failure-result {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        border: 3px solid #ef5350;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        animation: diedPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes diedPulse {
        0%, 100% { 
            box-shadow: 0 0 15px rgba(244, 67, 54, 0.3);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 30px rgba(244, 67, 54, 0.6);
            transform: scale(0.98);
        }
    }
    
    /* Result character */
    .result-character {
        font-size: 4rem;
        margin: 20px 0;
        animation: resultEntrance 1.5s ease-out;
    }
    
    @keyframes resultEntrance {
        0% { 
            transform: translateY(-100px) scale(0.5) rotate(180deg); 
            opacity: 0; 
        }
        60% { 
            transform: translateY(20px) scale(1.2) rotate(0deg); 
            opacity: 1; 
        }
        100% { 
            transform: translateY(0px) scale(1) rotate(0deg); 
            opacity: 1; 
        }
    }
    
    /* Streamlit input styling */
    .stSelectbox > div > div {
        background-color: rgba(13, 71, 161, 0.4);
        border: 2px solid #29b6f6;
        border-radius: 6px;
        color: white;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(13, 71, 161, 0.4);
        border: 2px solid #29b6f6;
        border-radius: 6px;
        color: white;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(13, 71, 161, 0.4);
        border: 2px solid #29b6f6;
        border-radius: 6px;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Press Start 2P', monospace;
        background: linear-gradient(45deg, #f44336, #e91e63);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-size: 12px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 6px 25px rgba(244, 67, 54, 0.5);
    }
    
    /* Loading animation */
    .loading-ship {
        font-size: 4rem;
        text-align: center;
        animation: loadingSail 2s ease-in-out infinite;
        margin: 20px 0;
    }
    
    @keyframes loadingSail {
        0%, 100% { transform: translateX(-20px) rotate(-5deg); }
        50% { transform: translateX(20px) rotate(5deg); }
    }
    
    .loading-text {
        font-family: 'Press Start 2P', monospace;
        font-size: 1rem;
        color: #81d4fa;
        text-align: center;
        animation: loadingPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* Stars background effect */
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, #fff, transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, #fff, transparent),
            radial-gradient(2px 2px at 160px 30px, #ddd, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: twinkle 4s ease-in-out infinite alternate;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.3; }
        100% { opacity: 0.8; }
    }
    
    </style>
    """, unsafe_allow_html=True)

def main():
    # Load CSS
    load_css()
    
    # Load model components
    model, scaler_a, scaler_f, scaler_p, scaler_s, encoder = load_model()
    
    # Add stars background
    st.markdown('<div class="stars"></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'form'
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'passenger_name' not in st.session_state:
        st.session_state.passenger_name = ""
    
    # Main title
    st.markdown('<h1 class="main-title">🚢 R.M.S. TITANIC 🚢</h1>', unsafe_allow_html=True)
    
    # Captain
    st.markdown('<div class="captain">👨‍✈️</div>', unsafe_allow_html=True)
    
    # Game states
    if st.session_state.game_state == 'form':
        show_passenger_form(model, scaler_a, scaler_f, scaler_p, scaler_s, encoder)
    elif st.session_state.game_state == 'loading':
        show_loading()
    elif st.session_state.game_state == 'result':
        show_result()

def show_passenger_form(model, scaler_a, scaler_f, scaler_p, scaler_s, encoder):
    # Captain's quote
    st.markdown('''
    <div class="captain-quote">
        "Welcome aboard! I need your details for the passenger manifest."
    </div>
    ''', unsafe_allow_html=True)
    
    # Form container
    with st.container():
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        with st.form("passenger_form", clear_on_submit=False):
            # Row 1: Name and Age
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("🎭 Passenger Name", placeholder="Enter your name")
            with col2:
                age = st.number_input("🎂 Age", min_value=0, max_value=120, step=1)
            
            # Row 2: Class and Gender
            col1, col2 = st.columns(2)
            with col1:
                pclass = st.selectbox("🎩 Passenger Class", 
                                    options=[None, 1, 2, 3],
                                    format_func=lambda x: "Select Class" if x is None else f"{x} Class")
            with col2:
                sex_str = st.selectbox("⚧️ Gender", 
                                 options=[None, "male", "female"],
                                 format_func=lambda x: "Select Gender" if x is None else x.title())
            if sex_str is not None:
                sex = 0 if sex_str == "male" else 1
            else:
                sex = None 
            # Row 3: Fare and Embarkation
            col1, col2 = st.columns(2)
            with col1:
                fare = st.number_input("💰 Fare (£)", min_value=0.0, step=0.01, format="%.2f")
            with col2:
                embarked = st.selectbox("🚢 Embarked From", 
                                      options=[None, "C", "Q", "S"],
                                      format_func=lambda x: "Select Port" if x is None else 
                                                {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
            
            # Row 4: Family details
            col1, col2 = st.columns(2)
            with col1:
                sibsp = st.number_input("👫 Siblings/Spouse Aboard", min_value=0, max_value=10, step=1)
            with col2:
                parch = st.number_input("👨‍👩‍👧‍👦 Parents/Children Aboard", min_value=0, max_value=10, step=1)
            
            # Row 5: Calculated family details
            col1, col2 = st.columns(2)
            with col1:
                family_size = st.number_input("👨‍👩‍👧‍👦 Total Family Size", 
                                            value=sibsp + parch + 1, 
                                            min_value=1, max_value=20, step=1)
            with col2:
                is_alone = st.selectbox("🚶 Traveling Alone?", 
                                      options=[None, "yes", "no"],
                                      format_func=lambda x: "Select" if x is None else x.title(),
                                      index=1 if family_size == 1 else 2 if family_size > 1 else 0)
            
            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚢 Board the Ship!", use_container_width=True)
            
            if submitted:
                # Validation
                passenger_data = {
                    'pclass': int(pclass),
                    'sex': sex,
                    'age': float(age),
                    'fare': float(fare),
                    'embarked': embarked,
                    'sibsp': int(sibsp),
                    'parch': int(parch),
                    'family_size': int(family_size),
                    'is_alone': 1 if is_alone == "yes" else 0
                }
                print (passenger_data)




                required_fields = {
                    "name": name,
                    "age": age,
                    "pclass": pclass,
                    "sex": sex,
                    "fare": fare,
                    "embarked": embarked,
                    "sibsp": sibsp,
                    "parch": parch,
                    "family_size": family_size,
                    "is_alone": is_alone
                }

                # Validation
                for field, value in required_fields.items():
                    if value is None or value == "":
                        st.error(f"⚠️ Please fill in the required field: {field}")
                        return






                # if not all([name, age is not None, pclass, sex, fare is not None, embarked, 
                #            sibsp is not None, parch is not None, family_size, is_alone]):
                #     st.error("⚠️ Please fill in all required fields!")
                #     return
                
                # Store data and change state
                passenger_data = {
                    'pclass': int(pclass),
                    'sex': sex,
                    'age': float(age),
                    'fare': float(fare),
                    'embarked': embarked,
                    'sibsp': int(sibsp),
                    'parch': int(parch),
                    'family_size': int(family_size),
                    'is_alone': 1 if is_alone == "yes" else 0
                }
                
                st.session_state.passenger_data = passenger_data
                st.session_state.passenger_name = name
                st.session_state.game_state = 'loading'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_loading():
    st.markdown('''
    <div style="text-align: center; margin: 100px 0;">
        <div class="loading-ship">🚢</div>
        <div class="loading-text">Determining your fate...</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Simulate loading time
    time.sleep(3)
    
    # Make prediction
    model, scaler_a, scaler_f, scaler_p, scaler_s, encoder = load_model()
    if model is not None:
        prediction = predict_survival(st.session_state.passenger_data, 
                                    model, scaler_a, scaler_f, scaler_p, scaler_s, encoder)
        st.session_state.prediction_result = prediction
    else:
        st.session_state.prediction_result = "Model not available"
    
    st.session_state.game_state = 'result'
    st.rerun()

def show_result():
    survived = st.session_state.prediction_result == "Survived"
    passenger_name = st.session_state.passenger_name
    
    if survived:
        st.markdown(f'''
        <div class="success-result">
            <div class="result-character">🎉</div>
            <h2 style="font-family: 'Press Start 2P', monospace; margin: 20px 0;">
                🎊 Congratulations {passenger_name}! 🎊
            </h2>
            <p style="font-size: 1.2rem; margin: 15px 0;">
                You survived the Titanic disaster!<br>
                🛟 Safe and Sound! 🛟<br><br>
                ✨ A lifeboat rescued you! ✨
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Add balloons effect
        st.balloons()
        
    else:
        st.markdown(f'''
        <div class="failure-result">
            <div class="result-character">🌊</div>
            <h2 style="font-family: 'Press Start 2P', monospace; margin: 20px 0;">
                💔 Oh no {passenger_name}... 💔
            </h2>
            <p style="font-size: 1.2rem; margin: 15px 0;">
                You didn't survive the disaster.<br>
                🌊 Lost at Sea 🌊<br><br>
                ⚓ May you rest in peace ⚓
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Add snow effect for somber mood
        st.snow()
    
    # Play Again button
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎮 Play Again", use_container_width=True):
            # Reset game state
            st.session_state.game_state = 'form'
            st.session_state.prediction_result = None
            st.session_state.passenger_name = ""
            st.rerun()

if __name__ == "__main__":
    main()