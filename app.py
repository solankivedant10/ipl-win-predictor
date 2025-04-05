import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="IPL Win Predictor", layout="centered")
st.title("🏏 IPL Win Predictor")

# Check current directory and files
st.write("🔍 Checking environment...")
st.code(f"Current Directory: {os.getcwd()}")
st.code(f"Files Available: {os.listdir()}")

# Load pipeline with error handling
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Error: 'pipe.pkl' not found. Please ensure the file is in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Team and city inputs
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Rajasthan Royals',
    'Delhi Capitals', 'Punjab Kings'
]
cities = ['Mumbai', 'Chennai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad']

# Input fields with validation
batting_team = st.selectbox('Select the Batting Team', teams)
bowling_team = st.selectbox('Select the Bowling Team', teams)
if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams cannot be the same!")

city = st.selectbox('Match City', cities)

target = st.number_input('🎯 Target Score', min_value=1.0, step=1.0)
score = st.number_input('🔢 Current Score', min_value=0.0, max_value=target, step=1.0)
balls_left = st.number_input('🏏 Balls Left', min_value=0.0, max_value=120.0, step=1.0)
wickets_left = st.slider('🧢 Wickets Left', 0, 10, 5)

# CRR and RRR Calculation
balls_bowled = 120 - balls_left
crr = score / (balls_bowled / 6) if balls_bowled > 0 else 0
rrr = ((target - score) / balls_left) * 6 if balls_left > 0 else float('inf') if target > score else 0

# Show metrics
st.markdown(f"📊 **Current Run Rate (CRR):** `{crr:.2f}`")
st.markdown(f"📈 **Required Run Rate (RRR):** `{rrr:.2f}`" if rrr != float('inf') else "📈 **RRR:** Impossible")

# Predict button logic
if st.button('🔮 Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("❗ Please select different teams for batting and bowling.")
    elif target <= score and balls_left > 0:
        st.success(f"✅ {batting_team} has already won!")
    elif balls_left == 0 and target > score:
        st.success(f"🏆 {bowling_team} has won!")
    else:
        runs_left = target - score
        target_to_chase = target

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'target': [target],
            'current_score': [score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'rrr': [rrr],
            'runs_left': [runs_left],
            'target_to_chase': [target_to_chase],
        })

        try:
            result = pipe.predict_proba(input_df)
            win_prob = result[0][1] * 100
            loss_prob = result[0][0] * 100

            st.subheader(f"🎉 {batting_team} Win Probability: `{win_prob:.2f}%`")
            st.subheader(f"💔 {bowling_team} Win Probability: `{loss_prob:.2f}%`")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
