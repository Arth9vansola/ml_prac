import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_heart.pkl")
scaler = joblib.load("knn_scaler.pkl")
expected_columns = joblib.load("knn_columns.pkl")

st.title("Heart disease prediction by Arth 💗")
st.markdown("Provide the following details")

age = st.slider('Age',18,100,40)
sex = st.selectbox('SEX',['M','F'])
chest_pain = st.selectbox("Chest pain type", ['ATA','NAP','TA','ASY'])
resting_bp = st.number_input('Resting blood pressure(mm Hg)', 80, 200, 120)
cholestrol = st.number_input('Cholestrol(mg/dL)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox('Exercise-Induced Angina',['Y','N'])
oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST slope', ['Up', 'Flat', 'Down'])

if st.button("Predict"):
  raw_input = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholestrol': cholestrol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_' + sex: 1,
    'ChestPainType_' + chest_pain: 1,
    'RestingECG_' + resting_ecg: 1,
    'ExerciseAngina_' + exercise_angina: 1,
    'ST_slope_'+ st_slope: 1
  } 

  input_df = pd.DataFrame([raw_input])

  for col in expected_columns:
    if col not in input_df.columns:
      input_df[col] = 0

  input_df = input_df[expected_columns]    

  scaled_input = scaler.transform(input_df)

  prediction = model.predict(scaled_input)[0]

  if prediction == 1:
    st.error('High risk of heart disease')
  else:
    st.success('Low risk of heart disease')  
