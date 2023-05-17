import streamlit as st
import time
import numpy as np

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title=" Arthritis Treatment Recommendation", page_icon="📈")

st.markdown("# Arthritis Treatment Recommendation")
st.sidebar.header(" Arthritis Treatment Recommendation ")
st.write("AyuAira : Personalized Ayurvedic Treatment Recommendations for Your Well-being.")


# Load the dataset
data = pd.read_csv('arthritis_data.csv')

# Set 'Patient ID' as index
data.set_index('Patient ID', inplace=True)

# Preprocess the data
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

le_overall_health = LabelEncoder()
data['Overall Health'] = le_overall_health.fit_transform(data['Overall Health'])

# Split the data into training and testing sets
X = data.drop(['Arthritis Type', 'Treatment Recommendation', 'Symptoms', 'Treatment Progress'], axis=1)
y = data['Treatment Recommendation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
clf = joblib.load('decision_tree_model.joblib')

# Make predictions for the testing set
y_pred = clf.predict(X_test)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)


# Streamlit app
# st.title("Arthritis Treatment Recommendation")

# Get user input
patient_id = st.number_input('Enter Patient ID', value=1, step=1)
age = st.number_input('Age')
sex = st.selectbox('Sex', ['Male', 'Female'])
overall_health = st.selectbox('Overall Health', ['Good', 'Fair', 'Poor'])
blood_test_result = st.number_input('Blood Test Result')
xray_image_result = st.number_input('X-ray Image Result')
symptoms = st.multiselect('Symptoms', ['pain', 'stiffness', 'fatigue', 'swelling'])
treatment_progress = st.radio('Treatment Progress', ['No improvement', '25% reduction in pain', '50% reduction in pain', '60% reduction in pain', '75% reduction in pain', '80% reduction in pain'])

# Display medical history when Patient ID is entered
if patient_id:
    medical_history_columns = ['Family_History_Hypertension', 'Family_History_Diabetes', 'Past_Surgeries', 'Smoking', 'Alcohol', 'Exercise', 'Current_Medications', 'Immunization_Flu']
    medical_history = data.loc[patient_id, medical_history_columns]
    st.sidebar.write("Medical History:")
    st.sidebar.write(medical_history)
    
    # Display Arthritis type when Patient ID is entered
    arthritis_type = data.loc[patient_id, 'Arthritis Type']
    arthritis_colors = {'Osteoarthritis': 'Blue', 'Rheumatoid arthritis': 'Red', 'Psoriatic Arthritis': 'Green', 'Gout': 'Purple'}
    st.markdown(f'<p style="font-size:25px;color:{arthritis_colors[arthritis_type]};">Arthritis Type: {arthritis_type}</p>', unsafe_allow_html=True)

if st.button("Get Treatment Recommendation"):
    # Preprocess user input
    user_data = pd.DataFrame({
        'Age': [age], 
        'Sex': [le_sex.transform([sex])[0]], 
        'Overall Health': [le_overall_health.transform([overall_health])[0]], 
        'Blood Test Result': [blood_test_result],
        'X-ray Image Result': [xray_image_result]
    })
        
    # Add medical history to user input
    for column in medical_history_columns:
        user_data[column] = medical_history[column]
        
    # Predict treatment recommendation
    user_data = user_data.reindex(columns=X.columns, fill_value=0)
    user_treatment_recommendation = clf.predict(user_data)
        
    # Color coding for different treatment recommendations
    treatment_colors = {
        'Massage therapy, Herbal remedies (turmeric, ginger, and Boswellia serrata), Panchakarma, Yoga and meditation': 'Blue', 
        'Apply Madhuca longifolia, Doing SwedanaKarma, Decoction, Eranda7, Rasna7': 'Red', 
        'Sesame oil, Doing SwedanaKarma, Apply paththu (chronic condition, Dashanga Lepaya), Diyalagapaththuwa': 'Green', 
        'Panchakarma, Massage therapy, Herbal remedies (Turmeric, Ginger, Boswellia Serrata), Follow an anti-inflammatory diet, Engage in regular low-impact exercise, Maintain a healthy weight': 'Purple'
    }
        
    try:
        treatment_color = treatment_colors[user_treatment_recommendation[0]]
    except KeyError:
        treatment_color = 'white'
        
    # Display treatment recommendation with color
    st.markdown(f'<p style="font-size:25px;color:{treatment_color};">Recommended treatment: {user_treatment_recommendation[0]}</p>', unsafe_allow_html=True)
