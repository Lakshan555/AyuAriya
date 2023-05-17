import streamlit as st
import time
import numpy as np

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import streamlit as st



st.set_page_config(page_title="Continuous Monitoring And Feedback", page_icon="📈")

st.markdown("# Continuous Monitoring And Feedback")
st.sidebar.header(" Continuous Monitoring And Feedback ")
st.write("This app allows you to input data about arthritis patients and receive feedback on their treatment progress.")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

start_button = st.sidebar.button("Make Prediction")

if start_button:
    for i in range(1, 101):
        status_text.text("%i%% Complete" % i)
        progress_bar.progress(i)
        time.sleep(0.05)

        if i == 100:
            break

    progress_bar.empty()


# Load the dataset
data = pd.read_csv("patient_data.csv")

# Preprocess the data
le = LabelEncoder()
data["Arthritis Type"] = le.fit_transform(data["Arthritis Type"])
X = data.drop(["Arthritis Type"], axis=1)
y = data["Arthritis Type"]
X = pd.get_dummies(X)


# Load the saved model
model = joblib.load('feedback_model.joblib')
# Define a function to generate the classification report
def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

# Define the streamlit app
def app():
    # st.title("Arthritis Feedback Management System")
    # st.write("This app allows you to input data about arthritis patients and receive feedback on their treatment progress.")
    
    # Create input fields for patient data
    st.header("Input Patient Data")
    patient_id = st.number_input("Patient ID", min_value=1)
    age = st.slider("Age", min_value=1, max_value=100)
    sex = st.selectbox("Sex", options=["M", "F"])
    symptoms = st.text_input("Symptoms")
    treatment_progress = st.text_input("Treatment Progress")
    overall_health = st.selectbox("Overall Health", options=["Poor", "Fair", "Good"])
    
    # Create a button to make the prediction
    if start_button:
        
        # Encode the input data and make a prediction
        input_data = pd.DataFrame({"Patient ID": [patient_id], "Age": [age], "Sex": [sex], "Symptoms": [symptoms], "Treatment Progress": [treatment_progress], "Overall Health": [overall_health]})
        input_data["Arthritis Type"] = le.transform(["Rheumatoid Arthritis"])[0]
        input_data = input_data.drop(["Arthritis Type"], axis=1)
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Check if the shape of input_data matches the shape of X
        if input_data.shape[1] != X.shape[1]:
            st.error(f"Invalid input: expected {X.shape[1]} features, but got {input_data.shape[1]}")
            return

        prediction = model.predict(input_data)
        arthritis_type = le.inverse_transform(prediction)[0]

        # Output the prediction
        st.header("Prediction")
        st.write("The predicted arthritis type for this patient is:", arthritis_type)

        # Check if patient situation is critical
        overall_health_val = 0


        if overall_health == "Fair":
            overall_health_val = 1
        elif overall_health == "Good":
            overall_health_val = 2

        if arthritis_type == "Rheumatoid Arthritis":
            if overall_health_val == 0:
                st.markdown("<h3 style='color:red'>Patient situation is critical! Please consider changing the treatment plan.</h3>", unsafe_allow_html=True)
            elif overall_health_val == 1:
                st.markdown("<h3 style='color:red'>Patient situation is critical. Consider increasing the treatment plan.</h3>", unsafe_allow_html=True)
            else:
                st.success("Patient situation is stable.")
                st.write("The predicted arthritis type for this patient is:", arthritis_type)
        else:
            if overall_health_val == 0:
                st.markdown("<h3 style='color:red'>Patient situation is critical. Treatment plan needs to be changed!</h3>", unsafe_allow_html=True)
            elif overall_health_val == 1:
                st.markdown("<h3 style='color:orange'>Patient situation is moderate. Current treatment plan needs to be continued.</h3>", unsafe_allow_html=True)
            else:
                st.success("Patient situation is stable.")
                st.write("The predicted arthritis type for this patient is:", arthritis_type)

if __name__ == '__main__':
    app()
