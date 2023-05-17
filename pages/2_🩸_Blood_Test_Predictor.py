import streamlit as st
import time
import numpy as np



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1


st.set_page_config(page_title="Arthritis Blood Report Analyzer", page_icon="🩸")

st.markdown("# Arthritis Blood Report Analyzer")
st.sidebar.header(" Arthritis Blood Report Analyzer ")
st.write("Blood report  values tracking the progression of arthritis and assessing the effectiveness of treatments, providing critical insights into disease management and therapeutic outcomes.")





import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Load the label encoder and scaler
label_encoder = LabelEncoder()
anti_ccp_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Load the dataset
dataset = pd.read_csv('arthritis_dataset.csv')

# Fit the label encoder on the 'Arthritis Progression' and 'Anti-CCP' columns
label_encoder.fit(dataset['Arthritis Progression'])
anti_ccp_encoder.fit(dataset['Anti-CCP'])

# Fit the scaler on the numerical columns only
scaler.fit(dataset[['CRP', 'ESR', 'RF']])

# Function to preprocess the input data
def preprocess_input(data):
    data_encoded = data.copy()
    data_encoded['Anti-CCP'] = anti_ccp_encoder.transform(data_encoded['Anti-CCP'])
    data_encoded[['CRP', 'ESR', 'RF']] = scaler.transform(data_encoded[['CRP', 'ESR', 'RF']])
    data_reshaped = data_encoded.values.reshape((data_encoded.shape[0], 1, data_encoded.shape[1]))
    return data_reshaped

# Function to make predictions
def make_prediction(data):
    data_preprocessed = preprocess_input(data)
    prediction_probs = model.predict(data_preprocessed)
    predicted_labels = np.argmax(prediction_probs, axis=1)
    return predicted_labels

# Streamlit app
def main():
    # st.title('Arthritis Blood Report Analyzer')

    # Input blood report values
    crp = st.number_input('CRP (C-reactive protein)')
    esr = st.number_input('ESR (Erythrocyte sedimentation rate)')
    rf = st.number_input('RF (Rheumatoid factor)')
    anti_ccp = st.selectbox('Anti-CCP antibodies', ['Negative', 'Positive'])

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'CRP': [crp], 'ESR': [esr], 'RF': [rf], 'Anti-CCP': [anti_ccp]})

    # Make prediction on button click
    if st.button('Predict'):
        prediction = make_prediction(input_data)
        class_labels = ['Moderate Progression', 'Mild Progression', 'No Progression', 'Severe Progression']
        predicted_label = class_labels[prediction[0]]
        st.write(f'Predicted Arthritis Progression: {predicted_label}')

if __name__ == '__main__':
    main()

