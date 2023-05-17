import streamlit as st
import time
import numpy as np

import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import streamlit as st
from PIL import Image


st.set_page_config(page_title=" X-Ray Arthritis Detector", page_icon="📈")

st.markdown("# X-Ray Analyzer")
st.sidebar.header(" X-Ray Analyzer ")
st.write("This app allows you to input X-Ray images and receive automated analysis for effective diagnosis and management.")


# Preprocessing
# 1. Resizing
# 2. Flatten

target = []
images = []
flat_data = []

DATADIR = 'images'
CATEGORIES = ['healthy', 'moderate', 'severe']

for category in CATEGORIES:
    class_num = CATEGORIES.index(category)  # label encoding
    path = os.path.join(DATADIR, category)  # create path to use all images
    for img in os.listdir(path):
        try:
            _, ext = os.path.splitext(img)
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue  # skip non-image files
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))  # Normalizes to 0 to 1
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(class_num)
        except:
            print(f"Unable to read image {img}")

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=109, stratify=target)

# KNN Algorithm
# Define the parameter grid
param_grid = {
    'n_neighbors': [1, 5, 10, 15, 20],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
}

# Create the KNN model
knn = KNeighborsClassifier()

# Create a StratifiedKFold cross-validator
cv = StratifiedKFold(n_splits=5)

# Set up the GridSearchCV with the KNN model, parameter grid, and stratified cross-validator
# clf = GridSearchCV(knn, param_grid, cv=cv)

# Fit the model to the training data
# clf.fit(X_train, y_train)

# Save the model using pickle library
# pickle.dump(clf, open('img_model.p', 'wb'))

model = pickle.load(open('img_model.p', 'rb'))

def get_color(arthritis_level):
    if arthritis_level == 'SEVERE':
        return 'red'
    elif arthritis_level == 'MODERATE':
        return 'orange'
    else:
        return 'green'

# st.title('X-Ray Arthritis Detector')
# st.text('Upload the X-ray Image (JPEG or PNG)')

uploaded_file = st.file_uploader("Please upload an X-ray image in JPEG or PNG format....", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')

    if st.button('PREDICT'):
        CATEGORIES = ['healthy', 'moderate', 'severe']
        ARTHRITIS_TYPE = "Osteoarthritis"
        st.write('Result...')
        flat_data = []
        img = np.array(img)
        img_resized = resize(img, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)

        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]].upper()

        color = get_color(y_out)
        st.markdown(f'<h2>Arthritis Type: {ARTHRITIS_TYPE}</h2>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:{color};">PREDICTED OUTPUT: {y_out}</h1>', unsafe_allow_html=True)

        q = model.predict_proba(flat_data)
        for index, item in enumerate(CATEGORIES):
            st.write(f'{item.upper()}: {q[0][index] * 100:.2f}%')
else:
    st.write("")