# Data manip
import streamlit as st
#from streamlit_drawable_canvas import st_canvas

# Import librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Sreamlit via ligne de commande
#streamlit run "C:\Users\zaome\Documents\Arthuro\Projet_reseau_neuronal\app.py"

st.title("📉 Prédiction d'images")

# Chemin du modele 
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'test_model.h5')
model = load_model("test_model.h5")

# Import Dataset test
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_test = pd.read_csv(uploaded_file)
  st.write(df_test.head())

# Preprocess
if df_test is not None:
    df_test = df_test/255 
    df_test = np.array(df_test)
    df_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
    prediction = model.predict(df_test)
    prediction = np.argmax(prediction, axis=1)


# Prédiction image 
def test_prediction(index):  
    st.button('Random')
    #print('Predicted category :', prediction[index])
    img = df_test[index].reshape(28,28)
    st.image(img, width=140)
    plt.imshow(img, cmap='gray')
    st.write(f'Prediction : {(prediction[index])}')

index = np.random.choice(df_test.shape[0])
test_prediction(index)