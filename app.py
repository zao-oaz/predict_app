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
#MODEL_DIR = os.path.join("C:/Users/zaome/Documents/Arthuro/Projet_reseau_neuronal", 'test_model.h5')

MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'test_model.h5')
model = load_model("test_model.h5")

# ______________________________________________ #

# Import Dataset test
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_test = pd.read_csv(uploaded_file)
  st.write(df_test.head())

# ______________________________________________ #
# Pages
# page = st.selectbox("Choose your page", ["Version 1", "Version 2"])
# if page == "Version 1":
#   print("tetst")
# if page == "Version 2":
#   print("test")
# ______________________________________________ #

st.title("Version 1")

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

# _____________________________________________ #

st.title("Version 2")

# Preprocess
if df_test is not None:
    df_test = df_test/255 
    df_test = np.array(df_test)
    df_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
    prediction = model.predict(df_test)
    prediction = np.argmax(prediction, axis=1)

# Prédiction image
def test_prediction(index):
    # Barre
    number = st.slider("Pick a number", 0, 9)
    st.write('Number :', number)
    print('Predicted category :', prediction[index])
    img = df_test[index].reshape(28,28)
    st.image(img, width=140)
    plt.imshow(img, cmap='gray')

index = np.random.choice(df_test.shape[0])
test_prediction(index)

# _____________________________________________ #

# st.title("Version 3")

# SIZE = 192
# # Premier modele  
# mode = st.checkbox("Dessinez votre chiffre", True)
# canvas_result = st_canvas(
#     fill_color='#000000',
#     stroke_width=15,
#     stroke_color='blue',
#     background_color='#FFFFFF',
#     width=SIZE, 
#     height=SIZE,
#     drawing_mode="freedraw" if mode else "transform",
#     key='canvas')

# # Entrainement du modele
# if canvas_result.image_data is not None:
#     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#     rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
#     st.write('Prédiction du modèle')
#     st.image(rescaled)

# # Prediction + graph
# if st.button('Prédiction'):
#     test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     val = model.predict(test.reshape(1,28,28,1))
#     st.write(f'Prédiction : {np.argmax(val[0])}')
#     #st.bar_chart(val[0])