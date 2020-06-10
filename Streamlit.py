import pandas as pd
import streamlit as st
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
import re
import time
import skimage
from sklearn import decomposition


character_list = ['Alex Louis Armstrong',
 'Alphonse Elric',
 'C.C.',
 'Edward Elric',
 'Euphemia Li Britannia',
 'Gluttony',
 'Izumi Curtis',
 'Jeremiah Gottwald',
 'Kururugi Suzaku',
 'Kôzuki Karen',
 'Lelouch Lamperouge',
 'Lust',
 'Milly Ashford',
 'Nunnally Lamperouge',
 'Riza Hawkeye',
 'Roy Mustang',
 'Scar',
 'Schneizel El Britannia',
 'Tôdô Kyoshirô',
 'Winry Rockbell']

character_list_2 = ['Edward Elric','Alphonse Elric','Gluttony','Roy Mustang','Lust','Winry Rockbell','Scar','Riza Hawkeye','Alex Louis Armstrong','Izumi Curtis','Lelouch Lamperouge','Kururugi Suzaku','C.C.','Kôzuki Karen','Nunnally Lamperouge','Euphemia Li Britannia','Schneizel El Britannia','Milly Ashford','Tôdô Kyoshirô','Jeremiah Gottwald']

IMG_SIZE = (60,60)

st.title("Upload + Classification de l'image")

# Import du modèle

filename = 'scaler.sav'
scaler = pickle.load(open(filename, 'rb'))

filename = 'best_model.sav'
model = pickle.load(open(filename, 'rb'))

filename = 'pca.sav'
pca = pickle.load(open(filename, 'rb'))

# resize_image : récupère une image et adapte sa taille à la taille cible tout en rajoutant un cadre blanc autour si nécessaire.

def resize_image(src_image, size=IMG_SIZE, bg_color="white"): 
    # Resize l'image au format cible tout en conservant les proportions.
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Créer un nouvelle image blanche au format cible.
    new_image = Image.new("L", size, bg_color)
    
    # Copie l'image resizé sur l'image blanche.
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    
    # Transforme la nouvelle image en greyscale.
    new_image = new_image.convert('L')
    
    # Retourne l'image resizée.
    return new_image

# Upload de la photographie

uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])
img_list = []
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploadée.', use_column_width=True)
    st.write("")
    st.write("Classification...")
    img = np.array(resize_image(image, size=IMG_SIZE, bg_color="white"))
    img_list.append(img)
    X = np.array(img_list).reshape((len(img_list),-1))
    X = scaler.transform(X)
    X_pca = pca.transform(X)
    y_pred = model.predict(X_pca)
    # st.write(model)
    # st.write(y_pred)
    result = int(y_pred[0])
    character = character_list[result]
    if character_list_2.index(character) < 10:
        anime = "Full Metal Alchemist Brotherhood"
    else:
        anime = "Code Geass"
    st.title("Il s'agit de %s !!" % character)
    st.title("Ce personnage appartient à l'anime %s !!" % anime)
