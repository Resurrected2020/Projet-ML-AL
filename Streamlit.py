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

character_list = ['Edward Elric','Alphonse Elric','Gluttony','Roy Mustang','Lust','Winry Rockbell','Scar','Riza Hawkeye','Alex Louis Armstrong','Izumi Curtis','Lelouch Lamperouge','Kururugi Suzaku','C.C.','Kôzuki Karen','Nunnally Lamperouge','Euphemia Li Britannia','Schneizel El Britannia','Milly Ashford','Tôdô Kyoshirô','Jeremiah Gottwald']

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

uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")
img_list = []
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploadée.', use_column_width=True)
    st.write("")
    st.write("Classification...")
    img = np.array(resize_image(image, size=IMG_SIZE, bg_color="white"))
    st.write(img)
    img_list.append(img)
    X = np.array(img_list).reshape((len(img_list),-1))
    X = scaler.transform(X)
    X_pca = pca.transform(X)
    st.write(X_pca)
    y_pred = model.predict(X_pca)
    result = int(np.floor(y_pred[0]))
    character = character_list[result]
    st.title("Il s'agit de %s !!" % character)
