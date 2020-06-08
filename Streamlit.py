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


# def preprocessing():
    
#     # on définit les colonnes et les transformations pour 
#     # les colonnes quantitatives

#     col_quanti = ['Age', 'Fare' ]

#     transfo_quanti = Pipeline(steps=[
#         ('imputation', SimpleImputer(strategy='median')),
#         ('standard', StandardScaler())])

#     # on définit les colonnes et les transformations pour
#     # les variables qualitatives

#     col_quali = ['Pclass', 'Sex', 'Embarked',"SibSp","Parch"]

#     transfo_quali = Pipeline(steps=[
#         ('imputation', SimpleImputer(strategy='constant', fill_value='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#     # on définit l'objet de la classe ColumnTransformer
#     # qui va permettre d'appliquer toutes les étapes
#     preparation = ColumnTransformer(
#         transformers=[
#             ('quanti', transfo_quanti , col_quanti),
#             ('quali', transfo_quali , col_quali)],
#         remainder = 'drop' )

    
#     return preparation 

# def input_data():

#     Pclass = st.sidebar.selectbox("Class", df['Pclass'].unique())
#     Sex = st.sidebar.radio("Sexe", df['Sex'].unique())
#     Age = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), 20)
#     SibSp = st.sidebar.selectbox("SibSp", df['SibSp'].unique())
#     Parch = st.sidebar.selectbox("Parch", df['Parch'].unique())
#     Fare = st.sidebar.slider("Fare", int(df.Fare.min()), int(df.Fare.max()), 50)
#     Embarked = st.sidebar.selectbox("Embarked", df['Embarked'].unique())

#     input_user = np.array([Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]).reshape(1,-1)
#     df_user = pd.DataFrame(input_user, columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']) 

#     #preprocessing
#     # preparation = preprocessing()
#     # user_data = preparation.fit_transform(df_user)

#     return df_user


# def modele():

#     preparation = preprocessing()
#     df_user=input_data()

#     # on crée un pipeline de traitement intégrant la préparation
#     pipeline_ml = Pipeline(steps=[('preparation', preparation),
#                         ('logit', LogisticRegression(solver='lbfgs'))])

#     # on sépare la cible du reste des données
#     x = df.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
#     y = df['Survived']

#     # on construit les échantillons d'apprentissage et de validation
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,, random_state=0)

#     # on ajuste le modèle en utilisant les données d'apprentissage
#     # le modèle comporte la préparation et le modèle logistique
#     pipeline_ml.fit(x, y)

#     user_prediction = pipeline_ml.predict_proba(df_user)[:,1]*100

#     return user_prediction 

# ###
# #title
# st.title('Titanic dataset')
# st.markdown("Faites varier les features ")

# # Permet de garder resultat fonction en cache
# @st.cache
# #load data
# def get_data():
    
#     return pd.read_csv("train.csv")

# df = get_data()


# # Apercu de notre dataset
# if st.checkbox('Show data'):
#     st.subheader('Raw data')
#     st.write(df)

# # Prediction
# user_prediction = modele()
# if st.checkbox('Prediction'):
#     st.subheader('Probabilité de survivre')
#     st.write('{} %'.format(user_prediction))
