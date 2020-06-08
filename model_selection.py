# Import des principales librairies.

import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression, ElasticNet, ElasticNetCV, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import re
import time
import skimage
from sklearn import decomposition

DIR = "./images_processed/"

# Fonction principale qui charge les données, testent une liste de modèle par le biais de pipeline et de GridsearchCV.
# Retourne un fichier .txt qui comporte le meilleur modèle.

def main():
    print("Nous chargeons nos données.")
    train_data, label_data = load_training_data()
    print("Nos données sont chargées.")
    print("Nous appliquons la PCA.")
    X_pca = apply_pca(train_data)
    print("La PCA est appliquée.")
    # Nous encodons les labels pour pouvoir appliquer les modèles.
    lb_make = LabelEncoder()
    label_data = lb_make.fit_transform(label_data)
    print("Nous commençons les essais sur les différents modèles.")
    test_model(X_pca, label_data)

#_________________________________________________________________

# Liste des modèles à tester

model_list = {
        
       'RF': { 'model':RandomForestRegressor(),
              'param':{
                'clf__n_estimators': [500, 1000, 1500],
                'clf__max_depth': [1,5,10,15,50,70],
#                 'clf__min_samples_split': [1,5,10,15]
#                   'cl_max_leaf_nodes': [ 100, 200, 300, 400, 500, 600, 650, 700, 800]
                  },
             },
        

#         'Lasso': { 'model': LassoCV(),
#                 'param': {'clf__alphas':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#              },
    
#         'Ridge': { 'model': RidgeCV(),
#                 'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#              },
    
#         'Elastic': { 'model': ElasticNet(),
#                 'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
#                          'clf__l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# #                         }
# #              },
    
#         'LR': { 'model': LogisticRegression(),
#                 'param': {'clf__penalty' : ['l1', 'l2'], 'clf__C' : np.logspace(-4, 4, 20), 'clf__solver' : ['liblinear']},
#              },
       
        'SVR':{ 'model': SVR(),
                'param': {'clf__C': [0.1,1, 10, 100], 'clf__gamma': [1,0.1,0.01,0.001],'clf__kernel': ['rbf', 'sigmoid'],
                         },
             },
    
        'SVC':{ 'model': SVC(),
                'param': {'clf__C': [0.1,1, 10, 100, 1000], 'clf__gamma': [1,0.1,0.01,0.001, 0.0001],'clf__kernel': ['rbf', 'poly', 'sigmoid'],
                         },
             },
      
#         'XGB':{ "model":XGBRegressor(),
#               "param":{"clf__learning_rate": [0.05,1,5],'clf__n_estimators': [100,50],
# #                        "clf__max_depth": [5,10,15]
#                   },
#             },
    
        'GradientBoost':{ "model":GradientBoostingRegressor(),
              "param":{"clf__n_estimators": [500,600,700,800,1000],
#                         "clf__max_depth": [2, 3, 4]
                  },
            },
    
        'decisionTree':{ "model":DecisionTreeClassifier(),
              "param":{"clf__criterion": ['gini'],
                'clf__min_samples_leaf': [5, 10, 15, 20, 25],
                'clf__max_depth': [6, 9, 12, 15, 20],
                  },
            },       
}

#_________________________________________________________________

# load_training_data : cette fonction permet de récupérer les images processés par le fichier "process_image.py".
# Ces images sous la forme d'array sont rajoutés à la liste train_data tandis que les labels correspondant sont rajoutés 
# dans la liste label_data.

def load_training_data():
    train_data = []
    label_data = []
    for img in os.listdir(DIR):
        label = str(img).split('.')[0]
        label = re.sub('\d', '', label)
        label = re.findall('[A-Z][^A-Z]*', label)
        label = " ".join(label)
        path = os.path.join(DIR, img)
        img = Image.open(path).convert('L')
        img = np.array(img)
        train_data.append(img)
        label_data.append(label)
    return train_data, label_data

#_________________________________________________________________

# apply_pca : cette fonction permet d'appliquer le traitement PCA à nos arrays.

def apply_pca(train_data):
    X = np.array(train_data).reshape((len(train_data),-1))
    # Nous utilisons le StandardScaler avant le traitement PCA.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    filename = "scaler.sav"
    pickle.dump(scaler, open(filename, 'wb'))
    # Nous cherchons une valeur de sommes de n_components pas trop élevées pour limiter le nombre de features. Ceci
    # a une incidence sur les performances de notre modèle.
    pca = decomposition.PCA(n_components = 0.6, whiten=True)
    X_pca = pca.fit_transform(X)
    filename = 'pca.sav'
    pickle.dump(pca, open(filename, 'wb'))
    print(X_pca.shape)
    return X_pca

#_________________________________________________________________

# apply_hog : cette fonction permet d'appliquer le traitement HOG à nos arrays. Deux classes sont créées pour faciliter la 
# transformation.

from sklearn.base import BaseEstimator, TransformerMixin
 
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
 
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])
        
def apply_hog(train_data):
    hogify = HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
    )
    scalify = StandardScaler()
    X_hog = hogify.fit_transform(train_data)
    X_hog = scalify.fit_transform(X_hog)
    print(X_hog.shape)
    return X_hog

#_________________________________________________________________

# test_model : cette fonction permet de tester tous les modèles de notre model_list sur les arrays traités par HOG ou PCA.

def test_model(X, label_data):
    X_train, X_test, y_train, y_test = train_test_split(X, label_data, random_state=42, test_size=0.2)
    list_model_accuracy = []
    list_model_estimator = []
    for model in model_list:
        print(model)
        clf = Pipeline(steps=[('clf', model_list[model]['model'])])
        param_grid = model_list[model]['param']
        start_time = time.time()

        grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=2, verbose=1)
        grid.fit(X_train,y_train)

        y_pred = grid.predict(X_test)
        # print(y_pred)
        accuracy= grid.score(X_test, y_test)
        print(("best model  : %.5f"
               % accuracy))
        # Affichage du temps d execution
        times = (time.time() - start_time)
        print("Temps d'execution : %s secondes ---" % times)
        # Rajout de l'accuracy à la liste 'list_model_accuracy'
        list_model_accuracy.append(accuracy)
        # Rajout du meilleur modèle à la liste 'liste_model_estimator'
        list_model_estimator.append(grid.best_estimator_[0])
    # Récupération de l'index de la meilleur accuracy.
    best_index = np.argmax(list_model_accuracy)
    best_accuracy = max(list_model_accuracy)
    print(list_model_accuracy)
    print(list_model_estimator)
    # Récupération du modèle correspondant à la meilleur accuracy.
    best_estimator = list_model_estimator[best_index]
    print("Le meilleur modèle est le suivant %s." % best_estimator)
    print("Avec l'accuracy suivante %s." % best_accuracy)
    # Fit du meilleur modèle sur le dataset complet et export grâce à la librairie Pickle
    best_model = best_estimator
    best_model.fit(X, label_data)
    filename = 'best_model.sav'
    pickle.dump(best_model, open(filename, 'wb'))
    
    # # Création d'un fichier 'best_estimator.txt' pour récupérer le meilleur modèle.
    # f = open('best_estimator.txt', 'w+' )
    # f.write(repr(best_estimator) )
    # f.close()

if __name__ == '__main__':
    main()
