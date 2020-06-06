# Ce fichier a pour objectif de transformer les images récupérées suite à un scrapping.

from PIL import Image, ImageOps # used for loading images
import numpy as np
import os, shutil # used for navigating to image path
import imageio # used for writing images
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import re
from PIL import Image, ImageOps 

# Définition de la taille cible des images après transformation.

IMG_SIZE = (60,60)
DIR = "./images/"

#_________________________________________________________________

# Fonction principale

def main():
    process_pic(DIR)

#_________________________________________________________________

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

#_________________________________________________________________

# process_pic : fonction qui permet transformer les images.

def process_pic(DIR):
    #Importer les photos
    image_folder = "./images_processed/"
    # Effacer le dossier cible s'il existe
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    # Créer le dossier cible
    os.makedirs(image_folder)
    print("Ready to save images in", image_folder)
    for img in os.listdir(DIR):
        file_name = str(img)
        file_path = os.path.join(image_folder, file_name)
        path = os.path.join(DIR, img)
        new_img = Image.open(path)
        # Resizer l'image
        new_image = resize_image(new_img, size=IMG_SIZE, bg_color="white")
        new_image = np.array(new_image)
        # L'ajouter au dossier cible
        plt.imsave(file_path, new_image)
        # Création d'une image avec un effet miroir afin d'augmenter le nombre d'image de notre dataset.
        file_path = os.path.join(image_folder, file_name)
        flip_img = Image.open(path)
        new_image_flip = resize_image(flip_img, size=IMG_SIZE, bg_color="white")
        new_image_flip = np.array(new_image_flip)
        new_image_flip = np.fliplr(new_image_flip)
        file_name_flip = str(img).split(".")[0]+"00.jpeg"
        file_path_flip = os.path.join(image_folder, file_name_flip)
        plt.imsave(file_path_flip, new_image_flip)
    print("Les images sont processées.")

if __name__ == '__main__':
    main()