from PIL import Image, ImageOps # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import re

IMG_SIZE = (300,300)
# DIR = "./fma/"

def main():
    DIR = input("Quel dossier voulez-vous étudier ?")
    DIR = "./" + DIR + "/"
    train_data, label_data = load_training_data(DIR)
    print(label_data)
    get_x_y(train_data, label_data)


def resize_image(src_image, size=IMG_SIZE, bg_color="white"): 
    from PIL import Image, ImageOps 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square background image
    new_image = Image.new("L", size, bg_color)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    
    #convert the image to grayscale
    new_image = new_image.convert('L')
    
    # return the resized image
    return new_image

def load_training_data(DIR):
    train_data = []
    label_data = []
    im_number = 0
    for img in os.listdir(DIR):
        label = str(img).split('.')[0]
        label = re.sub('\d', '', label)
        label = re.findall('[A-Z][^A-Z]*', label)
        label = " ".join(label)
        path = os.path.join(DIR, img)
        img = Image.open(path)
        new_image = resize_image(img, size=IMG_SIZE, bg_color="white")
        train_data.append(np.array(new_image))
        label_data.append(label)
        # Basic Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        new_image_flip = resize_image(flip_img, size=IMG_SIZE, bg_color="white")
        new_image_flip = np.array(new_image_flip)
        new_image_flip = np.fliplr(new_image_flip)
        train_data.append(new_image_flip)
        label_data.append(label)
#         shuffle(train_data)
    return train_data, label_data

def get_x_y(train_data, label_data):
    X = np.array(train_data).reshape((len(train_data),-1))
    y = label_data
    print(X.shape)
    return X, y

if __name__ == '__main__':
    main()