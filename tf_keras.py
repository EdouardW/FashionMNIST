import tensorflow
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Data
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#print ("Dimensions train_images: {}, Dimensions train_labels: {}".format(len(train_images), len(train_labels)))
#print ("Dimensions test_images: {}, Dimensions test_labels: {}".format(len(test_images), len(test_labels)))

print (train_images[0].shape)