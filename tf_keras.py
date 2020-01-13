import tensorflow
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scaling de la valeur des pixels de chaque image (initialement compris entre 0 et 255 )
train_images = train_images / 255.0
test_images = test_images / 255.0

#------------------  Data et première visualisation 
#print ("Dimensions train_images: {}, Dimensions train_labels: {}".format(len(train_images), len(train_labels)))
#print ("Dimensions test_images: {}, Dimensions test_labels: {}".format(len(test_images), len(test_labels)))
#print (train_images[0].shape)  # Array 28 x 28
#print (train_images[0])

# ----------------- On peut afficher les images (initialement pixels compris entre 0 et 250)
#plt.figure()
#plt.imshow(train_images[0]) # or plt.imshow(train_images[0], cmap = plt.cm.binary)
#plt.colorbar()
#plt.grid(False)
#lt.show()

#------------------ On affiche les 25 premières images 
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(test_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[test_labels[i]])
#plt.show()

# On construit le modèle (Sequential: Linear stack of layers)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # Input => This layer has no parameters to learn; it only reformats the data.
    keras.layers.Dense(128, activation="relu" ), # Hidden 
    keras.layers.Dense(10, activation="softmax")  # Output => returns an array of 10 probability scores that sum to 1.
])

# Before the model is ready for training, it needs a few more settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# On fit (et on peut tester sur le jeu de test ou prédire un nouvel élement)
model.fit(train_images, train_labels, epochs=5)

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)

prediction = model.predict(test_images)

for i in range (5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predicted: " + class_names[np.argmax(prediction[i])])  # retourne un array avec la proba pour les 10 classes => argmax ressort la proba max 
    plt.show()

# Pour prédire un unqiue élément, on ne peut pas le passer directement
# Tf est contruit pour prédire un enemble 
# Add the image to a batch where it's the only member.

img = test_images[15] # On choisit une image  => shape (28 , 28)
img = (np.expand_dims(img,0))  # shape (1, 28,28)

predictions_single = model.predict(img)

print ("Actual: " + class_names[test_labels[15]])
print("Predicted: "+ class_names[np.argmax(predictions_single)])

