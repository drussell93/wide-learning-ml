#!/usr/bin/python
'''
Cifar-10 classification
See: https://www.cs.toronto.edu/~kriz/cifar.html for more information
USAGE:
python cifar10_cnn.py --load-model 1 --weights cifr10_weights.hdf5
or
python cifar10_cnn.py --save-model 1 --weights cifr10_weights.hdf5
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import numpy as np
import cv2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator

# construct and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

np.random.seed()


class Data(object):
    def __init__(self, data_folder, image_width=150, image_height=150, n_components=3, training_set_size=10002, test_set_size=3002, n_classes=2, n_epochs=5):

        self.data_folder = data_folder
        self.image_width = image_width
        self.image_height = image_height
        self.n_components = n_components
        self.training_set_size = training_set_size  # Total training examples over all classes
        self.test_set_size = test_set_size # Total testing examples over all classes
        self.n_classes = n_classes
	self.n_epochs = n_epochs

	#self.training_set_size = 10002
	#self.test_set_size = 3002 

	train_data_dir = data_folder + 'training'
	test_data_dir = data_folder + 'testing'
	image_width, image_height = 150, 150

	# Rescale  the images and apply noise to the training dataset 
	train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	test_data = ImageDataGenerator(rescale=1./255)

	# Generator gives output of a tuple in form: (inputs, targets)
	self.training_generator = train_data.flow_from_directory(train_data_dir, target_size=(image_height, image_width), batch_size=32, class_mode='binary')
	self.testing_generator = test_data.flow_from_directory(test_data_dir, target_size=(image_height, image_width), batch_size=32, class_mode='binary')

if __name__ == '__main__':

# Get weights path from args
    weightsPath=args["weights"] if args["load_model"] > 0 else None

print('Dogs vs Cats Classification')

# Filepath of dataset
data = Data(data_folder='/home/doug/Downloads/data/')

# Simple image classification:
# 3 convolution layers with a ReLU activation. 3 max-pooling layers. 2 fully connected layers. Sigmoid for the binary classifier 
model = Sequential() 
model.add(Convolution2D(32, 3, 3, input_shape=(3, data.image_width, data.image_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Load weights path (for reading or saving) if param (-w 1) included
if weightsPath is not None:
    model.load_weights(weightsPath)

# Compile model 
print("[INFO] compiling model...")
#model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fit model if no model is to be loaded
if args["load_model"] < 0:
    print('Fitting model')
    model.fit_generator(data.training_generator, samples_per_epoch=data.training_set_size, nb_epoch=data.n_epochs, validation_data=data.testing_generator, nb_val_samples=data.test_set_size)

# Saved model to file if param (-s 1) included
if args["save_model"] > 0:
    print('[INFO] dumping weights to file')
    model.save_weights(args["weights"], overwrite=True)


# Helpful 
'''
# Show image
#img=mpimg.imread('/home/doug/Downloads/data/testing/dogs/dog.5001.jpg')
#imgplot = plt.imshow(img)
#plt.show()


# Load single test image for prediction
image = cv2.imread('/home/doug/Downloads/data/testing/dogs/dog.5001.jpg')
image = cv2.resize(image, (150,150))
image = np.expand_dims(image, axis=0)
image=np.reshape(image,[1,3,150,150])
classes = model.predict_classes(image)
print(classes)
'''

# Show classes
print('\nClasses: ', data.training_generator.class_indices)

# Test dogs 
num_dogs_guessed = 0
num_cats_guessed = 0

for i in range(9):
    image = cv2.imread('/home/doug/Downloads/data/testing/dogs/dog.500' + str(i + 1) + '.jpg')
    image = cv2.resize(image, (150,150))
    image = np.expand_dims(image, axis=0)
    image=np.reshape(image,[1,3,150,150])
    probs = model.predict_classes(image)
    print(probs)
    if probs == 0:
        print('cat')
        num_cats_guessed += 1
    if probs == 1:
        print('dog')
        num_dogs_guessed += 1
    
    # Show image
    img=mpimg.imread('/home/doug/Downloads/data/testing/dogs/dog.500' + str(i + 1) + '.jpg')
    imgplot = plt.imshow(img)
    plt.show()

print ('\n\nDog training')
print('Dogs guessed: %d' % num_dogs_guessed, ' Cats guessed: %d' % num_cats_guessed)


# Test cats
num_dogs_guessed = 0
num_cats_guessed = 0

for i in range(9):
    image = cv2.imread('/home/doug/Downloads/data/testing/cats/cat.502' + str(i + 1) + '.jpg')
    image = cv2.resize(image, (150,150))
    image = np.expand_dims(image, axis=0)
    image=np.reshape(image,[1,3,150,150])
    probs = model.predict_classes(image)
    print(probs)
    if probs == 0:
        print('cat')
        num_cats_guessed += 1
    if probs == 1:
        print('dog')
        num_dogs_guessed += 1
    
    # Show image
    img=mpimg.imread('/home/doug/Downloads/data/testing/cats/cat.502' + str(i + 1) + '.jpg')
    imgplot = plt.imshow(img)
    plt.show()

print('\n\nCat training')
print("Dogs guessed: %d" % num_dogs_guessed, ' Cats guessed: %d' % num_cats_guessed)

# Evaluate overall model
print("\n\nEvaluating model")
score = model.evaluate_generator(data.testing_generator)

print('Score: %1.3f' % score[0])
print('Accuracy: %1.3f' % score[1])


