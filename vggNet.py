#!/usr/bin/python
'''
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
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adadelta, SGD
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
    def __init__(self, data_folder, image_width=224, image_height=224, n_components=3, training_set_size=4000, test_set_size=973, n_classes=10, n_epochs=1):

        self.data_folder = data_folder
        self.image_width = image_width
        self.image_height = image_height
        self.n_components = n_components
        self.training_set_size = training_set_size  # Total training examples over all classes
        self.test_set_size = test_set_size # Total testing examples over all classes
        self.n_classes = n_classes
        self.n_epochs = n_epochs

        train_data_dir = data_folder + 'training'
        test_data_dir = data_folder + 'testing'

		# Rescale  the images and apply noise to the training dataset 
        train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_data = ImageDataGenerator(rescale=1./255)

		# Generator gives output of a tuple in form: (inputs, targets)
        self.training_generator = train_data.flow_from_directory(train_data_dir, target_size=(image_height, image_width), batch_size=10, class_mode='categorical')
        self.testing_generator = test_data.flow_from_directory(test_data_dir, target_size=(image_height, image_width), batch_size=10, class_mode='categorical')

if __name__ == '__main__':

# Get weights path from args
    weightsPath=args["weights"] if args["load_model"] > 0 else None

print('VGG-A CNN')

# Filepath of dataset
data = Data(data_folder='C:/Users/Doug/Py/Data/tiny-net-data/')

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(data.n_components, data.image_height, data.image_width)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', init='glorot_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', init='glorot_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', init='glorot_normal'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', init='glorot_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', init='glorot_normal'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', init='glorot_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', init='glorot_normal'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4', init='glorot_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5)) 
model.add(Dense(4096, activation='relu', init='glorot_normal')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, name='dense_classifier', activation='softmax', init='glorot_normal'))

# Load weights path (for reading or saving) if param (-w 1) included
if weightsPath is not None:
    model.load_weights(weightsPath)

# Compile model 
print ("[INFO] compiling model...")
sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True) 
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

# Fit model if no model is to be loaded
if args["load_model"] < 0:
    print ('Fitting model') 
    results = model.fit_generator(data.training_generator, samples_per_epoch=data.training_set_size, nb_epoch=data.n_epochs, validation_data=data.testing_generator, nb_val_samples=data.test_set_size)
	
	#  "Accuracy"
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.show()
	# "Loss"
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.show()


# Saved model to file if param (-s 1) included
if args["save_model"] > 0:
    print ('[INFO] dumping weights to file') 
    model.save_weights(args["weights"], overwrite=True)

# Show classes
print ('\nClasses: ', data.training_generator.class_indices)

# Evaluate overall model
print ("\n\nEvaluating model ... Please wait a few minutes")
score = model.evaluate_generator(data.testing_generator, 1000)

print ('Score: %1.3f' % score[0]) 
print ('Accuracy: %1.3f' % score[1]) 
