#!/usr/bin/python
'''
Cifar-10 classification
See: https://www.cs.toronto.edu/~kriz/cifar.html for more information
USAGE:
python cifar10_cnn.py --load-model 1 --weights cifr10_weights.hdf5
or
python cifar10_cnn.py --save-model 1 --weights cifr10_weights.hdf5
'''

import cPickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adadelta

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


class Cifar10(object):
    def __init__(self, data_folder, n_training_files=5, image_width=32, image_height=32, n_components=3,
                 data_block_size=10000, training_set_size=50000, test_set_size=10000, n_classes=10):

        self.data_folder = data_folder
        self.n_training_files = n_training_files
        self.image_width = image_width
        self.image_height = image_height
        self.n_components = n_components
        self.data_block_size = data_block_size
        self.training_set_size = training_set_size
        self.test_set_size = test_set_size
        self.n_classes = n_classes

        # Define training and test sets
        self.training_set = np.ndarray(shape=(self.training_set_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)
        self.training_set_labels = np.ndarray(shape=(self.training_set_size, self.n_classes)).astype(np.float32)

        self.test_set = np.ndarray(shape=(self.test_set_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)
        self.test_set_labels = np.ndarray(shape=(self.test_set_size, self.n_classes)).astype(np.float32)

        # Load data
        print('Loading training data')

        # Read training data
        for i in range(n_training_files):
            with open(self.data_folder + 'data_batch_' + str(i + 1), 'rb') as training_file:
                training_dict = cPickle.load(training_file)

                self.training_set[(self.data_block_size * i):(self.data_block_size * (i + 1)), :, :, :] = training_dict['data']. \
                    reshape((self.data_block_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)

                for idx, label in enumerate(training_dict['labels']):
                    self.training_set_labels[(self.data_block_size * i) + idx, :] = self.to_class(label)

        # Read test data
        print('Loading test data')

        with open(self.data_folder + 'test_batch', 'rb') as test_file:
            test_dict = cPickle.load(test_file)

            self.test_set[0:self.data_block_size, :, :, :] = test_dict['data']. \
                reshape((self.data_block_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)

            for idx, label in enumerate(test_dict['labels']):
                self.test_set_labels[idx, :] = self.to_class(label)

        # Read label data
        with open(data_folder + 'batches.meta', 'rb') as label_file:
            self.label_dict = cPickle.load(label_file)
            self.label_names = self.label_dict['label_names']

        # Normalize training and test data
        self.X_train, self.Y_train = (self.training_set / 255), self.training_set_labels
        self.X_test, self.Y_test = (self.test_set / 255), self.test_set_labels

    def to_class(self, label_idx):
        class_data = np.zeros(shape=self.n_classes).astype(np.float32)
        class_data[label_idx] = 1.0
        return class_data

    def to_label(self, class_vector):
        return self.label_names[np.argmax(class_vector)]

    def to_RGB(self, data):
        img = np.ndarray(shape=(self.image_width, self.image_height, self.n_components)).astype(np.uint8)

        for i in range(self.n_components):
            img[:, :, i] = data[i, :, :]

        return img

    def show_image(self, i, data_set='training'):
        if data_set == 'test':
            a_data_set = self.test_set
            a_data_set_labels = self.test_set_labels
        else:
            a_data_set = self.training_set
            a_data_set_labels = self.training_set_labels

        plt.imshow(self.to_RGB(a_data_set[i]))
        plt.show()
        return a_data_set_labels[i]


if __name__ == '__main__':
    weightsPath=args["weights"] if args["load_model"] > 0 else None

    print('CIFAR-10 Classification')

    cifar10 = Cifar10(data_folder='/home/doug/Downloads/cifar-10-batches2-py/')

    # Create Keras model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu',
                            input_shape=(cifar10.n_components, cifar10.image_height, cifar10.image_width))) #, dim_ordering="th"

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

    # Try to remove some convolutional layer
    #model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(ZeroPadding2D((1, 1)))

    #model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(ZeroPadding2D((1, 1)))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(cifar10.n_classes))
    model.add(Activation('softmax'))

    #load weights if needed
    if weightsPath is not None:
        model.load_weights(weightsPath)

    print("[INFO] compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

#train if not loading weights
if args["load_model"] < 0:
    print('Fitting model')
    model.fit(cifar10.X_train, cifar10.Y_train, batch_size=32, nb_epoch=8, validation_data=(cifar10.X_test, cifar10.Y_test))

# check to see if model should be saved to file
if args["save_model"] > 0:
    print('[INFO] dumping weights to file')
    model.save_weights(args["weights"], overwrite=True)


print("Evaluating model")
score = model.evaluate(cifar10.X_test, cifar10.Y_test, verbose=1)

print('Score: %1.3f' % score[0])
print('Accuracy: %1.3f' % score[1])

# randomly select some images for testing
for i in np.random.choice(np.arange(0, len(cifar10.Y_test)), size=(15,)):
    probs = model.predict(cifar10.X_test[np.newaxis, i])
    print(probs)
    prediction = probs.argmax(axis=1) #cifar10.Y_test[i].argmax(axis=0)
    print('checking %d' % i)
    if prediction == 0:
        print('airplane')
    if prediction == 1:
        print('automobile')
    if prediction == 2:
        print('bird')
    if prediction == 3:
        print('cat')
    if prediction == 4:
        print('deer')
    if prediction == 5:
        print('dog')
    if prediction == 6:
        print('frog')
    if prediction == 7:
        print('horse')
    if prediction == 8:
        print('ship')
    if prediction == 9:
        print('truck')
    print(cifar10.Y_test[i])
    cifar10.show_image(i, 'test')
