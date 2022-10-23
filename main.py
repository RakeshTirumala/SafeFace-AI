import math

import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, concatenate, AveragePooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras_preprocessing.image import img_to_array
import numpy as np
from keras.optimizers import Adam
import os
#from keras_preprocessing.image import ImageDataGenerator
import random
import cv2
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class DenseNet:
    def __init__(self, input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None,
                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):

        # Checks
        if nb_classes == None:
            raise Exception(
                'Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

        if compression <= 0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
        elif dense_layers == -1:
            dense_layers = int((depth - 4) / 3)
            if bottleneck:
                dense_layers = int(dense_layers / 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]

        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck
        self.compression = compression
        self.nb_classes = nb_classes

    def build_model(self):
        img_input = Input(shape=self.input_shape, name='img_input')
        nb_channels = self.growth_rate

        x = Conv2D(2 * self.growth_rate, (3, 3),
                   padding='same', strides=(1, 1),
                   kernel_regularizer=keras.regularizers.l2(self.weight_decay))(img_input)

        for block in range(self.dense_blocks - 1):
            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,
                                              self.dropout_rate, self.bottleneck, self.weight_decay)

            x = self.transition_layer(x, nb_channels, self.dropout_rate, self.compression, self.weight_decay)
            nb_channels = int(nb_channels * self.compression)

        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels, self.growth_rate, self.dropout_rate,
                                          self.weight_decay)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        prediction = Dense(self.nb_classes, activation='softmax')(x)

        return Model(inputs=img_input, outputs=prediction, name='densenet')

    print("[INFO]: CALLING THE DENSE BLOCK...")

    def dense_block(self, x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
        for i in range(nb_layers):
            cb = self.convolution_block(x, growth_rate, dropout_rate, bottleneck)
            nb_channels += growth_rate
            x = concatenate([cb, x])

        return x, nb_channels

    print("[INFO]: CALLING THE CONVO BLOCK...")

    def convolution_block(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(nb_channels * bottleneckWidth, (1, 1),
                       kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    print("[INFO]: CALLING THE TRANISITON LAYER...")

    def transition_layer(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',
                   kernel_regularizer=keras.regularizers.l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x


print("[INFO]: INITIALIZING THE DENSENET")
densenet = DenseNet((28, 28, 1), nb_classes=3, depth=35, dropout_rate=0.5)

print("[INFO]: CALLING THE BUILDING MODEL")
model = densenet.build_model()


print("[INFO]: MODEL OPTIMIZER...")
model_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='sparse_categorical_crossentropy', optimizer=model_optimizer, metrics = ['accuracy'])

dataDirPath = "P:\Folder Bulbasaur\mini_project\Kaggle2\data"

data = []
labels = []
random.seed(42)
imagePaths = sorted(list(os.listdir(f'{dataDirPath}')))
random.shuffle(imagePaths)
print("IMAGE PATHS:\n", imagePaths)

for img in imagePaths:
    path = sorted(list(os.listdir(f'{dataDirPath}/' + img)))
    for i in path:
        image = cv2.imread(f"{dataDirPath}/" + img + '/' + i)
        image = cv2.resize(image, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_to_array(image)
        data.append(image)
        l = label = img
        labels.append(l)

data = np.array(data)
labels = np.array(labels)
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)

# labelBin = LabelBinarizer()
# labels = labelBin.fit_transform(labels)

# print("[INFO] data:\n", data)
# print("[INFO] labels:\n", labels)


(xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.20,random_state=42)

filepath = "bestmodelRFMD.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',  verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
H = model.fit(xtrain, ytrain,
                    batch_size=32,
                    epochs=15,
                    shuffle=True,
                    verbose=1,
                    validation_data=(xtest, ytest),
                    steps_per_epoch= int(math.ceil((1. * len(xtrain)) / 32)),
                    callbacks=callbacks_list,
                    validation_steps= int(math.ceil((1. * len(xtest)) / 32))
                    )

# Make predictions
predictions_valid = model.predict_on_batch(xtest)

# Cross-entropy loss score
score = log_loss(ytest, predictions_valid)

model.save_weights('P:\Folder Bulbasaur\Project(4.1)\weights.h5')
model.save('RFMDModel(2.0).model', save_format="h5")

# acc=H.history['accuracy']
# val_acc=H.history['val_acc']
# loss=H.history['loss']
# val_loss=H.history['val_loss']

# epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch

# n = epochs
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, n), H.history['val_loss'], label="val_loss")
# plt.plot(np.arange(0, n), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, n), H.history["val_accuracy"], label="val_acc")
# plt.title("Training loss and accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("LOSS/ACCURACY")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")