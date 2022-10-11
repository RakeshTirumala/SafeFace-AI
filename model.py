import math
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

class Model:
    def __init__(self):
        self.model = Sequential([
            Conv2D(100, (3, 3),activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),

            Conv2D(100, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),

            Flatten(),
            Dropout(0.5),# drops 50% to prevent overfitting
            Dense(50, activation='relu'),
            Dense(3, activation='softmax')])


        self.directoryTrain = r"P:\Folder Bulbasaur\mini_project\Kaggle2\data"
        self.directoryTest = r"P:\Folder Bulbasaur\mini_project\Kaggle2\test"
        #self.categories = ["improper_mask", "with_mask", "without_mask"]

        self.trainAug = None
        self.testAug = None

        #self.toCategorical = tf.keras.utils.to_categorical
        self.trainGenerator = None
        self.testGenerator = None
        self.ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
        self.totalTrainingData = 7142
        self.totalTestingData = 1158
        self.batch_size = 32

    def calculate_spe(self, y):
        return int(math.ceil((1. * y) / self.batch_size))

    def LoadImgs(self):
        print("[INFO]: AUGUMENTING THE DATA...")
        self.trainAug = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        print("[INFO]: AUGUMENTING THE TRAIN DATA...")
        self.trainGenerator = self.trainAug.flow_from_directory(self.directoryTrain,
                                                                batch_size=32,
                                                                target_size=(150,150))
        self.testAug = ImageDataGenerator(rescale=1.0/255)
        print("[INFO]: AUGUMENTING THE TEST DATA...")
        self.testGenerator = self.testAug.flow_from_directory(self.directoryTest, batch_size=32, target_size=(150,150))


    def compileModel(self):
        print("[INFO]: COMPILING MODEL...")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.TrainingEvalSavModel()

    def TrainingEvalSavModel(self):
        print("[INFO]: TRAINING MODEL...")
        checkpoint = self.ModelCheckpoint('model2-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True,
                                     mode='auto')

        H = self.model.fit_generator(self.trainGenerator,
                                      epochs=50,
                                     steps_per_epoch=self.calculate_spe(self.totalTrainingData),
                                     validation_steps=self.calculate_spe(self.totalTestingData),
                                      validation_data=self.testGenerator,
                                      callbacks=[checkpoint])

        print("[INFO]: SAVING MASK DETECTOR MODEL...")
        self.model.save("MaskDiscernment.model", save_format="h5")

        print("[INFO]: PLOTTING THE TRAINING LOSS AND ACCURACY")
        n = 50
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, n), H.history['loss'], label="train_loss")
        plt.plot(np.arange(0, n), H.history['val_loss'], label="val_loss")
        plt.plot(np.arange(0, n), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, n), H.history["val_accuracy"], label="val_acc")
        plt.title("Training loss and accuracy")
        plt.xlabel("Epoch 50")
        plt.ylabel("LOSS/ACCURACY")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")

        print("[INFO]: COMPLETE")


obj = Model()
obj.LoadImgs()
obj.compileModel()
