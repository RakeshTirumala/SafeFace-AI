
import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array


model = load_model("bestmodelRFMD.hdf5")

results = {1: 'mask', 2: 'without mask', 0: 'improper mask'}
GR_dict = {1: (0, 255, 0), 2: (0, 0, 255), 0:(0,0,255)}
rect_size = 4
cap = cv2.VideoCapture(0)
haarcascade = cv2.CascadeClassifier(
    'P:\Folder Bulbasaur\mini_project\haarcascade_frontalface_default.xml')
while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y + h, x:x + w]
        Image = cv2.resize(face_img, (28, 28))
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Image = img_to_array(Image)
        Image = np.array(Image)
        Image = np.reshape(Image, (1,28,28,1))
        print("[INFO]: PREDICTING...")


        result = model.predict(Image)

        #print("Info: Result:", result)
        result = list((result[0]))
        print("Label list:", result)
        label = result.index(max(result))

        cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()