import tensorflow as tf
import cv2 as cv
import numpy as np


def prediction(file_parth):
    try:
        filenames = []
        with open('Model/class_names.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                filenames.append(line.rstrip())
        f.close()
        new_model = tf.keras.models.load_model('Model/nn.h5')
        image = cv.imread(file_parth)
        image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
        label_of_image = new_model.predict(np.array([image]))
        return filenames[np.argmax(label_of_image)].capitalize(), round((np.max(label_of_image)/np.sum(label_of_image)*100))
    except:
        return "No model", 0
