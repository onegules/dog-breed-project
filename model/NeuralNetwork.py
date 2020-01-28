from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
from glob import glob
import numpy as np
import tensorflow as tf
from extract_bottleneck_features import *
from dog_detector import *
from face_detector  import *
import logging

logger = logging.getLogger('root')

class NeuralNetwork():
    def __init__(self, weight_path):
        #tf.reset_default_graph()
        #self.session = tf.Session()
        #self.graph = tf.get_default_graph()


        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        logging.info("neural network initialised")

        # load list of dog names
        self.bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
        self.train_VGG19 = self.bottleneck_features['train']
        self.test_VGG19 = self.bottleneck_features['test']
        self.dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]
        # Define model Architecture
        self.VGG19_model = Sequential()
        self.VGG19_model.add(GlobalAveragePooling2D(input_shape = self.train_VGG19.shape[1:]))
        self.VGG19_model.add(Dense(133, activation = 'softmax'))

        # Compile the model
        self.VGG19_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

        # Load the model
        self.VGG19_model.load_weights(weight_path)


    def model_predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.VGG19_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        result = self.dog_names[np.argmax(predicted_vector)]

        return result


    def predict_breed(self, path):
        if dog_detector(path):
            return self.model_predict_breed(path)

        elif face_detector(path):
            return self.model_predict_breed(path)

        else:
            return "Error. Neither dog nor human detected."

if __name__ == '__main__':
    weight_path = 'saved_models/weights.best.from_VGG19.hdf5'
    model = NeuralNetwork(weight_path)
    result = model.predict_breed('uploaded_images\IMG_20200113_181002.jpg')
    print(result)
