from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
from glob import glob
import numpy as np
import tensorflow as tf
from model.extract_bottleneck_features import *
#from detectors.dog_detector import *
#from detectors.face_detector  import *
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tqdm import tqdm
import numpy as np
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
        self.bottleneck_features = np.load('model/bottleneck_features/DogVGG19Data.npz')
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

# Face detector

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Dog detector

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# Preprocess the data
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# Returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


if __name__ == '__main__':
    weight_path = 'model/saved_model/weights.best.from_VGG19.hdf5'
    #print(face_detector('uploaded_images/test.jpg') + '0')
    model = NeuralNetwork(weight_path)
    #print(face_detector('uploaded_images/test.jpg') + '1')
    result = model.predict_breed('model/uploaded_images/test.jpg')
    print(result)
