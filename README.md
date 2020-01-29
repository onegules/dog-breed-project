## Project Overview

This project is originally from Udacity's project found
[here](https://github.com/udacity/dog-project). The goal of the project is to
create a machine learning model for classifying images: in this case dogs. This
project uses a pretrained face detector and dog detector. The machine learning
model uses transfer learning from the VGG19 model architecture. Learn more [here](https://arxiv.org/abs/1409.1556)

## REQUIRED STEPS

### Step 0: Data

The following folder needs to be included in the data folder:

https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

### Step 1: Bottleneck Features

We also need the bottleneck features. They can be found [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz).
Put this file in a new folder with the following path ../model/bottleneck_features/DogVGG19Data.npz

### Step 2: Requirements

The requirements to run this project are found in requirements.txt. To install
them, simply run

'''python
pip install -r requirements.txt
'''

in your console while in the main folder.

### Step 3: Run app.py

In the console type in

'''python
python app.py
'''

and go to http://127.0.0.1:5000/

## The files and folders

### model

We first have the haarcascades folder which contains information that allows us
to read in the pretrained face detection model.

In the saved_model folder we have the weights to be loaded into the CNN. If you
wish to retrain the model, this is where the weights will be loaded with the name
weights.best.from_VGG19.hdf5.

extract_bottleneck_features does exactly what it says and extracts bottleneck
features from the required architecture to be used in transfer learning for our
model (in this case VGG19, but if you want to use other architectures, the code
is there as well).

__init__ is an empty file that python needs to be able to import extract_bottleneck_features.

### static

The static folder holds all the javascript and css information in main.js and main.css.
Further, it also contains all the pictures used in the templates folder.

### templates

Templates holds all HTML files. index.html is the homepage and process.html is the
HTML version of the jupyter notebook I used CNN and goes through the
steps I took to generate the CNN. Finally, the base.html and logos folder contain
information for the github picture and link found in the bottom left of the homepage.

### app.py

This file contains all the backend for the flask web app.

### neural_network

This file contains all the set up for NeuralNetwork: the class that defines the
CNN and reads in the required weights. The file also contains functions for face
detection for human and dog faces. The human face detection uses haarcascades as
mentioned above, and for dog faces we use ResNet50 to pretrain the dog detection.

### util.py

This file contains information that allows a serialized json request to be transformed
into an image.


## Results

The working web app:

![Screenshot](/data/screenshot.PNG)

The metric we were looking at was accuracy: how close the model could predict the
dog breed while given a train set and a test set. The model trained from the VGG19
architecture resulted in a 73.9% accuracy. To improve this, I would suggest tuning
the parameters, use data augmentation or increase the data set size.

## Acknowledgments

To complete this project I used the following resources:

* https://udacity.com
* https://stackoverflow.com
* https://getbootstrap.com/docs/4.4/getting-started/introduction/
* https://www.w3schools.com/
* https://github.com/mtobeiyf/keras-flask-deploy-webapp
* https://flask.palletsprojects.com/en/1.1.x/
* https://www.python.org/doc/
