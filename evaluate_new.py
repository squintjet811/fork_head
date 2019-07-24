#! /usr/bin/env python

import csv
import os
import sys
from math import sqrt

import numpy as np
# from keras.applications.mobilenet import DepthwiseConv2D # This is old
from keras.layers import DepthwiseConv2D
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, mean_squared_error, roc_auc_score, classification_report
from keras.utils.np_utils import to_categorical

from kalpha import krippendorff_alpha
from net import process_data

CLASSIFY = 0
REGRESS = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = 128

EMOTIONS = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprised',
    4: 'Afraid',
    5: 'Disgusted',
    6: 'Angry',
    7: 'Contemptuous'
}

EMOTIONS_S = {
    0: 'Neutral',
    1: 'Delighted',
    2: 'Happy',
    3: 'Miserable',
    4: 'Sad',
    5: 'Surprised',
    6: 'Angry',
    7: 'Afraid',
    8: 'Disgusted',
    9: 'Contemptuous'
}


def get_classifier_predictions_new_frozen(model, image)

def get_classifier_predictions_new(model, img):
    img = image.img_to_array(img) / 255
    img = img[np.newaxis, ...]
    # print(img)
    # print(img.shape)
    p = model.predict(img)
    predict_emotion = np.argmax(p)

    return predict_emotion