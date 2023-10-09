#create dataset
#post processing on the images so we create the data we need to train the classifiers

import mediapipe as mp
import os
import cv2
import pickle
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawings_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# we will iterate on all images in data folder and extract landmarks from them and save the extracted data in a place to be used later in training classifiers
DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we need to convert images to RGB (from bgr) in order to input them into mediapipe as it

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks: # check if there is at least 1 hand in the img
            for hand_landmarks in results.multi_hand_landmarks:
               for j in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[j].x # getting the x-axis of each landmark
                    y = hand_landmarks.landmark[j].y # getting the y-axis of each landmark
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle','wb')
pickle.dump({'data':data, 'labels': labels}, f)
f.close()

# by that we have created the dataset we will use to train our classifier
