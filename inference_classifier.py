# for testing the classifier
import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawings_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'A', 1:'B', 2:'C', 3:'L', 4:'V'}
while True:

    data_aux = []
    X = []
    Y = []

    ref,frame= cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks: # check if there is at least 1 hand in the img
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawings_styles.get_default_hand_landmarks_style(),
                mp_drawings_styles.get_default_hand_connections_style()) # to draw landmarkes on top of webcam

        # iterate over all the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x # getting the x-axis of each landmark
                y = hand_landmarks.landmark[j].y # getting the y-axis of each landmark
                data_aux.append(x)
                data_aux.append(y)
                X.append(x)
                Y.append(y)

        x1 = int(min(X) * W) - 10 # converting them to int as after multiyplying with of the frame (float) it will be a float value
        y1 = int(min(Y) * H) - 10

        x2 = int(max(X) * W) - 10
        y2 = int(max(Y) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])] # prediction[0] as the prediction is returned in a list from (lisy of one element)
        print(predicted_character)

        cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,0), 4)
        cv2.putText(frame, predicted_character, (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)

# then we want to extract all the landmakers in the hand appears in the webcam

cap.release()
cv2.destroyAllWindows()