import tensorflow as tf
import numpy as np
import json
import pyttsx3
import cv2
import mediapipe as mp


# Load Model
model = tf.keras.models.load_model("gesture_model.keras")


# Voice
engine = pyttsx3.init()
engine.setProperty("rate", 150)

last_gesture = ""


# Load Labels
with open("labels.json", "r") as f:
    labels = json.load(f)

class_names = {v: k for k, v in labels.items()}


display_names = {
    "01_palm": "Palm",
    "02_l": "L",
    "03_fist": "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb": "Thumb",
    "06_index": "Index",
    "07_ok": "OK",
    "08_palm_moved": "Palm Moved",
    "09_c": "C",
    "10_down": "Down"
}


# MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# Camera
cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        break


    frame = cv2.flip(frame, 1)


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    result = hands.process(rgb)


    gesture_name = ""
    confidence = 0


    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            img = cv2.resize(
                rgb,
                (224, 224)
            )

            img = img.astype("float32") / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )


            prediction = model.predict(
                img,
                verbose=0
            )


            index = np.argmax(prediction)

            confidence = np.max(prediction) * 100


            gesture = class_names[index]

            gesture_name = display_names.get(
                gesture,
                gesture
            )


    # Voice Output
    if gesture_name != "" and gesture_name != last_gesture:

        print("Speaking:", gesture_name)

        engine.stop()

        engine.say(gesture_name)

        engine.runAndWait()

        last_gesture = gesture_name



    # Show Gesture
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # Show Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    cv2.imshow(
        "Human Gesture Recognition",
        frame
    )


    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()

cv2.destroyAllWindows()