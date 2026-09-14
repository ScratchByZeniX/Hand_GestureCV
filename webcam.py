import cv2
import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model("gesture_model.keras")

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

class_names = list(labels.keys())

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    text = f"{class_names[index]} ({confidence:.2f}%)"

    cv2.putText(frame, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()