import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
DATASET_PATH = "dataset"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20
train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True

)
train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True

)
validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False

)
labels = train_generator.class_indices

with open("labels.json","w") as f:

    json.dump(labels,f,indent=4)

print(labels)
base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)
for layer in base_model.layers:

    layer.trainable=False
    # -------------------------------
# Build Classification Head
# -------------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

predictions = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# -------------------------------
# Compile Model
# -------------------------------

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "gesture_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# -------------------------------
# Train Model
# -------------------------------

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]

)

# -------------------------------
# Save Final Model
# -------------------------------

model.save("gesture_model.keras")

print("\nModel Saved Successfully!")

# -------------------------------
# Final Accuracy
# -------------------------------

loss, accuracy = model.evaluate(validation_generator)

print("\nValidation Accuracy : {:.2f}%".format(accuracy * 100))
plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.legend()

plt.grid(True)

plt.show()
plt.figure(figsize=(10,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")

plt.legend()

plt.grid(True)

plt.show()