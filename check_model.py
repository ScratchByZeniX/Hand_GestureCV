import tensorflow as tf

model = tf.keras.models.load_model("gesture_model.keras")

print("Input Shape :", model.input_shape)
print("Output Shape:", model.output_shape)
import tensorflow as tf

model = tf.keras.models.load_model("gesture_model.keras")

for i, layer in enumerate(model.layers):
    print(i, layer.name)