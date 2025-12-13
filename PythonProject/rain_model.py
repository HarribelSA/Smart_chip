import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_DIR = "training_data/"
IMG_SIZE = 224
BATCH = 16

datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    subset="training"
)

val = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    subset="validation"
)

model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet"
)

x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(64, activation="relu")(x)
output = tf.keras.layers.Dense(3, activation="softmax")(x)

final_model = tf.keras.Model(model.input, output)

final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

final_model.fit(train, validation_data=val, epochs=10)

final_model.save("model/injury_model.h5")

print("Model Saved → model/injury_model.h5")
