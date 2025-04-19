def print_prog():
    print(
        """
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

mnist_path = 'mnist.npz'  

with np.load(mnist_path) as data:
    train_images, train_labels = data['x_train'], data['y_train']
    test_images, test_labels = data['x_test'], data['y_test']

train_images, test_images = train_images / 255.0, test_images / 255.0

 

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

 

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
    )
