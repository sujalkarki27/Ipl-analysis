def print_prog():
    print(
        """
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

 

mnist_path = 'mnist.npz'
with np.load(mnist_path, allow_pickle=True) as data:
    train_images, train_labels = data['x_train'], data['y_train']
    test_images, test_labels = data['x_test'], data['y_test']

 

train_images, test_images = train_images / 255.0, test_images / 255.0

 

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

lenet_model = models.Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.AveragePooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.AveragePooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])

 

alexnet_model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(10)
])

 

lenet_model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

alexnet_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

 

lenet_history = lenet_model.fit(train_images, train_labels, epochs=5,
                                validation_data=(test_images, test_labels))

alexnet_history = alexnet_model.fit(train_images, train_labels, epochs=5,
                                    validation_data=(test_images, test_labels))

 

plt.plot(lenet_history.history['accuracy'], label='LeNet Training Accuracy')
plt.plot(lenet_history.history['val_accuracy'], label='LeNet Validation Accuracy')

 

plt.plot(alexnet_history.history['accuracy'], label='AlexNet Training Accuracy')
plt.plot(alexnet_history.history['val_accuracy'], label='AlexNet Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""
    )
