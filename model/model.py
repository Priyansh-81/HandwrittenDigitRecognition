import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Flatten, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.datasets import mnist
import pickle

class DigitRecognizer:
    def __init__(self, enable_augmentation=True):
        layers_list = [
            Input(shape=(784,)),
            Reshape((28, 28, 1))
        ]

        if enable_augmentation:
            # Data augmentation: tilt (rotation), zoom, and translation (shift)
            # Note: We intentionally avoid horizontal/vertical flips for digits because
            # flipping changes digits (e.g., 6 <-> 9) or creates non-existent symbols (flipped 3, 7).
            layers_list.extend([
                RandomRotation(0.05),           # Tilt: ~18 degrees rotation
                RandomZoom((-0.1, 0.1)),        # Zoom: 10% zoom in/out
                RandomTranslation(0.08, 0.08),  # Shift: 8% horizontal/vertical shift
            ])

        layers_list.extend([
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),                       # Increased from 0.001 to prevent overfitting
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),                       # Increased from 0.001 to prevent overfitting
            Dense(10, activation='softmax')
        ])

        self.model = Sequential(layers_list)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def train_model(self, X_train_flat, y_train, epochs=15, batch_size=64, validation_split=0.1):
        return self.model.fit(
            X_train_flat,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def evaluate_model(self, X_test_flat, y_test):
        return self.model.evaluate(X_test_flat, y_test)
    
    def predict(self, img):   
        img=img/255.0
        img_flat=img.reshape(1, 28*28)
        predict=self.model.predict(img_flat)
        return predict
    
if __name__ == "__main__":
    model = DigitRecognizer()
    model.compile_model()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_flat = X_train.reshape(-1, 28*28)
    X_train_flat = X_train_flat / 255.0
    X_test_flat = X_test.reshape(-1, 28*28)
    X_test_flat = X_test_flat / 255.0
    model.train_model(X_train_flat, y_train, epochs=10)
    results = model.evaluate_model(X_test_flat, y_test)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_dir, "model_results.pkl")
    model_path = os.path.join(current_dir, "model.keras")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    model.model.save(model_path)
    print(results)

