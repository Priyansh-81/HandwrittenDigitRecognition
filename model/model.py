import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.datasets import mnist
import pickle

class DigitRecognizer:
    def __init__(self):
        self.model=Sequential([
            Dense(128, activation='relu', input_shape=(28*28,), kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
            Dense(10, activation='softmax')
        ])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def compile_model(self):
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])

    def train_model(self, X_train_flat, y_train, epochs=50):
        self.model.fit(X_train_flat, y_train, epochs=epochs)

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

