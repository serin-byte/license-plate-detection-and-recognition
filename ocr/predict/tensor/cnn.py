# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Model


class Predict:
    def __init__(self, model_path):
        self.model_path = model_path
        
        self.fashion = tf.keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.fashion.load_data()

        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1) 
            
    def load(self):
        self.new_model = tf.keras.models.load_model(self.model_path)

    def run(self):
        self.load()
        print(self.new_model.predict(self.x_test[0][tf.newaxis, ...]))
