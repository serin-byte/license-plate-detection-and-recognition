# -*- coding: utf-8 -*-
import sys
sys.path.append(r'C:\Users\master\reco')

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from layer.tensor.cnn import Cnn


class CnnTrainer:
    
    def __init__(self, model_path):
        self.save_path = model_path
    
        self.fashion = tf.keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.fashion.load_data()
        
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
    
    def train(self):
        self.model = Cnn()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train,  
                                      batch_size=32, epochs=5,
                                      validation_data=(self.x_test, self.y_test),
                                      validation_freq=1)
        self.model.summary()
        
    def show(self):
        acc = self.history.history['sparse_categorical_accuracy']
        val_acc = self.history.history['val_sparse_categorical_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure()
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    # 保存模型
    def save(self):
        self.model.save(self.save_path)

    def run(self):
        self.train()
        self.save()
        self.show()
