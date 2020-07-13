# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
# sys.path.append(r'C:\Users\master\reco')
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from ocr.layer.tensor.cnn_plate import Cnn
from tensorflow.keras import Model


class CnnTrainer:

    def __init__(self, model_path, x_train_path, y_train_path, x_test_path, y_test_path):
        self.save_path = model_path

        self.x_train = np.load(x_train_path)
        self.y_train = np.load(y_train_path)
        self.x_test = np.load(x_test_path)
        self.y_test = np.load(y_test_path)

    def train(self):
        self.model = Cnn()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=[tf.keras.metrics.Accuracy()])

        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=64, epochs=1,
                                      validation_data=(self.x_test, self.y_test),
                                      validation_freq=1)
        self.model.summary()

    def show(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
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
