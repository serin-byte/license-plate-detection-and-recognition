# -*- coding: utf-8 -*-
# import sys
# sys.path.append(r'../..')
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from recog.layer.tensor.cnn import Layer, Baseline, AlexNet8, Inception10, LeNet5, ResNet18, VGG16


class Train:

    def __init__(self, dic_config={}):

        self.model_path = dic_config['model_path']

        self.x_train = np.load(dic_config['x_train_path'])
        self.y_train = np.load(dic_config['y_train_path'])
        self.x_test = None
        self.y_test = None

        self.categorical_type = dic_config.get('categorical_type', 1)
        self.validation_split = dic_config.get('validation_split', 0.2)
        self.demon = dic_config.get('demon', True)

        if 'x_test_path' in dic_config:
            self.x_test = np.load(dic_config['x_test_path'])
            self.y_test = np.load(dic_config['y_test_path'])
            self.validation_split = 0

    # 都是计算多分类crossentropy的，只是对y的格式要求不同。
    # 如果是categorical_crossentropy，那y必须是one-hot处理过的
    def categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=['accuracy'])

    def sparse_categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

    # 如果是sparse_categorical_crossentropy，那y就是原始的整数形式，比如[1, 0, 2, 0, 2]
    def train(self):
        # self.model = Layer()
        # self.model = Baseline()
        # self.model = AlexNet8()
        # self.model = Inception10()
        # self.model = LeNet5()
        self.model = ResNet18()
        # self.model = VGG16()

        if self.categorical_type == 1:
            self.categorical()
        elif self.categorical_type == 2:
            self.sparse_categorical()

        if self.validation_split == 0:
            self.history = self.model.fit(self.x_train, self.y_train,
                                          batch_size=16, epochs=10,
                                          validation_data=(self.x_test, self.y_test),
                                          validation_freq=1)
        else:
            self.history = self.model.fit(self.x_train, self.y_train,
                                          batch_size=16, epochs=10,
                                          validation_split=self.validation_split)

        self.model.summary()

    def evalute(self):
        '''
        模型评估
        :return:
        '''
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(test_loss, test_acc)

    def show(self):
        if self.categorical_type == 1:
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
        elif self.categorical_type == 2:
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
        self.model.save(self.model_path)

    def run(self):
        self.train()
        self.save()
        if self.demon:
            self.show()
