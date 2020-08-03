# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from layer.tensor.resnet import ResNet18


class Train:

    def __init__(self, dic_config={}):

        self.model_path = dic_config['model_path']

        self.x_train = np.load(dic_config['x_train_path'])
        self.y_train = np.load(dic_config['y_train_path'])
        self.x_test = np.load(dic_config['x_test_path'])
        self.y_test = np.load(dic_config['y_test_path'])

        self.categorical_type = dic_config.get('categorical_type', 1)

    # 都是计算多分类crossentropy的，只是对y的格式要求不同。
    # 如果是categorical_crossentropy，那y必须是one-hot处理过的
    def categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=['accuracy'])

    def sparse_categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

    # 如果是sparse_categorical_crossentropy，那y就是原始的整数形式，比如[1, 0, 2, 0, 2]
    def train(self):
        self.model = ResNet18([1])

        if self.categorical_type == 1:
            self.categorical()
        elif self.categorical_type == 2:
            self.sparse_categorical()

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path + 'test_7_22.ckpt',
        #                                                  save_weights_only=True,
        #                                                  save_best_only=True)

        # self.history = self.model.fit(self.x_train, self.y_train,
        #                               batch_size=32, epochs=1,  # 我电脑支持32size
        #                               validation_data=(self.x_test, self.y_test),
        #                               validation_freq=1, callbacks=[cp_callback])

        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=32, epochs=1,  # 我电脑支持32size
                                      validation_data=(self.x_test, self.y_test),
                                      validation_freq=1)

        self.model.summary()

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
        self.show()
