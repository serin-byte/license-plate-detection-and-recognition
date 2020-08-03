# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from layer.tensor.resnet1 import resnet18
import os
import tensorflow as tf

from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# tf.random.set_seed(2345)


class Train:

    def __init__(self, dic_config={}):

        self.model_path = dic_config['model_path']
        self.categorical_type = dic_config.get('categorical_type', 1)



        self.x_train = np.load(dic_config['x_train_path'])[0:50]
        # self.x_train = self.x_train[np.newaxis, :]
        # print(self.x_train.shape)


        self.y_train = np.load(dic_config['y_train_path'])[0:50]


        self.x_test = np.load(dic_config['x_test_path'])
        self.y_test = np.load(dic_config['y_test_path'])
        # print("形状大小", self.x_test.shape)

    # 都是计算多分类crossentropy的，只是对y的格式要求不同。
    # 如果是categorical_crossentropy，那y必须是one-hot处理过的
    def categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=['accuracy'])

    def sparse_categorical(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

    # 如果是sparse_categorical_crossentropy，那y就是原始的整数形式，比如[1, 0, 2, 0, 2]
    def train(self):


        self.model = resnet18()
        if self.categorical_type == 1:
            self.categorical()
        elif self.categorical_type == 2:
            self.sparse_categorical()

        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=10, epochs=1)
                                      # validation_data=(self.x_test, self.y_test),
                                      # validation_freq=1)

        self.model.summary()

        # self.model.build(input_shape=(None, 70, 220, 1))
        # self.model.summary()
        # optimizer = optimizers.Adam(lr=1e-3)
        # for epoch in range(50):
        #     for step, (x, y) in (self.x_train, self.x_test):
        #         with tf.GradientTape() as tape:
        #             logits = self.model(x)
        #             # y_onehot = tf.one_hot(y, depth=10)
        #
        #             loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
        #             loss = tf.reduce_mean(loss)
        #         grads = tape.gradient(loss, self.model.trainable_variables)
        #         optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        #         if step % 100 == 0:
        #             print(epoch, step, 'loss', float(loss))
        #     total_num = 0
        #     total_correct = 0
        #     for x, y in self.x_test:
        #         logits = self.model(x)
        #         prob = tf.nn.softmax(logits, axis=1)
        #         pred = tf.argmax(prob, axis=1)
        #         pred = tf.cast(pred, dtype=tf.int32)
        #         correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        #         correct = tf.reduce_sum(correct)
        #         total_num += x.shape[0]
        #         total_correct += int(correct)
        #     acc = total_correct / total_num
        #     print(epoch, 'acc:', acc)

    def preprocess(self, x, y):
        x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
        y = tf.cast(y, dtype=tf.int32)
        return x, y

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
        print("成功保存模型")

    def run(self):
        self.train()
        self.save()
        # self.show()

