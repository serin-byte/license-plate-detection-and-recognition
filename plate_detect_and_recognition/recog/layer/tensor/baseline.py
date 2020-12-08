# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Dropout,BatchNormalization,MaxPool2D,Reshape


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)

        self.d3 = Dense(7 * 65)
        self.r1 = Reshape([7, 65])
        self.a2 = Activation("softmax")


    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)

        x = self.d3(x)
        x = self.r1(x)
        y = self.a2(x)

        return y