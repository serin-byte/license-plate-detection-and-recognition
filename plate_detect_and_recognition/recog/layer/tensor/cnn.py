# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D,BatchNormalization,Reshape


class Layer(Model):
    def __init__(self):
        super(Layer, self).__init__()
        self.c1 = Conv2D(32, (3, 3), activation='relu')
        # self.b1 = BatchNormalization(momentum=0.9)
        # self.a1 = Activation('relu')
        self.p1 = MaxPooling2D((2, 2), strides=2)
        # self.d1 = Dropout(0.2)

        self.c2 = Conv2D(64, (3, 3), activation='relu')
        # self.b2 = BatchNormalization(momentum=0.9)
        self.p2 = MaxPooling2D((2, 2), strides=2)

        self.c3 = Conv2D(128, (3, 3), activation='relu')
        self.p3 = MaxPooling2D((2, 2), strides=2)

        self.f4 = Flatten()
        self.d4 = Dense(7 * 65)
        self.r4 = Reshape([7, 65])
        self.a4 = Activation("softmax")

    def call(self, x):
        x = self.c1(x)
        # x = self.b1(x)
        x = self.p1(x)

        x = self.c2(x)
        # x = self.b2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.p3(x)

        x = self.f4(x)
        x = self.d4(x)
        x = self.r4(x)
        y = self.a4(x)

        return y
