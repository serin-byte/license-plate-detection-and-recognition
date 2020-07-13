# -*- coding: utf-8 -*-
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense


class Cnn(Model):
    def __init__(self):
        super(Cnn, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same') 
        # self.b1 = BatchNormalization()
        self.a1 = Activation('relu')  
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  
        # self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        # self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        # x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        # x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        # x = self.d2(x)
        y = self.f2(x)
        
        return y
