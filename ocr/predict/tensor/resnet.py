# -*- coding: utf-8 -*-
import tensorflow as tf
import time
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import Model


class Predict:
    def __init__(self, dic_config={}):
        self.chars = dic_config['chars']

    def load(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("加载模型成功")
        except Exception as ret:
            print("出现异常:", ret)

    def vec2text(self, vec):
        text = []
        for i, c in enumerate(vec):
            text.append(self.chars[c])
        return "".join(text)

    def load_image(self, image_path):
        img = Image.open(image_path)

        img = img.resize((220, 70), Image.ANTIALIAS)  # 先高后宽
        img = np.array(img.convert('L'))
        img = tf.reshape(img, (70, 220, 1))  # 高度、宽度、通道
        img = np.array(img)
        print(img.shape)
        img_arr = img / 255.0
        self.x_predict = img_arr[tf.newaxis, ...]

    def predict(self, image_path):
        # start_time = time.time()
        self.load_image(image_path)

        result = self.model.predict(self.x_predict)
        pred = np.argmax(result, axis=2)[0]
        return pred, self.vec2text(pred)

        # end_time = time.time()
        # total_time = end_time - start_time
        # print("所花费的时间为：%s秒" % total_time)
        # print("预测的结果是：%s" % rs)
