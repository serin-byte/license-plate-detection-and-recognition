# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import Model

class Predict:
    def __init__(self, model_path, image_path, chars):
        self.model_path = model_path
        self.image_path = image_path
        self.chars = chars

    def load(self):
        try:
            self.new_model = tf.keras.models.load_model(self.model_path)
            print("加载模型成功")
        except Exception as ret:
            print("出现异常:", ret)

    def vec2text(self, vec):
        text = []
        for i, c in enumerate(vec):
            text.append(self.chars[c])
        return "".join(text)

    def forecast(self):
        # image_path = r"E:\端到端车牌识别-byme\train_plate\[2, 51, 57, 33, 40, 40, 40].jpg"
        start_time = time.time()
        img = Image.open(self.image_path)
        img = img.resize((220, 70), Image.ANTIALIAS)  # 先高后宽
        plt.imshow(img)
        img = np.array(img.convert('L'))
        img = tf.reshape(img, (70, 220, 1))  # 高度、宽度、通道
        img = np.array(img)
        print(img.shape)
        img_arr = img / 255.0
        x_predict = img_arr[tf.newaxis, ...]

        result = self.new_model.predict(x_predict)
        pred = np.argmax(result, axis=2)[0]
        zuihou = self.vec2text(pred)

        end_time = time.time()
        total_time = end_time - start_time
        print("所花费的时间为：%s秒" % total_time)
        print("预测的结果是：%s" % zuihou)

        plt.pause(1)
        plt.close()


    def run(self):
        self.load()
        self.forecast()

