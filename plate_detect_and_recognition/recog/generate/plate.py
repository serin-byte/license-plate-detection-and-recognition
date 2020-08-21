# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import ast

dic_config = {}


# 收集图片和对应的txt文件，将它们做成数据集

def init():
    global dic_config
    dic_config = {
        'test_path': './test_plate/',
        'test_txt': './Ptest.txt',
        'x_test_savepath': './x_test.npy',
        'y_test_savepath': './y_test.npy'
    }


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split("  ")  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = img.resize((220, 70), Image.ANTIALIAS)  # 这是后面的基础和关键
        img = np.array(img.convert('L'))
        img = tf.reshape(img, (70, 220, 1))
        # img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = np.array(img)  # 图片变为np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        text = ast.literal_eval(value[1].strip("\n"))

        # y_.append(ast.literal_eval(value[1].strip("\n")))  # 标签贴到列表y_/去除\n，去除字符串的标致

        vector = np.zeros([7, 65])
        for i, c in enumerate(text):
            # idx = CHAR_SET.index(c)
            vector[i][c] = 1.0
        y_.append(vector)

        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    # y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_


if __name__ == "__main__":
    try:
        x_test, y_test = generateds(dic_config['test_path'], dic_config['test_txt'])
        np.save(dic_config['x_test_savepath'], x_test)
        np.save(dic_config['y_test_savepath'], y_test)
    except Exception as errors:
        print("errors happen:", errors)
    else:
        print("---------------数据集制作完成----------------")
