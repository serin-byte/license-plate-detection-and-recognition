# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(r'../..')


dic_config = {}


def init():
    global dic_config
    data_dir = 'E:/lizi/dingjian_testing/data/'
    # data_dir = '/apps/data/dingjian_testing/plate'

    dic_config = {
        # 'model_path': os.path.join(data_dir, 'resnet'),
        'model_path': os.path.join(data_dir, 'my_model_cnn'),

        'x_train_path': os.path.join(data_dir, 'x_train_c65.npy'),
        'y_train_path': os.path.join(data_dir, 'y_train_c65.npy'),
        'x_test_path': os.path.join(data_dir, 'x_test_c65.npy'),
        'y_test_path': os.path.join(data_dir, 'y_test_c65.npy'),

        'chars': ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
                  "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7",
                  "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                  "V", "W", "X", "Y", "Z"],

        'image_path': os.path.join(data_dir, 'picture/pic.jpg'),
    }


def generate():
    pass


def train():
    # from model.tensor.cnn import Train
    # from model.tensor.resnet import Train
    from model.tensor.resnet1 import Train

    t = Train(dic_config)
    t.run()


def predict():
    # from predict.tensor.cnn import Predict
    # from predict.tensor.resnet import Predict
    from predict.tensor.resnet1 import Predict

    p = Predict(dic_config)
    p.load(dic_config['model_path'])
    pred, rs = p.predict(dic_config['image_path'])
    print(rs)


def run():
    init()
    # train()
    predict()


if __name__ == '__main__':
    run()
