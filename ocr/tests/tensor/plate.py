# -*- coding: utf-8 -*-
import sys
sys.path.append(r'../..')

import os

from ocr.model.tensor.cnn_plate import CnnTrainer
from ocr.predict.tensor.cnn_plate import Predict

data_dir = '/apps/data/dingjian_testing'

dic_config = {
    'movie_path': os.path.join(data_dir, 'raw/movies.csv'),
    'rating_path': os.path.join(data_dir, 'raw/ratings.csv'),
    'model_path': os.path.join(data_dir, 'model/item2vec'),

    model_path = 'my_model_cnn',

    x_train_path = r'E:\lizi\dingjian_testing\data\x_train_c65.npy'
    y_train_path = r'E:\lizi\dingjian_testing\data\y_train_c65.npy'
    x_test_path = r'E:\lizi\dingjian_testing\data\x_test_c65.npy'
    y_test_path = r'E:\lizi\dingjian_testing\data\y_test_c65.npy'

    model_path = 'my_model_cnn'
    image_path = r'E:\lizi\dingjian_testing\picture\pic.jpg'
    'chars' = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
             "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7",
             "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
             "V","W", "X", "Y", "Z"]
}


def _train():

    t = CnnTrainer(dic_config)
    t.run()


def _infer():

    i = Predict(dic_config)
    i.run()


if __name__ == '__main__':
    _train()
    _infer()
