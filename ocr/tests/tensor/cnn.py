# -*- coding: utf-8 -*-
import sys
sys.path.append(r'../..')

from model.tensor.cnn import CnnTrainer
from predict.tensor.cnn import Predict


def _train():
    model_path = 'my_model_cnn'
    t = CnnTrainer(model_path)
    t.run()


def _infer():
    model_path = 'my_model_cnn'
    i = Predict(model_path)
    i.run()


if __name__ == '__main__':
    _train()
    _infer()
