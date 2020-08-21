# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(r'../..')
from Utilities.io import DataLoader
from Utilities.painter import Visualizer
from Models.RRDBNet import RRDBNet
import glob

dic_config = {}

def init():
    global painter, data, dic_config
    loader = DataLoader()
    painter = Visualizer()
    dic_config = {
        "DATA_PATH": 'Samples',
        'MODEL_PATH': 'Pretrained/rrdb',
        'save_path': 'show/'
    }
    data = loader.load(glob.glob(dic_config['DATA_PATH'] + '/*.jpg'), batchSize=1)


def load_demo():
    i = 0
    for downSample, original in data.take(2):
        i += 1
        painter.plot(downSample, original, dic_config['save_path']+'demo%s.png' % i)


def enhance_plate():
    model = RRDBNet(blockNum=10)
    model.load_weights(dic_config['MODEL_PATH'])
    i = 0
    for downSample, original in data.take(4):
        i +=1
        yPred = model.predict(downSample)
        painter.plot(downSample, original, dic_config['save_path']+"enhance%s.png" % i, yPred)


def run():
    init()
    load_demo()
    enhance_plate()


if __name__ == "__main__":
    run()
