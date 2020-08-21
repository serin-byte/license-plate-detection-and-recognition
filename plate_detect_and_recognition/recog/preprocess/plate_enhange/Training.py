from pre_process.Utilities.io import DataLoader
from pre_process.Utilities.lossMetric import *
from pre_process.Utilities.trainVal import MinMaxGame
from pre_process.Models.RRDBNet import RRDBNet
from pre_process.Models.GAN import Discriminator

#%% md

### Load in the training dataset
# I used the Chinese City Parking Dataset for this project. Please download the dataset from https://github.com/detectRecog/CCPD
# Before loading the dataset, it is critical that you run the preprocessing script (preprocess.py) first!!!
# `
# python preprocess.py 5 PATH_TO_UNZIPPED_DATA PATH_TO_OUTPUT_DIR
# `

#%%

import numpy as np
import glob
PATH = 'PATH_TO_OUTPUT_DIR/192_96'  # only use images with shape 192 by 96 for training
files = glob.glob(PATH + '/*.jpg') * 3  # data augmentation, same image with different brightness and contrast
np.random.shuffle(files)
train, val = files[:int(len(files)*0.8)], files[int(len(files)*0.8):]
loader = DataLoader()
trainData = DataLoader().load(train, batchSize=16)
valData = DataLoader().load(val, batchSize=64)

#%% md

### Training

#%%

discriminator = Discriminator()
extractor = buildExtractor()
generator = RRDBNet(blockNum=10)

#%% md

# *  It's a good idea to pretrain the generator model before the min-max game - Reference: https://arxiv.org/abs/1701.00160

#%%

# a simple custom loss function that combines MAE loss with VGG loss, as defined in the SRGAN paper
def contentLoss(y_true, y_pred):
    featurePred = extractor(y_pred)
    feature = extractor(y_true)
    mae = tf.reduce_mean(tfk.losses.mae(y_true, y_pred))
    return 0.1*tf.reduce_mean(tfk.losses.mse(featurePred, feature)) + mae

optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
generator.compile(loss=contentLoss, optimizer=optimizer, metrics=[psnr, ssim])
# epoch is set to 1 for demonstration purpose. In practice I found 20 is a good number
# When the model reaches PSNR=20/ssim=0.65, we can start the min-max game
history = generator.fit(x=trainData, validation_data=valData, epochs=1, steps_per_epoch=300, validation_steps=100)

#%% md

### Generative adverserial network training

#%%

# training parameter. epoch is set to 1 for demonstration
# please train the network utill it reaches snRatio ~= 22
PARAMS = dict(lrGenerator = 1e-4,
              lrDiscriminator = 1e-4,
              epochs = 1,
              stepsPerEpoch = 500,
              valSteps = 100)
game = MinMaxGame(generator, discriminator, extractor)
log, valLog = game.train(trainData, valData, PARAMS)
# ideally peak signal noise ratio(snRation or psnr) should reach ~22

#%% md

### Save the model
# Because I defined the model as inherited class of tf keras model, they cannot be safely serialized.
# Therefore, please save the weights only and follow the instructions in tutorial 1 to reload the model
# You can found my pretrained model in the *Pretrained* folder

#%%

#generator.save_weights(YOUR_PATH), save_format='tf')
