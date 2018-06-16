"""
Purpose: train a machine learning segmenter that can segment out the nodules on a given 2D patient CT scan slice
Note:
- this will train from scratch, with no preloaded weights
- weights are saved to unet.hdf5 in the specified output folder
"""

from __future__ import print_function
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

#WORKING_PATH = "/home/marshallee/Documents/lung/output/"

TRAIN_PATH = '/home/ubuntu/data/train_pre/'
VAL_PATH = '/home/ubuntu/data/val_pre/'
TEST_PATH = '/home/ubuntu/data/test_pre/'
IMG_ROWS = 512
IMG_COLS = 512

SMOOTH = 1.

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (np.sum(y_true_f) + np.sum(y_pred_f) + SMOOTH)

def get_unet():
    inputs = Input((1,IMG_ROWS, IMG_COLS))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def generator(path,batch_size):

    lung_mask_list = glob(path + 'final_lung_mask_*.npy')
    nodule_mask_list = glob(path + 'final_nodule_mask_*.npy')
    lung_mask_list.sort()
    nodule_mask_list.sort()

    flag = 0
    start = 0
    cnt = 0

    while (1):
        
        for i in range(len(lung_mask_list)):
            lung_file = lung_mask_list[i]
            nodule_file = nodule_mask_list[i]
            lung = np.load(lung_file)
            nodule = np.load(nodule_file)

            if(flag):
                lung_train = np.concatenate((lung_train,lung[0:batch_supply]),axis=0).reshape([batch_size,1,512,512])
                nodule_train = np.concatenate((nodule_train,nodule[0:batch_supply]),axis=0).reshape([batch_size,1,512,512])
                yield (lung_train / 255.0, nodule_train / 255.0)
                start = batch_supply
                flag = 0
            while(start + batch_size < len(lung)):
                lung_train = lung[start:start+batch_size].reshape([batch_size,1,512,512])
                nodule_train = nodule[start:start+batch_size].reshape([batch_size,1,512,512])
                yield (lung_train / 255.0, nodule_train / 255.0)
                start += batch_size
            if(start + batch_size == len(lung)):
                lung_train = lung[start:start+batch_size].reshape([batch_size,1,512,512])
                nodule_train = nodule[start:start+batch_size].reshape([batch_size,1,512,512])
                yield (lung_train / 255.0, nodule_train / 255.0)
                flag = 0
                start = 0
            else:
                lung_train = lung[start:]
                nodule_train = nodule[start:]
                batch_supply = batch_size - (len(lung)-start)
                start = 0
                flag = 1


def train_generator(batch_size):
    model = get_unet()
    print('model compileover ...')
    model.fit_generator(generator(TRAIN_PATH,batch_size),steps_per_epoch = 4540, epochs = 2, verbose = 1, validation_data=generator(VAL_PATH,batch_size),validation_steps=650)
    
    data_pred = np.load(TEST_PATH+'final_lung_mask_9.npy').reshape([1007,1,512,512])
    nodule_true = np.load(TEST_PATH+'final_nodule_mask_9.npy').reshape([1007,1,512,512])

    '''
    nodule_pred = model.predict(data_pred)

    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(data_pred[0],cmap='gray')
    ax[0,1].imshow(nodule_true[0],cmap='gray')
    ax[1,0].imshow(data_pred[0]*nodule_true[0],cmap='gray')
    plt.show()

    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(data_pred[0],cmap='gray')
    ax[0,1].imshow(nodule_pred[0],cmap='gray')
    ax[1,0].imshow(data_pred[0]*nodule_pred[0],cmap='gray')
    '''

    print(model.evaluate(data_pred,nodule_true,batch_size=2,verbose=1))


    
if __name__ == '__main__':
    batch_size = 2
    train_generator(batch_size)