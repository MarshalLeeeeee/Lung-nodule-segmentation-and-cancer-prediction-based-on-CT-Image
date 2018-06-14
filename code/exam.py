import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import traceback
import random
from PIL import Image

path = '/home/ubuntu/data/train-3d/'
#path='try/'

'''file_list=glob(path+"*.npy")
data_x = np.load(file_list[0])
np.save(path+'try.npy',np.transpose(data_x,(1,2,0)))'''
file_list=glob(path+"*.npy")

for file in file_list:
	data = np.load(file)
	print(data.shape)
	if data.shape != (26,40,40):
		os.remove(file)