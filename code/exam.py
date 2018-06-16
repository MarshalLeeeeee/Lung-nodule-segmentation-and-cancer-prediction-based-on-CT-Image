"""
	Remove some wrong file
"""

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

file_list=glob(path+"*.npy")

for file in file_list:
	data = np.load(file)
	print(data.shape)
	if data.shape != (26,40,40):
		os.remove(file)