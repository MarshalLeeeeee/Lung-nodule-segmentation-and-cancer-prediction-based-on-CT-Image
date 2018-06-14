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

file_list=glob(path+"*true*")
for file in file_list:
	e = np.load(file)
	file_name = file.split('/')[-1]
	np.save(path+'1_'+file_name,np.transpose(e,(0,2,1)))
	np.save(path+'2_'+file_name,np.flipud(e))
	np.save(path+'3_'+file_name,np.flipud(np.transpose(e,(0,2,1))))
	np.save(path+'4_'+file_name,np.fliplr(e))
	np.save(path+'5_'+file_name,np.transpose(np.fliplr(e),(0,2,1)))
	np.save(path+'6_'+file_name,np.flipud(np.fliplr(e)))
	np.save(path+'7_'+file_name,np.flipud(np.transpose(np.fliplr(e),(0,2,1))))
	np.save(path+'8_'+file_name,np.transpose(np.rot90(e,1,(1,2)),(0,2,1)))
	np.save(path+'9_'+file_name,np.rot90(e,1,(1,2)))
	np.save(path+'10_'+file_name,np.flipud(np.transpose(np.rot90(e,1,(1,2)),(0,2,1))))
	np.save(path+'11_'+file_name,np.flipud(np.rot90(e,1,(1,2))))


'''
print(e.shape)
print(e) # 1-1
print('-'*10)
print(np.transpose(e,(0,2,1))) # 1-2
print('-'*10)
print(np.flipud(e)) # 1-3
print('-'*10)
print(np.flipud(np.transpose(e,(0,2,1)))) # 1-4
print('-'*10)
print(np.fliplr(e)) # 3-1
print('-'*10)
print(np.transpose(np.fliplr(e),(0,2,1))) # 3-1
print('-'*10)
print(np.flipud(np.fliplr(e))) # 3-1
print('-'*10)
print(np.flipud(np.transpose(np.fliplr(e),(0,2,1)))) # 3-1
print('-'*10)
print(np.transpose(np.rot90(e,1,(1,2)),(0,2,1))) # 2-1
print('-'*10)
print(np.rot90(e,1,(1,2))) # 2-2
print('-'*10)
print(np.flipud(np.transpose(np.rot90(e,1,(1,2)),(0,2,1)))) # 2-3
print('-'*10)
print(np.flipud(np.rot90(e,1,(1,2)))) # 2-2
print('-'*10)
'''
