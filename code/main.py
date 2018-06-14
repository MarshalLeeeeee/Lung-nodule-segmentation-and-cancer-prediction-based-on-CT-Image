# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from model2 import model
import numpy as np

from project_config import *

if __name__ =='__main__':
    print(" beigin...")
    epoch = 5
    model = model(learning_rate,keep_prob,batch_size,epoch)
    model.inference(normalization_output_path,test_path,2,True)







