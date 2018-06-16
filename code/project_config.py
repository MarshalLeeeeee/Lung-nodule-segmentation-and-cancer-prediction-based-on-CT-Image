"""
	Project configuration
	Pathes and train setting
"""
import os
'''
base_dir = '/home/ubuntu/data/'
annatation_file = '/home/ubuntu/data/CSVFILES/annotations.csv'
candidate_file = '/home/ubuntu/data/CSVFILES/candidates.csv'
plot_output_path = '/home/ubuntu/data/output'
normalization_output_path = '/home/ubuntu/data/train-3d'
test_path = '/home/ubuntu/data/test-3d'
'''
base_dir = '/home/marshallee/Documents/lung/'
annatation_file = '/home/marshallee/Documents/lung/mask-generate/CSVFILES/annotations.csv'
candidate_file = '/home/marshallee/Documents/lung/mask-generate/CSVFILES/candidates.csv'
plot_output_path = '/home/marshallee/Documents/lung/output'
normalization_output_path = '/home/marshallee/Documents/lung/output/train-3d-3'
test_path = '/home/marshallee/Documents/lung/output/test-3d-3'
if not os.path.exists(plot_output_path):
    os.mkdir(plot_output_path)
if not os.path.exists(normalization_output_path):
    os.mkdir(normalization_output_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)
###  training and test configuration #####
batch_size = 32
learning_rate = 0.01
keep_prob = 0.7