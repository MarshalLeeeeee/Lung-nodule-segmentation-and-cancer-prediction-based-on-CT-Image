"""
    Utility for vowel extraction
"""

import numpy as np
import pandas as pd
import os
import csv
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import pandas as pd
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import scipy.misc
import matplotlib.pyplot as plt
import traceback
import random
from PIL import Image
from project_config import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
import time

def draw_circles(image,cands,origin,spacing):
    #make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]
    image_mask = np.zeros(image.shape)

    #run over all the nodules in the lungs
    for ca in cands.values:
        #get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4])/2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z,coord_y,coord_x))

        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)

        #determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        #create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
                    if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
    
    return image_mask

def get_segmented_lungs(im, plot=False):
    binary = im < 604
    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0

# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def extract(imagePath, cands, anno, fcount, normalization_output_path):

    print('file %d pre-processing starts...' % fcount)
    img, origin, spacing = load_itk(imagePath)
    print('origin: ', origin)
    print('spacing: ', spacing)

    #calculate resize factor
    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize
    
    #resize image
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
    
    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    nodule_mask = draw_circles(lung_img,anno,origin,new_spacing)

    #create nodule mask
    lung_mask_512 = np.zeros((lung_mask.shape[0], 512, 512))
    nodule_mask_512 = np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape	
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset/2))
        lower_offset = int(offset - upper_offset)
        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:] * nodule_mask[z,:,:]

    print('file %d pre-processing over...' % fcount)
    print('file %d cubic extract strats...' % fcount)

    for node_idx, cur_row in cands.iterrows():
        node_x = cur_row["coordX"]
        node_y = cur_row["coordY"]
        node_z = cur_row["coordZ"]
        nodule_class = cur_row['class']
        center = np.array([node_x, node_y, node_z])  # nodule center
        v_center = world_2_voxel(center,origin,spacing)  # nodule center in voxel space ( x,y,z ordering)

        # every nodules saved into size of 20x20x6,30x30x10,40x40x26
        if nodule_class == 0:
            imgs_fake1 = np.zeros([26, 40, 40], dtype=np.float32)
            try:
                # these following imgs saves for plot
                imgs_fake1 = lung_mask_512[int(v_center[0]-13):int(v_center[0]+13),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-20):int(v_center[2]+20)]
                
                if(np.max(imgs_fake1) != 0.0):
                    np.save(os.path.join(normalization_output_path, "%d_fake_size40x40.npy" % node_idx), imgs_fake1)
                    print('file %d nodule %d fake saves...' % (fcount, node_idx))

            except Exception as e:
                print('*')
        
        elif nodule_class == 1:
            imgs_true1 = np.zeros([26, 40, 40], dtype=np.float32)
            try:
                # these following imgs saves for plot
                imgs_true1 = lung_mask_512[int(v_center[0]-13):int(v_center[0]+13),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-20):int(v_center[2]+20)]
                
                if(np.max(imgs_true1) != 0.0):
                    np.save(os.path.join(normalization_output_path, "%d_true_size40x40.npy" % node_idx), imgs_true1)
                    print('file %d nodule %d true saves...' % (fcount, node_idx))

            except Exception as e:
                print('*')
                

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

if __name__ =='__main__':

    '''
    base_dir = '/home/ubuntu/data/'
    annatation_file = '/home/ubuntu/data/CSVFILES/annotations.csv'
    candidate_file = '/home/ubuntu/data/CSVFILES/candidates.csv'
    plot_output_path = '/home/ubuntu/data/output'
    normalization_output_path = '/home/ubuntu/data/train-3d'
    test_path = '/home/ubuntu/data/test-3d'
    '''
    
    print('test set starts processing...')
    for i in range(9, 10):
        print('subset %d preprocessing starts...' % i)
        LUNA_SUBSET_PATH = base_dir + 'subset'+str(i)+'/'
        FILE_LIST = glob(LUNA_SUBSET_PATH + '*.mhd')

        cand_df_node = pd.read_csv(candidate_file)
        cand_df_node["file"] = cand_df_node["seriesuid"].map(lambda file_name: get_filename(FILE_LIST, file_name))
        cand_df_node = cand_df_node.dropna()

        anno_df_node = pd.read_csv(annatation_file)
        anno_df_node["file"] = anno_df_node["seriesuid"].map(lambda file_name: get_filename(FILE_LIST, file_name))
        anno_df_node = anno_df_node.dropna()

        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(FILE_LIST)):
            cand_mini_df = cand_df_node[cand_df_node["file"]==img_file] # get all nodules associate with file
            anno_mini_df = anno_df_node[anno_df_node["file"]==img_file]
            if cand_mini_df.shape[0]>0 and anno_mini_df.shape[0]>0: # some files may not have a nodule--skipping those
                extract(img_file, cand_mini_df, anno_mini_df, fcount, test_path)
        print('subset %d preprocessing ends...' % i)
        print('*'*20)
    print('test set ends processing...')
    print('-'*25)
    

    print('train set starts processing...')
    for i in range(0, 9):
        print('subset %d preprocessing starts...' % i)
        LUNA_SUBSET_PATH = base_dir + 'subset'+str(i)+'/'
        FILE_LIST = glob(LUNA_SUBSET_PATH + '*.mhd')

        cand_df_node = pd.read_csv(candidate_file)
        cand_df_node["file"] = cand_df_node["seriesuid"].map(lambda file_name: get_filename(FILE_LIST, file_name))
        cand_df_node = cand_df_node.dropna()

        anno_df_node = pd.read_csv(annatation_file)
        anno_df_node["file"] = anno_df_node["seriesuid"].map(lambda file_name: get_filename(FILE_LIST, file_name))
        anno_df_node = anno_df_node.dropna()

        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(FILE_LIST)):
            cand_mini_df = cand_df_node[cand_df_node["file"]==img_file] # get all nodules associate with file
            anno_mini_df = anno_df_node[anno_df_node["file"]==img_file]
            if cand_mini_df.shape[0]>0 and anno_mini_df.shape[0]>0: # some files may not have a nodule--skipping those
                extract(img_file, cand_mini_df, anno_mini_df, fcount, normalization_output_path)
        print('subset %d preprocessing ends...' % i)
        print('*'*20)
    print('train set ends processing...')
    print('-'*25)
    
    