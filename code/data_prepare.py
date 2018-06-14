# -*- coding:utf-8 -*-
'''
 LUNA2016 data prepare
'''

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


from project_config import *

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def extract_real_cubic_from_mhd(dcim_path,annatation_file,plot_output_path,normalization_output_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: annatation_file:            the annatation csv file,contains every nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(plot ),every nodule end up withs three size
      @param:normalization_output_path:   the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(after normalization)
    '''
    file_list=glob(dcim_path+"*.mhd")
    # The locations of the nodes
    df_node = pd.read_csv(annatation_file)
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()

    for img_file in file_list:
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        #file_name = str(img_file).split("/")[-1]
        file_name = os.path.basename(img_file)
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            # go through all nodes
            print("begin to process real nodules...")
            img_array = img_array.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z
            print(img_array.shape)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                nodule_pos_str = str(node_x)+"_"+str(node_y)+"_"+str(node_z)
                # every nodules saved into size of 20x20x6,30x30x10,40x40x26
                imgs1 = np.ndarray([20,20,6],dtype=np.float32)
                imgs2 = np.ndarray([30,30,10],dtype=np.float32)
                imgs3 = np.ndarray([40,40,26],dtype=np.float32)
                center = np.array([node_x, node_y, node_z])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                print(v_center[0],v_center[1],v_center[2])
                try:
                    # these following imgs saves for plot
                    imgs1[:,:,:]=img_array[int(v_center[0]-10):int(v_center[0]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[2]-3):int(v_center[2]+3)]
                    imgs2[:,:,:]=img_array[int(v_center[0]-15):int(v_center[0]+15),int(v_center[1]-15):int(v_center[1]+15),int(v_center[2]-5):int(v_center[2]+5)]
                    imgs3[:,:,:]=img_array[int(v_center[0]-20):int(v_center[0]+20),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-13):int(v_center[2]+13)]

                    # these following are the standard data as input of CNN
                    truncate_hu(imgs1)
                    truncate_hu(imgs2)
                    truncate_hu(imgs3)
                    imgs1 = normalazation(imgs1)
                    imgs2 = normalazation(imgs2)
                    imgs3 = normalazation(imgs3)
                    np.save(os.path.join(normalization_output_path, "%d_real_size20x20.npy" % node_idx),imgs1)
                    np.save(os.path.join(normalization_output_path, "%d_real_size30x30.npy" % node_idx),imgs2)
                    np.save(os.path.join(normalization_output_path, "%d_real_size40x40.npy" % node_idx),imgs3)
                    print("save real %d finished!..." % node_idx)

                except Exception as e:
                    print(" process images %s error..."%str(file_name))
                    print(Exception,":",e)
                    traceback.print_exc()


def extract_fake_cubic_from_mhd(dcim_path,candidate_file,plot_output_path,normalization_output_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: candidate_file:             the candidate csv file,contains every **fake** nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(plot ),every nodule end up withs three size
      @param:normalization_output_path:   the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(after normalization)
    '''
    file_list=glob(dcim_path+"*.mhd")
    # The locations of the nodes
    df_node = pd.read_csv(candidate_file)
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()

    for img_file in file_list:
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        #file_name = str(img_file).split("/")[-1]
        file_name = os.path.basename(img_file)
        num = 0
        if mini_df.shape[0]>0 and num<10000: # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            # go through all nodes
            print("begin to process fake nodules...")
            img_array = img_array.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z
            print(img_array.shape)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                nodule_pos_str = str(node_x) + "_" + str(node_y) + "_" + str(node_z)
                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space ( x,y,z ordering)
                # false nodule
                # every nodules saved into size of 20x20x6,30x30x10,40x40x26
                imgs_fake1 = np.ndarray([20, 20, 6], dtype=np.float32)
                imgs_fake2 = np.ndarray([30, 30, 10], dtype=np.float32)
                imgs_fake3 = np.ndarray([40, 40, 26], dtype=np.float32)
                try:
                    # these following imgs saves for plot
                    imgs_fake1[:,:,:] = img_array[int(v_center[0]-10):int(v_center[0]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[2]-3):int(v_center[2]+3)]
                    imgs_fake2[:,:,:] = img_array[int(v_center[0]-15):int(v_center[0]+15),int(v_center[1]-15):int(v_center[1]+15),int(v_center[2]-5):int(v_center[2]+5)]
                    imgs_fake3[:,:,:] = img_array[int(v_center[0]-20):int(v_center[0]+20),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-13):int(v_center[2]+13)]

                    # save fake nodule
                    truncate_hu(imgs_fake1)
                    truncate_hu(imgs_fake2)
                    truncate_hu(imgs_fake3)
                    imgs_fake1 = normalazation(imgs_fake1)
                    imgs_fake2 = normalazation(imgs_fake2)
                    imgs_fake3 = normalazation(imgs_fake3)
                    np.save(os.path.join(normalization_output_path, "%d_fake_size20x20.npy" % node_idx), imgs_fake1)
                    np.save(os.path.join(normalization_output_path, "%d_fake_size30x30.npy" % node_idx), imgs_fake2)
                    np.save(os.path.join(normalization_output_path, "%d_fake_size40x40.npy" % node_idx), imgs_fake3)

                    num = num+1

                    print("save fake %d finished!..." % node_idx)
                except Exception as e:
                    print(" process images %s error..." % str(file_name))
                    print(Exception, ":", e)
                    traceback.print_exc()

def check_nan(path):
    '''
     a function to check if there is nan value in current npy file path
    :param path:
    :return:
    '''
    for file in os.listdir(path):
        array = np.load(os.path.join(path,file))
        a = array[np.isnan(array)]
        if len(a)>0:
            print("file is nan :  ",file )
            print(a)

def plot_cubic(npy_file):
    '''
       plot the cubic slice by slice

    :param npy_file:
    :return:
    '''
    cubic_array = np.load(npy_file)
    f, plots = plt.subplots(int(cubic_array.shape[2]/3), 3, figsize=(50, 50))
    for i in range(1, cubic_array.shape[2]+1):
        plots[int(i / 3), int((i % 3) )].axis('off')
        plots[int(i / 3), int((i % 3) )].imshow(cubic_array[:,:,i], cmap=plt.cm.bone)

def plot_3d_cubic(image):
    '''
        plot the 3D cubic
    :param image:   image saved as npy file path
    :return:
    '''
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = np.load(image)
    verts, faces = measure.marching_cubes(image,0)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
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

def search(path, word):
    '''
       find filename match keyword from path
    :param path:  path search from
    :param word:  keyword should be matched
    :return:
    '''
    filelist = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and word in filename:
            filelist.append(fp)
        elif os.path.isdir(fp):
            search(fp, word)
    return filelist


def get_all_filename(path,size):
    list_real = search(path, 'real_size' + str(size) + "x" + str(size))
    list_fake = search(path, 'fake_size' + str(size) + "x" + str(size))
    return list_real+list_fake

def get_test_batch(path):
    '''
            prepare every batch file data and label 【test】
    :param path:
    :return:
    '''
    files = [os.path.join(path,f) for f in os.listdir(path)]
    batch_array = []
    batch_label = []
    for npy in files:
        try:
            if len(batch_label)<32:
                arr = np.load(npy)
                batch_array.append(arr)
                if 'real_' in npy.split("/")[-1]:
                    batch_label.append([0, 1])
                elif 'fake_' in npy.split("/")[-1]:
                    batch_label.append([1, 0])
        except Exception as e:
            print("file not exists! %s" % npy)
            batch_array.append(batch_array[-1])  # some nodule process error leading nonexistent of the file, using the last file copy to fill
            print(e.message)

    return np.array(batch_array), np.array(batch_label)

def get_train_batch(batch_filename):
    '''
      prepare every batch file data and label 【train】
    :param batch_filename:
    :return:
    '''
    batch_array = []
    batch_label = []
    for npy in batch_filename:
        try:
            arr = np.load(npy)
            batch_array.append(arr)

            if 'real_' in  npy.split("/")[-1]:
                batch_label.append([0,1])
            elif 'fake_' in npy.split("/")[-1]:
                batch_label.append([1,0])
        except Exception  as e:
            print("file not exists! %s"%npy)
            batch_array.append(batch_array[-1])  # some nodule process error leading nonexistent of the file, using the last file copy to fill

    return np.array(batch_array),np.array(batch_label)


if __name__ =='__main__':

    #base_dir = '/home/ubuntu/data/'
    #annatation_file = '/home/ubuntu/data/CSVFILES/annotations.csv'
    #candidate_file = '/home/ubuntu/data/CSVFILES/candidates.csv'
    #plot_output_path = '/home/ubuntu/data/output'
    #normalization_output_path = '/home/ubuntu/data/train-3d'
    #est_path = '/home/ubuntu/data/test-3d'

    for i in range(0,9):
        print('start precessing subset: ', i)
        dcim_path = base_dir +'subset'+str(i)+"/"
        extract_real_cubic_from_mhd(dcim_path, annatation_file, plot_output_path,normalazation_output_path)
        extract_fake_cubic_from_mhd(dcim_path, candidate_file, plot_output_path, normalazation_output_path)
        print('end precessing subset: ', i)
        print('-'*10)

    for i in range(9,10):
        print('start precessing subset: ', i)
        dcim_path = base_dir +'subset'+str(i)+"/"
        extract_real_cubic_from_mhd(dcim_path, annatation_file, plot_output_path,test_path)
        extract_fake_cubic_from_mhd(dcim_path, candidate_file, plot_output_path,test_path)
        print('end precessing subset: ', i)
        print('-'*10)
        #print("  #######  extract cubic from subset%d finished    #######"%i)
    print("finished!...")





