"""
    Preprocessing for U-net
    Thresholding and mask the lung part
    Use annotation to mask the nodules 
"""

import numpy as np
import pandas as pd
import os

from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi

from glob import glob
from tqdm import tqdm

import SimpleITK as sitk
import scipy.misc
import matplotlib.pyplot as plt


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

def create_nodule_mask(imagePath, cands, fcount, subsetnum,final_lung_mask,final_nodule_mask):
    #if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)
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

    #create nodule mask
    nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)

    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros((lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape	
    i_start = 0
    i_end = 0
    flag = 0
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset/2))
        lower_offset = int(offset - upper_offset)

        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
        nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

    # return final_lung_mask,final_nodule_mask
    # save images.
    np.save(os.path.join(OUTPUT_PATH,"lung_mask_%04d_%04d.npy" % (subsetnum, fcount)),lung_mask_512)
    np.save(os.path.join(OUTPUT_PATH,"nodule_mask_%04d_%04d.npy" % (subsetnum, fcount)),nodule_mask_512)



# Helper function to get rows in data frame associated with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

# Getting list of image files
LUNA_DATA_PATH = '/home/marshallee/Documents/lung/'
OUTPUT_PATH = '/home/marshallee/Documents/lung/output'

final_lung_mask = np.zeros((1,512,512))
final_nodule_mask = np.zeros((1,512,512))

# create a list of subsets, which are lists of file paths
FILE_LIST = []
for i in range(0, 9):
    LUNA_SUBSET_PATH = LUNA_DATA_PATH + 'subset'+str(i)+'/'
    FILE_LIST.append(glob(LUNA_SUBSET_PATH + '*.mhd'))


for subsetnum, subsetlist in enumerate(FILE_LIST):
    # The locations of the nodes
    df_node = pd.read_csv(LUNA_DATA_PATH + "mask-generate/CSVFILES/annotations.csv")
    #df_node = pd.read_csv(LUNA_DATA_PATH + "CSVFILES/annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(subsetlist, file_name))
    df_node = df_node.dropna()

    # Looping over the image files
    for fcount, img_file in enumerate(tqdm(subsetlist)):
        mini_df = df_node[df_node["file"]==img_file] # get all nodules associate with file
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those
            # feeding mini_df to the function will work for "cands"
            final_lung_mask, final_nodule_mask =create_nodule_mask(img_file, mini_df, fcount, subsetnum,final_lung_mask,final_nodule_mask)

final_lung_mask = final_lung_mask[1:]
final_nodule_mask = final_nodule_mask[1:]
print(final_lung_mask.shape)
print(final_nodule_mask.shape)
np.save(os.path.join(OUTPUT_PATH,'final_lung_mask.npy'),final_lung_mask)
np.save(os.path.join(OUTPUT_PATH,'final_nodule_mask.npy'),final_nodule_mask)

