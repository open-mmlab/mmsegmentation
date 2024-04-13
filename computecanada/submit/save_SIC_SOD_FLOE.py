import os
import numpy as np
import xarray as xr
import random
from tqdm import tqdm
import cv2
# -- Built-in modules -- #
import copy
from tqdm import tqdm
# -- Proprietary modules -- #
from AI4ArcticSeaIceChallenge.convert_raw_icechart import convert_polygon_icechart

# Example usage
folder_path = '/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_test_v3'
output_folder = '/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_test_v3_segmaps'
os.makedirs(output_folder, exist_ok=True)

def duplicate_channels(arr):
    # Reshape the array to add a new dimension for the channels
    arr_channels = arr[:, :, np.newaxis]
    # Duplicate the values along the new channel dimension
    arr_rgb = np.concatenate((arr_channels, arr_channels, arr_channels), axis=2)
    return arr_rgb

def save_image(arr,path, suffix):
    path = path.replace(".nc", suffix)
    cv2.imwrite(path, arr)

files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

for file in tqdm(files):
    save_path = os.path.join(output_folder,file)
    xarr = xr.open_dataset(os.path.join(folder_path,file), engine='h5netcdf')
    xarr = convert_polygon_icechart(xarr)

    # SIC, SOD FLOE
    SIC = xarr['SIC'].values
    SOD = xarr['SOD'].values
    FLOE = xarr['FLOE'].values

    # Convert nan to num
    SIC = np.nan_to_num(SIC,nan=255).astype(np.uint8)
    SOD = np.nan_to_num(SOD,nan=255).astype(np.uint8)
    FLOE = np.nan_to_num(FLOE,nan=255).astype(np.uint8)

    SIC = duplicate_channels(SIC)
    SOD = duplicate_channels(SOD)
    FLOE = duplicate_channels(FLOE)
    save_image(SIC, save_path,'_SIC.png')
    save_image(SOD, save_path,'_SOD.png')
    save_image(FLOE, save_path,'_FLOE.png')

print("Maps saved to", output_folder)





