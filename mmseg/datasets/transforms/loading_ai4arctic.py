import xarray as xr
from mmcv.image import imread
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from icecream import ic
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
import numpy as np
from AI4ArcticSeaIceChallenge.convert_raw_icechart import convert_polygon_icechart
try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class PreLoadImageandSegFromNetCDFFile(BaseTransform):
    """Pre-load images and segmentation maps from NetCDF files into memory.

       This transform pre-loads images and optionally segmentation maps from NetCDF files
       into memory to speed up the data loading process during training and inference.

       Required Keys:
           - img_path: Path to the NetCDF file containing the image data.

       Modified Keys:
           - img: The loaded image data as a numpy array.
           - img_shape: The shape of the loaded image.
           - ori_shape: The original shape of the loaded image.
           - gt_seg_map (optional): The loaded segmentation map as a numpy array.

       Args:
           channels (list[str]): List of variable names to load as channels of the image.
           data_root (str): Root directory of the NetCDF files.
           gt_root (str): Root directory of the ground truth segmentation maps.
           ann_file (str, optional): Path to the annotation file listing NetCDF files to load.
           mean (list[float]): Mean values for normalization of each channel. Defaults to values provided.
           std (list[float]): Standard deviation values for normalization of each channel. Defaults to values provided.
           to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. Defaults to True.
           color_type (str): The color type for image loading. Defaults to 'color'.
           imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
           nan (float): Value to replace NaNs in the image. Defaults to 255.
           downsample_factor (int): Factor by which to downsample the images. Defaults to 10.
           pad_size (tuple[int], optional): Desired size to pad the images to. Defaults to None.
           with_seg (bool): Whether to also load segmentation maps. Defaults to False.
           GT_type (list[str]): List of ground truth types to load (e.g., ['SOD', 'SIC', 'FLOE']). Defaults to ['SOD'].
           ignore_empty (bool): Whether to ignore empty images or non-existent file paths. Defaults to False.
       """

    def __init__(self,
                 channels,
                 data_root,
                 gt_root,
                 ann_file=None,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 color_type='color',
                 imdecode_backend='cv2',
                 nan=255,
                 downsample_factor=10,
                 pad_size=None,
                 with_seg=False,
                 GT_type=['SOD'],
                 ignore_empty=False):
        self.channels = channels
        self.mean = mean
        self.std = std
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
        self.data_root = data_root
        self.gt_root = gt_root
        self.downsample_factor = downsample_factor
        self.nan = nan
        self.with_seg = with_seg
        self.GT_type = GT_type
        self.pad_size = pad_size
        self.nc_files = self.list_nc_files(data_root, ann_file)
        self.pre_loaded_image_dic = {}
        self.pre_loaded_seg_dic = {}
        ic('Starting to load all the images into memory...')
        for filename in tqdm(self.nc_files):
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            img = xarr[self.channels].to_array().data
            img = np.transpose(img, (1, 2, 0))
            mean = np.array(self.mean)
            std = np.array(self.std)
            img = (img - mean) / std
            shape = img.shape
            if self.downsample_factor != 1:
                img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
                img = torch.nn.functional.interpolate(img,
                                                      size=(shape[0] // self.downsample_factor,
                                                            shape[1] // self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0, 2, 3, 1).squeeze(0)
                if self.pad_size is not None:
                    pad_height = max(0, self.pad_size[0] - img.shape[0])
                    pad_width = max(0, self.pad_size[1] - img.shape[1])
                    img = torch.nn.functional.pad(
                        img, (0, 0, 0, pad_width, 0, pad_height), mode='constant', value=self.nan)
                img = img.numpy()
            if self.to_float32:
                img = img.astype(np.float32)
            self.pre_loaded_image_dic[filename] = img
            if self.with_seg:
                seg_maps = []
                for gt_type in self.GT_type:
                    gt_filename = os.path.basename(filename).replace('.nc', f'_{gt_type}.png')
                    gt_filename = os.path.join(self.gt_root, gt_filename)
                    gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
                    if self.downsample_factor != 1:
                        gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0)
                        gt_seg_map = torch.nn.functional.interpolate(gt_seg_map,
                                                                     size=(shape[0] // self.downsample_factor,
                                                                           shape[1] // self.downsample_factor),
                                                                     mode='nearest')
                        gt_seg_map = gt_seg_map.squeeze(0).squeeze(0)
                        if self.pad_size is not None:
                            pad_height = max(0, self.pad_size[0] - gt_seg_map.shape[0])
                            pad_width = max(0, self.pad_size[1] - gt_seg_map.shape[1])
                            gt_seg_map = torch.nn.functional.pad(
                                gt_seg_map, (0, pad_width, 0, pad_height), mode='constant', value=self.nan)
                        gt_seg_map = gt_seg_map.numpy()
                    seg_maps.append(gt_seg_map)
                self.pre_loaded_seg_dic[filename] = np.stack(seg_maps, axis=-1)
        ic('Finished loading all the images into memory...')

    def list_nc_files(self, folder_path, ann_file):
        nc_files = []
        if ann_file is not None:
            with open(ann_file, "r") as file:
                filenames = file.readlines()
            nc_files = [os.path.join(folder_path, filename.strip()) for filename in filenames]
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".nc"):
                        nc_files.append(os.path.join(root, file))
        return nc_files

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        img = self.pre_loaded_image_dic[filename]
        if self.to_float32:
            img = img.astype(np.float32)
        img = np.nan_to_num(img, nan=self.nan)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        if self.with_seg:
            results['gt_seg_map'] = self.pre_loaded_seg_dic[filename]
            results['seg_fields'].append('gt_seg_map')
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'channels={self.channels}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'ignore_empty={self.ignore_empty})')
        return repr_str


@TRANSFORMS.register_module()
class LoadGTFromPNGFile(BaseTransform):
    """Load multiple types of ground truth segmentation maps from PNG files.

    This transform loads multiple types of ground truth segmentation maps from
    PNG files, concatenating them into a single segmentation map with multiple channels.

    Required Keys:
        - img_path: Path to the NetCDF file containing the image data.

    Modified Keys:
        - gt_seg_map: The loaded segmentation map as a numpy array with multiple channels.
        - seg_fields: List of segmentation fields, updated to include 'gt_seg_map'.

    Args:
        gt_root (str): Root directory of the ground truth segmentation maps.
        GT_types (list[str]): List of types of ground truth to load (e.g., ['SOD', 'SIC', 'FLOE']).
        downsample_factor (int): Factor by which to downsample the segmentation maps. Defaults to 10.
        pad_size (tuple[int], optional): Desired size to pad the segmentation maps to. Defaults to None.
        pad_val (float): Value to pad the segmentation maps with. Defaults to 255.
    """

    def __init__(self,
                 gt_root,
                 GT_type=['SOD'],
                 downsample_factor=10,
                 pad_size=None,
                 pad_val=255):
        self.gt_root = gt_root
        self.GT_types = GT_type
        self.downsample_factor = downsample_factor
        self.pad_size = pad_size
        self.pad_val = pad_val

    def transform(self, results):
        """Load multiple types of ground truth segmentation maps.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded segmentation map and meta information.
        """
        filename = results['img_path']
        gt_maps = []
        for GT_type in self.GT_types:
            gt_filename = os.path.basename(filename).replace('.nc', f'_{GT_type}.png')
            gt_filename = os.path.join(self.gt_root, gt_filename)
            gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
            shape = gt_seg_map.shape

            if self.downsample_factor != 1:
                gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0)
                gt_seg_map = torch.nn.functional.interpolate(
                    gt_seg_map, size=(shape[0] // self.downsample_factor, shape[1] // self.downsample_factor), mode='nearest')
                gt_seg_map = gt_seg_map.squeeze(0).squeeze(0)

                if self.pad_size is not None:
                    pad_height = max(0, self.pad_size[0] - gt_seg_map.shape[0])
                    pad_width = max(0, self.pad_size[1] - gt_seg_map.shape[1])
                    gt_seg_map = torch.nn.functional.pad(
                        gt_seg_map, (0, pad_width, 0, pad_height), mode='constant', value=self.pad_val)
                gt_seg_map = gt_seg_map.numpy()

            gt_maps.append(gt_seg_map)

        # Concatenate the segmentation maps along the channel dimension
        gt_seg_map = np.stack(gt_maps, axis=0)

        results['gt_seg_map'] = gt_seg_map
        results['seg_fields'].append('gt_seg_map')
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'gt_root={self.gt_root}, '
                    f'GT_types={self.GT_types}, '
                    f'downsample_factor={self.downsample_factor}, '
                    f'pad_size={self.pad_size}, '
                    f'pad_val={self.pad_val})')
        return repr_str
