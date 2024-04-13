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
    """Load an image from an xarray dataset.

    Required Keys:
        - img_path

    Modified Keys:
        - img
        - img_shape
        - ori_shape
        - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (H, W) by default.

    Args:
        channels (list[str]): List of variable names to load as channels of the image.
        to_float32 (bool): Whether to convert the loaded image to a float32 numpy array.
            If set to False, the loaded image is a uint8 array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`. Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path not existent.
            Defaults to False.
    """

    def __init__(self,
                 channels,
                 data_root,
                 gt_root,
                 ann_file = None,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 color_type='color',
                 imdecode_backend='cv2',
                 nan=255,
                 downsample_factor=10,
                 with_seg=False,
                 GT_type='SOD',
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
        self.with_seg = with_seg
        self.GT_type = GT_type
        self.nc_files = self.list_nc_files(data_root, ann_file)
        # key represents full path of the image and value represents the np image loaded
        self.pre_loaded_image_dic = {}
        self.pre_loaded_seg_dic = {}
        ic('Starting to load all the images into memory...')
        for filename in tqdm(self.nc_files):
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            if self.with_seg:
                seg_maps = {}
                gt_filename = os.path.basename(filename)
                gt_filename = gt_filename.replace(
                    '.nc', f'_{self.GT_type}.png')
                gt_filename = os.path.join(gt_root, gt_filename)
                # Load the image in grayscale
                gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
                # Display the grayscale image
            img = xarr[self.channels].to_array().data
            # reorder from (2, H, W) to (H, W, 2)
            img = np.transpose(img, (1, 2, 0))
            mean = np.array(self.mean)
            std = np.array(self.std)
            img = (img-mean)/std
            shape = img.shape
            if self.downsample_factor != 1:
                # downsample by taking max over a 10x10 block
                # img = torch.from_numpy(np.expand_dims(img, 0))
                img = torch.from_numpy(img)
                img = img.unsqueeze(0).permute(0, 3, 1, 2)
                img = torch.nn.functional.interpolate(img,
                                                      size=(shape[0]//self.downsample_factor,
                                                            shape[1]//self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0, 2, 3, 1).squeeze(0)
                img = img.numpy()
                if self.with_seg:
                    gt_seg_map = torch.from_numpy(
                        gt_seg_map).unsqueeze(0).unsqueeze(0)
                    gt_seg_map = torch.nn.functional.interpolate(gt_seg_map,
                                                                 size=(shape[0]//self.downsample_factor,
                                                                       shape[1]//self.downsample_factor),
                                                                 mode='nearest')
                    gt_seg_map = gt_seg_map.squeeze(0).squeeze(0)
                    gt_seg_map = gt_seg_map.numpy()

            if to_float32:
                img = img.astype(np.float32)
            self.pre_loaded_image_dic[filename] = img
            if self.with_seg:
                self.pre_loaded_seg_dic[filename] = gt_seg_map
        ic('Finished loading all the images into memory...')

    def list_nc_files(self, folder_path, ann_file):
        nc_files = []
        if ann_file != None:
            with open(ann_file, "r") as file:
                # Read the lines of the file into a list
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
        img = np.nan_to_num(img, nan=255)
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
    """Load an image from an xarray dataset.

    Required Keys:
        - img_path

    Modified Keys:

        - seg_fields List : Contains segmentation keys
        - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (H, W) by default.

    Args:
        gt_root (str): Location of the folder containing the GT files
        imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path not existent.
            Defaults to False.
        GT_type (str): One of 'SOD'/'FLOE'/'SIC'
        downsample_factor (int): The downsampling factor the ground
    """

    def __init__(self,
                 gt_root,
                 downsample_factor=10,
                 GT_type='SOD'):
        self.gt_root = gt_root
        self.downsample_factor = downsample_factor
        self.GT_type = GT_type

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        gt_filename = os.path.basename(filename)
        gt_filename = gt_filename.replace('.nc', f'_{self.GT_type}.png')
        gt_filename = os.path.join(self.gt_root, gt_filename)
        # Load the image in grayscale
        gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
        shape = gt_seg_map.shape
        if self.downsample_factor != 1:
            gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0)
            gt_seg_map = torch.nn.functional.interpolate(gt_seg_map,
                                                         size=(shape[0]//self.downsample_factor,
                                                               shape[1]//self.downsample_factor),
                                                         mode='nearest')
            gt_seg_map = gt_seg_map.squeeze(0).squeeze(0)
            gt_seg_map = gt_seg_map.numpy()
        results['gt_seg_map'] = gt_seg_map
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
