import xarray as xr
from mmcv.image import imread
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from icecream import ic
import os

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
        self.downsample_factor = downsample_factor
        self.with_seg = with_seg
        self.GT_type = GT_type
        self.nc_files = self.list_nc_files(data_root)
        # key represents full path of the image and value represents the np image loaded
        self.pre_loaded_image_dic = {}
        self.pre_loaded_seg_dic = {}
        ic('Starting to load all the images into memory...')
        for filename in tqdm(self.nc_files):
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            if self.with_seg:
                seg_maps = {}
                xarr = convert_polygon_icechart(xarr)
                # SIC, SOD FLOE
                SIC = xarr['SIC'].values
                SOD = xarr['SOD'].values
                FLOE = xarr['FLOE'].values
                # Convert nan to num
                SIC = np.nan_to_num(SIC, nan=255).astype(np.uint8)
                SOD = np.nan_to_num(SOD, nan=255).astype(np.uint8)
                FLOE = np.nan_to_num(FLOE, nan=255).astype(np.uint8)
                seg_maps['SIC'] = SIC
                seg_maps['SOD'] = SOD
                seg_maps['FLOE'] = FLOE
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
                    gt_seg_map = seg_maps[self.GT_type]
                    gt_seg_map = gt_seg_map.unsqueeze(0).permute(0, 3, 1, 2)
                    gt_seg_map = torch.nn.functional.interpolate(gt_seg_map,
                                                                 size=(shape[0]//self.downsample_factor,
                                                                       shape[1]//self.downsample_factor),
                                                                 mode='nearest')
                    gt_seg_map = gt_seg_map.permute(0, 2, 3, 1).squeeze(0)
                    gt_seg_map = gt_seg_map.numpy()

            if to_float32:
                img = img.astype(np.float32)
            self.pre_loaded_image_dic[filename] = img
            if self.with_seg:
                self.pre_loaded_seg_dic[filename] = gt_seg_map
        ic('Finished loading all the images into memory...')

    def list_nc_files(self, folder_path):
        nc_files = []
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

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        if self.with_seg:
            results['gt_seg_map'] = self.pre_loaded_seg_dic[filename]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'channels={self.channels}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'ignore_empty={self.ignore_empty})')
        return repr_str