# Copyright (c) OpenMMLab. All rights reserved.
import threading
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseModel
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

try:
    from osgeo import gdal
except ImportError:
    gdal = None

from mmseg.registry import MODELS
from .utils import _preprare_data


class RSImage:
    """Remote sensing image class.

    Args:
        img (str or gdal.Dataset): Image file path or gdal.Dataset.
    """

    def __init__(self, image):
        self.dataset = gdal.Open(image, gdal.GA_ReadOnly) if isinstance(
            image, str) else image
        assert isinstance(self.dataset, gdal.Dataset), \
            f'{image} is not a image'
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.channel = self.dataset.RasterCount
        self.trans = self.dataset.GetGeoTransform()
        self.proj = self.dataset.GetProjection()
        self.band_list = []
        self.band_list.extend(
            self.dataset.GetRasterBand(c + 1) for c in range(self.channel))
        self.grids = []

    def read(self, grid: Optional[List] = None) -> np.ndarray:
        """Read image data. If grid is None, read the whole image.

        Args:
            grid (Optional[List], optional): Grid to read. Defaults to None.
        Returns:
            np.ndarray: Image data.
        """
        if grid is None:
            return np.einsum('ijk->jki', self.dataset.ReadAsArray())
        assert len(
            grid) >= 4, 'grid must be a list containing at least 4 elements'
        data = self.dataset.ReadAsArray(*grid[:4])
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return np.einsum('ijk->jki', data)

    def write(self, data: Optional[np.ndarray], grid: Optional[List] = None):
        """Write image data.

        Args:
            grid (Optional[List], optional): Grid to write. Defaults to None.
            data (Optional[np.ndarray], optional): Data to write.
                Defaults to None.

        Raises:
            ValueError: Either grid or data must be provided.
        """
        if grid is not None:
            assert len(grid) == 8, 'grid must be a list of 8 elements'
            for band in self.band_list:
                band.WriteArray(
                    data[grid[5]:grid[5] + grid[7], grid[4]:grid[4] + grid[6]],
                    grid[0] + grid[4], grid[1] + grid[5])
        elif data is not None:
            for i in range(self.channel):
                self.band_list[i].WriteArray(data[..., i])
        else:
            raise ValueError('Either grid or data must be provided.')

    def create_seg_map(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = 'output_label.tif'
        driver = gdal.GetDriverByName('GTiff')
        seg_map = driver.Create(output_path, self.width, self.height, 1,
                                gdal.GDT_Byte)
        seg_map.SetGeoTransform(self.trans)
        seg_map.SetProjection(self.proj)
        seg_map_img = RSImage(seg_map)
        seg_map_img.path = output_path
        return seg_map_img

    def create_grids(self,
                     window_size: Tuple[int, int],
                     stride: Tuple[int, int] = (0, 0)):
        """Create grids for image inference.

        Args:
            window_size (Tuple[int, int]): the size of the sliding window.
            stride (Tuple[int, int], optional): the stride of the sliding
                window. Defaults to (0, 0).

        Raises:
            AssertionError: window_size must be a tuple of 2 elements.
            AssertionError: stride must be a tuple of 2 elements.
        """
        assert len(
            window_size) == 2, 'window_size must be a tuple of 2 elements'
        assert len(stride) == 2, 'stride must be a tuple of 2 elements'
        win_w, win_h = window_size
        stride_x, stride_y = stride

        stride_x = win_w if stride_x == 0 else stride_x
        stride_y = win_h if stride_y == 0 else stride_y

        x_half_overlap = (win_w - stride_x + 1) // 2
        y_half_overlap = (win_h - stride_y + 1) // 2

        for y in range(0, self.height, stride_y):
            y_end = y + win_h >= self.height
            y_offset = self.height - win_h if y_end else y
            y_size = win_h
            y_crop_off = 0 if y_offset == 0 else y_half_overlap
            y_crop_size = y_size if y_end else win_h - y_crop_off

            for x in range(0, self.width, stride_x):
                x_end = x + win_w >= self.width
                x_offset = self.width - win_w if x_end else x
                x_size = win_w
                x_crop_off = 0 if x_offset == 0 else x_half_overlap
                x_crop_size = x_size if x_end else win_w - x_crop_off

                self.grids.append([
                    x_offset, y_offset, x_size, y_size, x_crop_off, y_crop_off,
                    x_crop_size, y_crop_size
                ])


class RSInferencer:
    """Remote sensing inference class.

    Args:
        model (BaseModel): The loaded model.
        batch_size (int, optional): Batch size. Defaults to 1.
        thread (int, optional): Number of threads. Defaults to 1.
    """

    def __init__(self, model: BaseModel, batch_size: int = 1, thread: int = 1):
        self.model = model
        self.batch_size = batch_size
        self.END_FLAG = object()
        self.read_buffer = Queue(self.batch_size)
        self.write_buffer = Queue(self.batch_size)
        self.thread = thread

    @classmethod
    def from_config_path(cls,
                         config_path: str,
                         checkpoint_path: str,
                         batch_size: int = 1,
                         thread: int = 1,
                         device: Optional[str] = 'cpu'):
        """Initialize a segmentor from config file.

        Args:
            config_path (str): Config file path.
            checkpoint_path (str): Checkpoint path.
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        init_default_scope('mmseg')
        cfg = Config.fromfile(config_path)
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.to(device)
        model.eval()
        return cls(model, batch_size, thread)

    @classmethod
    def from_model(cls,
                   model: BaseModel,
                   checkpoint_path: Optional[str] = None,
                   batch_size: int = 1,
                   thread: int = 1,
                   device: Optional[str] = 'cpu'):
        """Initialize a segmentor from model.

        Args:
            model (BaseModel): The loaded model.
            checkpoint_path (Optional[str]): Checkpoint path.
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        if checkpoint_path is not None:
            load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.to(device)
        return cls(model, batch_size, thread)

    def read(self,
             image: RSImage,
             window_size: Tuple[int, int],
             strides: Tuple[int, int] = (0, 0)):
        """Load image data to read buffer.

        Args:
            image (RSImage): The image to read.
            window_size (Tuple[int, int]): The size of the sliding window.
            strides (Tuple[int, int], optional): The stride of the sliding
                window. Defaults to (0, 0).
        """
        image.create_grids(window_size, strides)
        for grid in image.grids:
            self.read_buffer.put([grid, image.read(grid=grid)])
        self.read_buffer.put(self.END_FLAG)

    def inference(self):
        """Inference image data from read buffer and put the result to write
        buffer."""
        while True:
            item = self.read_buffer.get()
            if item == self.END_FLAG:
                self.read_buffer.put(self.END_FLAG)
                self.write_buffer.put(item)
                break
            data, _ = _preprare_data(item[1], self.model)
            with torch.no_grad():
                result = self.model.test_step(data)
            item[1] = result[0].pred_sem_seg.cpu().data.numpy()[0]
            self.write_buffer.put(item)
            self.read_buffer.task_done()

    def write(self, image: RSImage, output_path: Optional[str] = None):
        """Write image data from write buffer.

        Args:
            image (RSImage): The image to write.
            output_path (Optional[str], optional): The path to save the
                segmentation map. Defaults to None.
        """
        seg_map = image.create_seg_map(output_path)
        while True:
            item = self.write_buffer.get()
            if item == self.END_FLAG:
                break
            seg_map.write(data=item[1], grid=item[0])
            self.write_buffer.task_done()

    def run(self,
            image: RSImage,
            window_size: Tuple[int, int],
            strides: Tuple[int, int] = (0, 0),
            output_path: Optional[str] = None):
        """Run inference with multi-threading.

        Args:
            image (RSImage): The image to inference.
            window_size (Tuple[int, int]): The size of the sliding window.
            strides (Tuple[int, int], optional): The stride of the sliding
                window. Defaults to (0, 0).
            output_path (Optional[str], optional): The path to save the
                segmentation map. Defaults to None.
        """
        read_thread = threading.Thread(
            target=self.read, args=(image, window_size, strides))
        read_thread.start()
        inference_threads = []
        for _ in range(self.thread):
            inference_thread = threading.Thread(target=self.inference)
            inference_thread.start()
            inference_threads.append(inference_thread)
        write_thread = threading.Thread(
            target=self.write, args=(image, output_path))
        write_thread.start()
        read_thread.join()
        for inference_thread in inference_threads:
            inference_thread.join()
        write_thread.join()
