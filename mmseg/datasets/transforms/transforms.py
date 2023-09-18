# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class ResizeToMultiple(BaseTransform):
    """Resize images & seg to multiple of divisor.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - pad_shape

    Args:
        size_divisor (int): images and gt seg maps need to resize to multiple
            of size_divisor. Default: 32.
        interpolation (str, optional): The interpolation mode of image resize.
            Default: None
    """

    def __init__(self, size_divisor=32, interpolation=None):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.
        img = results['img']
        img = mmcv.imresize_to_multiple(
            img,
            self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation
            if self.interpolation else 'bilinear')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]

        # Align segmentation map to multiple of size divisor.
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(
                gt_seg,
                self.size_divisor,
                scale_factor=1,
                interpolation='nearest')
            results[key] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size_divisor={self.size_divisor}, '
                     f'interpolation={self.interpolation})')
        return repr_str


@TRANSFORMS.register_module()
class Rerange(BaseTransform):
    """Rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, ' \
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate, degree = self.generate_degree()
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class RGB2Gray(BaseTransform):
    """Convert RGB image to grayscale image.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def transform(self, results: dict) -> dict:
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@TRANSFORMS.register_module()
class AdjustGamma(BaseTransform):
    """Using gamma correction to process the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def transform(self, results: dict) -> dict:
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@TRANSFORMS.register_module()
class SegRescale(BaseTransform):
    """Rescale semantic segmentation maps.

    Required Keys:

    - gt_seg_map

    Modified Keys:

    - gt_seg_map

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class RandomCutOut(BaseTransform):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): cutout probability.
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default: None.
    """

    def __init__(self,
                 prob,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None):

        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    @cache_randomness
    def do_cutout(self):
        return np.random.rand() < self.prob

    @cache_randomness
    def generate_patches(self, results):
        cutout = self.do_cutout()

        h, w, _ = results['img'].shape
        if cutout:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        else:
            n_holes = 0
        x1_lst = []
        y1_lst = []
        index_lst = []
        for _ in range(n_holes):
            x1_lst.append(np.random.randint(0, w))
            y1_lst.append(np.random.randint(0, h))
            index_lst.append(np.random.randint(0, len(self.candidates)))
        return cutout, n_holes, x1_lst, y1_lst, index_lst

    def transform(self, results: dict) -> dict:
        """Call function to drop some regions of image."""
        cutout, n_holes, x1_lst, y1_lst, index_lst = self.generate_patches(
            results)
        if cutout:
            h, w, c = results['img'].shape
            for i in range(n_holes):
                x1 = x1_lst[i]
                y1 = y1_lst[i]
                index = index_lst[i]
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in}, '
        repr_str += f'seg_fill_in={self.seg_fill_in})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotFlip(BaseTransform):
    """Rotate and flip the image & seg or just rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        rotate_prob (float): The probability of rotate image.
        flip_prob (float): The probability of rotate&flip image.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
    """

    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def random_rotate(self, results: dict) -> dict:
        angle = np.random.uniform(min(*self.degree), max(*self.degree))
        results['img'] = mmcv.imrotate(results['img'], angle=angle)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(results[key], angle=angle)
        return results

    def transform(self, results: dict) -> dict:
        """Call function to rotate or rotate & flip image, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated or rotated & flipped results.
        """
        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rotate(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_prob={self.rotate_prob}, ' \
                    f'flip_prob={self.flip_prob}, ' \
                    f'degree={self.degree})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):
    """Flip the image & bbox & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and gt_depth_map.
    There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string: the image will be
      ``direction``ly flipped with probability of ``prob`` .
      E.g., ``prob=0.5``, ``direction='horizontal'``,
      then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string: the image will
      be ``direction[i]``ly flipped with probability of
      ``prob/len(direction)``.
      E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
      then image will be horizontally flipped with probability of 0.25,
      vertically with probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string:
      given ``len(prob) == len(direction)``, the image will
      be ``direction[i]``ly flipped with probability of ``prob[i]``.
      E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
      'vertical']``, then image will be horizontally flipped with
      probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Added Keys:

    - flip
    - flip_direction
    - swap_seg_labels (optional)

    Args:
        prob (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip seg map
        for key in results.get('seg_fields', []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(
                    results[key], direction=results['flip_direction']).copy()
                results['swap_seg_labels'] = self.swap_seg_labels


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):
    """Resize images & seg & depth map.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Seg map, depth map and other relative annotations are
    then resized with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Modified Keys:

    - img
    - gt_seg_map
    - gt_depth_map

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[seg_key] = gt_seg


@TRANSFORMS.register_module()
class RandomMosaic(BaseTransform):
    """Mosaic augmentation. Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_seg_map
    - mix_results

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    - gt_seg_map

    Args:
        prob (float): mosaic probability.
        img_scale (Sequence[int]): Image size after mosaic pipeline of
            a single image. The size of the output image is four times
            that of a single image. The output image comprises 4 single images.
            Default: (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default: (0.5, 1.5).
        pad_val (int): Pad value. Default: 0.
        seg_pad_val (int): Pad value of segmentation map. Default: 255.
    """

    def __init__(self,
                 prob,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 pad_val=0,
                 seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @cache_randomness
    def do_mosaic(self):
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        mosaic = self.do_mosaic()
        if mosaic:
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
        return results

    def get_indices(self, dataset: MultiImageMixDataset) -> list:
        """Call function to collect indices.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indices.
        """

        indices = [random.randint(0, len(dataset)) for _ in range(3)]
        return indices

    @cache_randomness
    def generate_mosaic_center(self):
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        return center_x, center_y

    def _mosaic_transform_img(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        if len(results['img'].shape) == 3:
            c = results['img'].shape[2]
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), c),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        self.center_x, self.center_y = self.generate_mosaic_center()
        center_position = (self.center_x, self.center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape

        return results

    def _mosaic_transform_seg(self, results: dict) -> dict:
        """Mosaic transform function for label annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.seg_pad_val,
                dtype=results[key].dtype)

            # mosaic center x, y
            center_position = (self.center_x, self.center_y)

            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                h_i, w_i = gt_seg_i.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                gt_seg_i = mmcv.imresize(
                    gt_seg_i,
                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                    interpolation='nearest')

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, gt_seg_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = \
                    gt_seg_i[y1_c:y2_c, x1_c:x2_c]

            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float],
                        img_shape_wh: Sequence[int]) -> tuple:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateEdge(BaseTransform):
    """Generate Edge for CE2P approach.

    Edge will be used to calculate loss of
    `CE2P <https://arxiv.org/abs/1809.05996>`_.

    Modified from https://github.com/liutinglt/CE2P/blob/master/dataset/target_generation.py # noqa:E501

    Required Keys:

        - img_shape
        - gt_seg_map

    Added Keys:
        - gt_edge_map (np.ndarray, uint8): The edge annotation generated from the
            seg map by extracting border between different semantics.

    Args:
        edge_width (int): The width of edge. Default to 3.
        ignore_index (int): Index that will be ignored. Default to 255.
    """

    def __init__(self, edge_width: int = 3, ignore_index: int = 255) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:
        """Call function to generate edge from segmentation map.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with edge mask.
        """
        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)
        seg_map = results['gt_seg_map']

        # down
        edge_down = edge[1:h, :]
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1
        # left
        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1
        # up_left
        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1
        # up_right
        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.edge_width, self.edge_width))
        edge = cv2.dilate(edge, kernel)

        results['gt_edge_map'] = edge
        results['edge_width'] = self.edge_width

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'edge_width={self.edge_width}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeShortestEdge(BaseTransform):
    """Resize the image and mask while keeping the aspect ratio unchanged.

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py#L130 # noqa:E501
    Copyright (c) Facebook, Inc. and its affiliates.
    Licensed under the Apache-2.0 License

    This transform attempts to scale the shorter edge to the given
    `scale`, as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer
    edge does not exceed `max_size`.

    Required Keys:

    - img
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional))

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio


    Args:
        scale (Union[int, Tuple[int, int]]): The target short edge length.
            If it's tuple, will select the min value as the short edge length.
        max_size (int): The maximum allowed longest edge length.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]],
                 max_size: int) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size

        # Create a empty Resize object
        self.resize = TRANSFORMS.build({
            'type': 'Resize',
            'scale': 0,
            'keep_ratio': True
        })

    def _get_output_shape(self, img, short_edge_length) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(short_edge_length, int):
            size = short_edge_length * 1.0
        elif isinstance(short_edge_length, tuple):
            size = min(short_edge_length) * 1.0
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h *= scale
            new_w *= scale

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return (new_w, new_h)

    def transform(self, results: Dict) -> Dict:
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)


@TRANSFORMS.register_module()
class BioMedical3DRandomCrop(BaseTransform):
    """Crop the input patch for medical image & segmentation mask.

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
        N is the number of modalities, and data type is float32.
    - gt_seg_map (np.ndarray, optional): Biomedical semantic segmentation mask
        with shape (Z, Y, X).

    Modified Keys:

        - img
        - img_shape
        - gt_seg_map (optional)

    Args:
        crop_shape (Union[int, Tuple[int, int, int]]):  Expected size after
            cropping with the format of (z, y, x). If set to an integer,
            then cropping width and height are equal to this integer.
        keep_foreground (bool): If keep_foreground is True, it will sample a
            voxel of foreground classes randomly, and will take it as the
            center of the crop bounding-box. Default to True.
    """

    def __init__(self,
                 crop_shape: Union[int, Tuple[int, int, int]],
                 keep_foreground: bool = True):
        super().__init__()
        assert isinstance(crop_shape, int) or (
            isinstance(crop_shape, tuple) and len(crop_shape) == 3
        ), 'The expected crop_shape is an integer, or a tuple containing '
        'three integers'

        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape, crop_shape)
        assert crop_shape[0] > 0 and crop_shape[1] > 0 and crop_shape[2] > 0
        self.crop_shape = crop_shape
        self.keep_foreground = keep_foreground

    def random_sample_location(self, seg_map: np.ndarray) -> dict:
        """sample foreground voxel when keep_foreground is True.

        Args:
            seg_map (np.ndarray): gt seg map.

        Returns:
            dict: Coordinates of selected foreground voxel.
        """
        num_samples = 10000
        # at least 1% of the class voxels need to be selected,
        # otherwise it may be too sparse
        min_percent_coverage = 0.01
        class_locs = {}
        foreground_classes = []
        all_classes = np.unique(seg_map)
        for c in all_classes:
            if c == 0:
                # to avoid the segmentation mask full of background 0
                # and the class_locs is just void dictionary {} when it return
                # there add a void list for background 0.
                class_locs[c] = []
            else:
                all_locs = np.argwhere(seg_map == c)
                target_num_samples = min(num_samples, len(all_locs))
                target_num_samples = max(
                    target_num_samples,
                    int(np.ceil(len(all_locs) * min_percent_coverage)))

                selected = all_locs[np.random.choice(
                    len(all_locs), target_num_samples, replace=False)]
                class_locs[c] = selected
                foreground_classes.append(c)

        selected_voxel = None
        if len(foreground_classes) > 0:
            selected_class = np.random.choice(foreground_classes)
            voxels_of_that_class = class_locs[selected_class]
            selected_voxel = voxels_of_that_class[np.random.choice(
                len(voxels_of_that_class))]

        return selected_voxel

    def random_generate_crop_bbox(self, margin_z: int, margin_y: int,
                                  margin_x: int) -> tuple:
        """Randomly get a crop bounding box.

        Args:
            seg_map (np.ndarray): Ground truth segmentation map.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        offset_z = np.random.randint(0, margin_z + 1)
        offset_y = np.random.randint(0, margin_y + 1)
        offset_x = np.random.randint(0, margin_x + 1)
        crop_z1, crop_z2 = offset_z, offset_z + self.crop_shape[0]
        crop_y1, crop_y2 = offset_y, offset_y + self.crop_shape[1]
        crop_x1, crop_x2 = offset_x, offset_x + self.crop_shape[2]

        return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

    def generate_margin(self, results: dict) -> tuple:
        """Generate margin of crop bounding-box.

        If keep_foreground is True, it will sample a voxel of foreground
        classes randomly, and will take it as the center of the bounding-box,
        and return the margin between of the bounding-box and image.
        If keep_foreground is False, it will return the difference from crop
        shape and image shape.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: The margin for 3 dimensions of crop bounding-box and image.
        """

        seg_map = results['gt_seg_map']
        if self.keep_foreground:
            selected_voxel = self.random_sample_location(seg_map)
            if selected_voxel is None:
                # this only happens if some image does not contain
                # foreground voxels at all
                warnings.warn(f'case does not contain any foreground classes'
                              f': {results["img_path"]}')
                margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
                margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
                margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)
            else:
                margin_z = max(0, selected_voxel[0] - self.crop_shape[0] // 2)
                margin_y = max(0, selected_voxel[1] - self.crop_shape[1] // 2)
                margin_x = max(0, selected_voxel[2] - self.crop_shape[2] // 2)
                margin_z = max(
                    0, min(seg_map.shape[0] - self.crop_shape[0], margin_z))
                margin_y = max(
                    0, min(seg_map.shape[1] - self.crop_shape[1], margin_y))
                margin_x = max(
                    0, min(seg_map.shape[2] - self.crop_shape[2], margin_x))
        else:
            margin_z = max(seg_map.shape[0] - self.crop_shape[0], 0)
            margin_y = max(seg_map.shape[1] - self.crop_shape[1], 0)
            margin_x = max(seg_map.shape[2] - self.crop_shape[2], 0)

        return margin_z, margin_y, margin_x

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """
        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if len(img.shape) == 3:
            # crop seg map
            img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            # crop image
            assert len(img.shape) == 4
            img = img[:, crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        margin = self.generate_margin(results)
        crop_bbox = self.random_generate_crop_bbox(*margin)

        # crop the image
        img = results['img']
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape[1:]

        # crop semantic seg
        seg_map = results['gt_seg_map']
        results['gt_seg_map'] = self.crop(seg_map, crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_shape={self.crop_shape})'


@TRANSFORMS.register_module()
class BioMedicalGaussianNoise(BaseTransform):
    """Add random Gaussian noise to image.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/noise_transforms.py#L53  # noqa:E501

    Copyright (c) German Cancer Research Center (DKFZ)
    Licensed under the Apache License, Version 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
            N is the number of modalities, and data type is float32.

    Modified Keys:

    - img

    Args:
        prob (float): Probability to add Gaussian noise for
            each sample. Default to 0.1.
        mean (float): Mean or “centre” of the distribution. Default to 0.0.
        std (float): Standard deviation of distribution. Default to 0.1.
    """

    def __init__(self,
                 prob: float = 0.1,
                 mean: float = 0.0,
                 std: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0 and std >= 0.0
        self.prob = prob
        self.mean = mean
        self.std = std

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian noise to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        if np.random.rand() < self.prob:
            rand_std = np.random.uniform(0, self.std)
            noise = np.random.normal(
                self.mean, rand_std, size=results['img'].shape)
            # noise is float64 array, convert to the results['img'].dtype
            noise = noise.astype(results['img'].dtype)
            results['img'] = results['img'] + noise
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'mean={self.mean}, '
        repr_str += f'std={self.std})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalGaussianBlur(BaseTransform):
    """Add Gaussian blur with random sigma to image.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/noise_transforms.py#L81 # noqa:E501

    Copyright (c) German Cancer Research Center (DKFZ)
    Licensed under the Apache License, Version 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
            N is the number of modalities, and data type is float32.

    Modified Keys:

    - img

    Args:
        sigma_range (Tuple[float, float]|float): range to randomly
            select sigma value. Default to (0.5, 1.0).
        prob (float): Probability to apply Gaussian blur
            for each sample. Default to 0.2.
        prob_per_channel  (float): Probability to apply Gaussian blur
            for each channel (axis N of the image). Default to 0.5.
        different_sigma_per_channel (bool): whether to use different
            sigma for each channel (axis N of the image). Default to True.
        different_sigma_per_axis (bool): whether to use different
            sigma for axis Z, X and Y of the image. Default to True.
    """

    def __init__(self,
                 sigma_range: Tuple[float, float] = (0.5, 1.0),
                 prob: float = 0.2,
                 prob_per_channel: float = 0.5,
                 different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = True) -> None:
        super().__init__()
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= prob_per_channel <= 1.0
        assert isinstance(sigma_range, Sequence) and len(sigma_range) == 2
        self.sigma_range = sigma_range
        self.prob = prob
        self.prob_per_channel = prob_per_channel
        self.different_sigma_per_channel = different_sigma_per_channel
        self.different_sigma_per_axis = different_sigma_per_axis

    def _get_valid_sigma(self, value_range) -> Tuple[float, ...]:
        """Ensure the `value_range` to be either a single value or a sequence
        of two values. If the `value_range` is a sequence, generate a random
        value with `[value_range[0], value_range[1]]` based on uniform
        sampling.

        Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/augmentations/utils.py#L625 # noqa:E501

        Args:
            value_range (tuple|list|float|int): the input value range
        """
        if (isinstance(value_range, (list, tuple))):
            if (value_range[0] == value_range[1]):
                value = value_range[0]
            else:
                orig_type = type(value_range[0])
                value = np.random.uniform(value_range[0], value_range[1])
                value = orig_type(value)
        return value

    def _gaussian_blur(self, data_sample: np.ndarray) -> np.ndarray:
        """Random generate sigma and apply Gaussian Blur to the data
        Args:
            data_sample (np.ndarray): data sample with multiple modalities,
                the data shape is (N, Z, Y, X)
        """
        sigma = None
        for c in range(data_sample.shape[0]):
            if np.random.rand() < self.prob_per_channel:
                # if no `sigma` is generated, generate one
                # if `self.different_sigma_per_channel` is True,
                # re-generate random sigma for each channel
                if (sigma is None or self.different_sigma_per_channel):
                    if (not self.different_sigma_per_axis):
                        sigma = self._get_valid_sigma(self.sigma_range)
                    else:
                        sigma = [
                            self._get_valid_sigma(self.sigma_range)
                            for _ in data_sample.shape[1:]
                        ]
                # apply gaussian filter with `sigma`
                data_sample[c] = gaussian_filter(
                    data_sample[c], sigma, order=0)
        return data_sample

    def transform(self, results: Dict) -> Dict:
        """Call function to add random Gaussian blur to image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with random Gaussian noise.
        """
        if np.random.rand() < self.prob:
            results['img'] = self._gaussian_blur(results['img'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'prob_per_channel={self.prob_per_channel}, '
        repr_str += f'sigma_range={self.sigma_range}, '
        repr_str += 'different_sigma_per_channel=' \
                    f'{self.different_sigma_per_channel}, '
        repr_str += 'different_sigma_per_axis=' \
                    f'{self.different_sigma_per_axis})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedicalRandomGamma(BaseTransform):
    """Using random gamma correction to process the biomedical image.

    Modified from
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py#L132 # noqa:E501
    With licence: Apache 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
        N is the number of modalities, and data type is float32.

    Modified Keys:
    - img

    Args:
        prob (float): The probability to perform this transform. Default: 0.5.
        gamma_range (Tuple[float]): Range of gamma values. Default: (0.5, 2).
        invert_image (bool): Whether invert the image before applying gamma
            augmentation. Default: False.
        per_channel (bool): Whether perform the transform each channel
            individually. Default: False
        retain_stats (bool): Gamma transformation will alter the mean and std
            of the data in the patch. If retain_stats=True, the data will be
            transformed to match the mean and standard deviation before gamma
            augmentation. Default: False.
    """

    def __init__(self,
                 prob: float = 0.5,
                 gamma_range: Tuple[float] = (0.5, 2),
                 invert_image: bool = False,
                 per_channel: bool = False,
                 retain_stats: bool = False):
        assert 0 <= prob and prob <= 1
        assert isinstance(gamma_range, tuple) and len(gamma_range) == 2
        assert isinstance(invert_image, bool)
        assert isinstance(per_channel, bool)
        assert isinstance(retain_stats, bool)
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    @cache_randomness
    def _do_gamma(self):
        """Whether do adjust gamma for image."""
        return np.random.rand() < self.prob

    def _adjust_gamma(self, img: np.array):
        """Gamma adjustment for image.

        Args:
            img (np.array): Input image before gamma adjust.

        Returns:
            np.arrays: Image after gamma adjust.
        """

        if self.invert_image:
            img = -img

        def _do_adjust(img):
            if retain_stats_here:
                img_mean = img.mean()
                img_std = img.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(self.gamma_range[0], 1), self.gamma_range[1])
            img_min = img.min()
            img_range = img.max() - img_min  # range
            img = np.power(((img - img_min) / float(img_range + 1e-7)),
                           gamma) * img_range + img_min
            if retain_stats_here:
                img = img - img.mean()
                img = img / (img.std() + 1e-8) * img_std
                img = img + img_mean
            return img

        if not self.per_channel:
            retain_stats_here = self.retain_stats
            img = _do_adjust(img)
        else:
            for c in range(img.shape[0]):
                img[c] = _do_adjust(img[c])
        if self.invert_image:
            img = -img
        return img

    def transform(self, results: dict) -> dict:
        """Call function to perform random gamma correction
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with random gamma correction performed.
        """
        do_gamma = self._do_gamma()

        if do_gamma:
            results['img'] = self._adjust_gamma(results['img'])
        else:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'gamma_range={self.gamma_range},'
        repr_str += f'invert_image={self.invert_image},'
        repr_str += f'per_channel={self.per_channel},'
        repr_str += f'retain_stats={self.retain_stats}'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DPad(BaseTransform):
    """Pad the biomedical 3d image & biomedical 3d semantic segmentation maps.

    Required Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Modified Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Added Keys:

    - pad_shape (Tuple[int, int, int]): The padded shape.

    Args:
        pad_shape (Tuple[int, int, int]): Fixed padding size.
            Expected padding shape (Z, Y, X).
        pad_val (float): Padding value for biomedical image.
            The padding mode is set to "constant". The value
            to be filled in padding area. Default: 0.
        seg_pad_val (int): Padding value for biomedical 3d semantic
            segmentation maps. The padding mode is set to "constant".
            The value to be filled in padding area. Default: 0.
    """

    def __init__(self,
                 pad_shape: Tuple[int, int, int],
                 pad_val: float = 0.,
                 seg_pad_val: int = 0) -> None:

        # check pad_shape
        assert pad_shape is not None
        if not isinstance(pad_shape, tuple):
            assert len(pad_shape) == 3

        self.pad_shape = pad_shape
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.pad_shape``

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: The dict contains the padded image and shape
                information.
        """
        padded_img = self._to_pad(
            results['img'], pad_shape=self.pad_shape, pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape[1:]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to ``self.pad_shape`` if
        ``gt_seg_map`` is not None in results dict.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Update the padded gt seg map in dict.
        """
        if results.get('gt_seg_map', None) is not None:
            pad_gt_seg = self._to_pad(
                results['gt_seg_map'][None, ...],
                pad_shape=results['pad_shape'],
                pad_val=self.seg_pad_val)
            results['gt_seg_map'] = pad_gt_seg[1:]

    @staticmethod
    def _to_pad(img: np.ndarray,
                pad_shape: Tuple[int, int, int],
                pad_val: Union[int, float] = 0) -> np.ndarray:
        """Pad the given 3d image to a certain shape with specified padding
        value.

        Args:
            img (ndarray): Biomedical image with shape (N, Z, Y, X)
                to be padded. N is the number of modalities.
            pad_shape (Tuple[int,int,int]): Expected padding shape (Z, Y, X).
            pad_val (float, int): Values to be filled in padding areas
                and the padding_mode is set to 'constant'. Default: 0.

        Returns:
            ndarray: The padded image.
        """
        # compute pad width
        d = max(pad_shape[0] - img.shape[1], 0)
        pad_d = (d // 2, d - d // 2)
        h = max(pad_shape[1] - img.shape[2], 0)
        pad_h = (h // 2, h - h // 2)
        w = max(pad_shape[2] - img.shape[2], 0)
        pad_w = (w // 2, w - w // 2)

        pad_list = [(0, 0), pad_d, pad_h, pad_w]

        img = np.pad(img, pad_list, mode='constant', constant_values=pad_val)
        return img

    def transform(self, results: dict) -> dict:
        """Call function to pad images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'pad_shape={self.pad_shape}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'seg_pad_val={self.seg_pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class BioMedical3DRandomFlip(BaseTransform):
    """Flip biomedical 3D images and segmentations.

    Modified from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/spatial_transforms.py # noqa:E501

    Copyright 2021 Division of
    Medical Image Computing, German Cancer Research Center (DKFZ) and Applied
    Computer Vision Lab, Helmholtz Imaging Platform.
    Licensed under the Apache-2.0 License.

    Required Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Modified Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Added Keys:

    - do_flip
    - flip_axes

    Args:
        prob (float): Flipping probability.
        axes (Tuple[int, ...]): Flipping axes with order 'ZXY'.
        swap_label_pairs (Optional[List[Tuple[int, int]]]):
        The segmentation label pairs that are swapped when flipping.
    """

    def __init__(self,
                 prob: float,
                 axes: Tuple[int, ...],
                 swap_label_pairs: Optional[List[Tuple[int, int]]] = None):
        self.prob = prob
        self.axes = axes
        self.swap_label_pairs = swap_label_pairs
        assert prob >= 0 and prob <= 1
        if axes is not None:
            assert max(axes) <= 2

    @staticmethod
    def _flip(img, direction: Tuple[bool, bool, bool]) -> np.ndarray:
        if direction[0]:
            img[:, :] = img[:, ::-1]
        if direction[1]:
            img[:, :, :] = img[:, :, ::-1]
        if direction[2]:
            img[:, :, :, :] = img[:, :, :, ::-1]
        return img

    def _do_flip(self, img: np.ndarray) -> Tuple[bool, bool, bool]:
        """Call function to determine which axis to flip.

        Args:
            img (np.ndarry): Image or segmentation map array.
        Returns:
            tuple: Flip action, whether to flip on the z, x, and y axes.
        """
        flip_c, flip_x, flip_y = False, False, False
        if self.axes is not None:
            flip_c = 0 in self.axes and np.random.rand() < self.prob
            flip_x = 1 in self.axes and np.random.rand() < self.prob
            if len(img.shape) == 4:
                flip_y = 2 in self.axes and np.random.rand() < self.prob
        return flip_c, flip_x, flip_y

    def _swap_label(self, seg: np.ndarray) -> np.ndarray:
        out = seg.copy()
        for first, second in self.swap_label_pairs:
            first_area = (seg == first)
            second_area = (seg == second)
            out[first_area] = second
            out[second_area] = first
        return out

    def transform(self, results: Dict) -> Dict:
        """Call function to flip and swap pair labels.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Flipped results, 'do_flip', 'flip_axes' keys are added into
                result dict.
        """
        # get actual flipped axis
        if 'do_flip' not in results:
            results['do_flip'] = self._do_flip(results['img'])
        if 'flip_axes' not in results:
            results['flip_axes'] = self.axes
        # flip image
        results['img'] = self._flip(
            results['img'], direction=results['do_flip'])
        # flip seg
        if results['gt_seg_map'] is not None:
            if results['gt_seg_map'].shape != results['img'].shape:
                results['gt_seg_map'] = results['gt_seg_map'][None, :]
            results['gt_seg_map'] = self._flip(
                results['gt_seg_map'], direction=results['do_flip'])
            results['gt_seg_map'] = results['gt_seg_map'].squeeze()
            # swap label pairs
            if self.swap_label_pairs is not None:
                results['gt_seg_map'] = self._swap_label(results['gt_seg_map'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, axes={self.axes}, ' \
                    f'swap_label_pairs={self.swap_label_pairs})'
        return repr_str


@TRANSFORMS.register_module()
class Albu(BaseTransform):
    """Albumentation augmentation. Adds custom transformations from
    Albumentations library. Please, visit
    `https://albumentations.readthedocs.io` to get more information. An example
    of ``transforms`` is as followed:

    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
        update_pad_shape (bool): Whether to update padding shape according to \
            the output shape of the last transform
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None,
                 update_pad_shape: bool = False):
        if not ALBU_INSTALLED:
            raise ImportError(
                'albumentations is not installed, '
                'we suggest install albumentation by '
                '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
            )

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
        self.keymap = keymap
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {'img': 'image', 'gt_seg_map': 'mask'}
        else:
            self.keymap_to_albu = copy.deepcopy(keymap)
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> object:
        """Build a callable object from a dict containing albu arguments.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            Callable: A callable object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            if not ALBU_INSTALLED:
                raise ImportError(
                    'albumentations is not installed, '
                    'we suggest install albumentation by '
                    '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
                )
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a valid type or str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(t) for t in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, _ in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # Convert to RGB since Albumentations works with RGB images
        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)

        results = self.aug(**results)

        # Convert back to BGR
        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_RGB2BGR)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@TRANSFORMS.register_module()
class ConcatCDInput(BaseTransform):
    """Concat images for change detection.

    Required Keys:

    - img
    - img2

    Args:
        input_keys (tuple):  Input image keys for change detection.
            Default: ('img', 'img2').
    """

    def __init__(self, input_keys=('img', 'img2')):
        self.input_keys = input_keys

    def transform(self, results: dict) -> dict:
        img = []
        for input_key in self.input_keys:
            img.append(results.pop(input_key))
        results['img'] = np.concatenate(img, axis=2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_keys={self.input_keys}, '
        return repr_str


@TRANSFORMS.register_module()
class RandomDepthMix(BaseTransform):
    """This class implements the RandomDepthMix transform.

    Args:
        prob (float): Probability of applying the transformation.
            Defaults to 0.25.
        mix_scale_ratio (float): Ratio to scale the mix width.
            Defaults to 0.75.
    """

    def __init__(
        self,
        prob: float = 0.25,
        mix_scale_ratio: float = 0.75,
    ):
        super().__init__()

        self.prob = prob
        self.mix_scale_ratio = mix_scale_ratio

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        h, w = results['img_shape'][:2]
        left = int(w * random.random())
        width_ratio = self.mix_scale_ratio * random.random()
        width = int(max(1, (w - left) * width_ratio))

        img = results['img']
        depth_rescale_factor = results.get('depth_rescale_factor', 1)
        depth_map = results['gt_depth_map'] / depth_rescale_factor

        if img.ndim == 3:
            for c in range(img.shape[-1]):
                img[:, left:left + width, c] = depth_map[:, left:left + width]
        elif img.ndim == 2:
            img[:, left:left + width] = depth_map[:, left:left + width]
        else:
            raise ValueError(f'Invalid image shape ({img.shape})')

        results['img'] = img
        return results
