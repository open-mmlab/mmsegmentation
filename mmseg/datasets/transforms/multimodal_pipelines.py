"""
Multi-Modal Data Loading Pipeline - MMSeg 1.x Version

Key changes from 0.x:
- Registry: PIPELINES -> TRANSFORMS
- __call__ -> transform (BaseTransform)
- Normalize moved to SegDataPreProcessor
- Collect + FormatBundle -> PackMultiModalSegInputs (SegDataSample)
"""
import copy

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import mmcv
except ImportError:
    mmcv = None

import os.path as osp


@TRANSFORMS.register_module()
class LoadMultiModalImageFromFile(BaseTransform):
    """Load multi-modal image - no zero padding version.

    Required Keys:
        - img_path (str)
        - modal_type (str)
        - actual_channels (int)

    Added Keys:
        - img (np.ndarray)
        - img_shape (tuple)
        - ori_shape (tuple)
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict) -> dict:
        filename = results['img_path']

        # Support TIFF format
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            if tifffile is not None:
                img = tifffile.imread(filename)
                if len(img.shape) == 3 and img.shape[0] < img.shape[2]:
                    img = np.transpose(img, (1, 2, 0))
            elif mmcv is not None:
                img_bytes = mmcv.FileClient.infer_client(
                    None, filename).get(filename)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend=self.imdecode_backend)
            else:
                raise ImportError(
                    'tifffile or mmcv required for TIFF loading')
        else:
            if mmcv is not None:
                img_bytes = mmcv.FileClient.infer_client(
                    None, filename).get(filename)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend=self.imdecode_backend)
            else:
                from PIL import Image
                img = np.array(Image.open(filename))

        if self.to_float32:
            img = img.astype(np.float32)

        actual_channels = results.get('actual_channels', 3)

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        current_channels = img.shape[2]

        if current_channels != actual_channels:
            actual_channels = current_channels
            results['actual_channels'] = actual_channels

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['ori_filename'] = osp.basename(filename)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class MultiModalNormalize(BaseTransform):
    """Multi-modal normalization - supports dynamic channel count.

    Required Keys:
        - img (np.ndarray)
        - modal_type (str)
        - actual_channels (int)

    Modified Keys:
        - img (np.ndarray)

    Added Keys:
        - img_norm_cfg (dict)
    """

    NORM_CONFIGS = {
        'rgb': {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
        },
        'sar': {
            'mean': [0.23651549, 0.31761484, 0.18514981, 0.26901252,
                     -14.57879175, -8.6098158, -14.2907338, -8.33534564],
            'std': [0.16280619, 0.20849304, 0.14008107, 0.19767644,
                    4.07141682, 3.94773216, 4.21006244, 4.05494136],
        },
        'multispectral': {
            'mean': [1353., 1329., 1627., 1935., 2268., 2723., 3154.,
                     3541., 3652., 3416., 1112., 2619., 2060.],
            'std': [1108., 942., 976., 1164., 1196., 1351., 1500.,
                    1605., 1611., 1288., 770., 1325., 1186.],
        },
        'GF': {
            'mean': [432.02181, 315.92948, 246.468659,
                     310.61462, 360.267789],
            'std': [97.73313111900238, 85.78646917160748,
                    95.78015824658593,
                    124.84677067613467, 251.73965882246978],
        },
        # ---- Sen1Floods11: S1Hand (2-band VV/VH SAR, dB) ----
        # Defaults computed from the Sen1Floods11 train split with
        # the nodata pixels (-1) masked out. Re-run
        # tools/compute_sen1floods11_stats.py on your own split to
        # refresh these numbers if needed.
        's1': {
            'mean': [-10.483032437170175, -17.362463068117055],
            'std':  [4.178513068178825, 4.863193681650141],
        },
        # ---- Sen1Floods11: S2Hand (13-band Sentinel-2 MSI, TOA*10000) ----
        's2': {
            'mean':  [1483.1443242989628, 1234.2590152666212, 1204.8650733526135, 1034.8886830055903, 1305.9293935728826, 2257.5830489084565, 2723.9229150471333, 2515.8173451011917, 2957.957558849772, 447.04848745605636, 57.01156794081842, 1893.1678433033005, 1040.2051810757566],
            'std':  [314.8831761693824, 341.1134613706103, 367.08026898219816, 524.9257541131836, 446.1063090649358, 697.2651115379473, 897.1816646156293, 868.071173572936, 1031.6451147331716, 287.1789646791449, 130.50110493994782, 830.2058546614454, 610.0878654966048],
        },
        # ---- Sen2GF3Floods: fused Sentinel-2 RGBN (4ch) + GF-3 HH/HV (2ch) ----
        # Placeholder defaults. Run tools/setup_sen2gf3floods.py to
        # compute actual stats and paste them here.
        'sen2gf3': {
            'mean': [128.0, 128.0, 128.0, 128.0, 0.0, 0.0],
            'std':  [50.0, 50.0, 50.0, 50.0, 1.0, 1.0],
        },
    }

    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        img = results['img']
        modal_type = results.get('modal_type', 'rgb')
        actual_channels = results['actual_channels']

        if modal_type in self.NORM_CONFIGS:
            config = self.NORM_CONFIGS[modal_type]
            mean = np.array(
                config['mean'][:actual_channels], dtype=np.float32)
            std = np.array(
                config['std'][:actual_channels], dtype=np.float32)
        else:
            mean = np.array([128.0] * actual_channels, dtype=np.float32)
            std = np.array([50.0] * actual_channels, dtype=np.float32)

        mean_b = mean.reshape(1, 1, -1)
        std_b = std.reshape(1, 1, -1)

        # Sen1Floods11 S1Hand (and many other SAR/MSI products) encode
        # nodata pixels as NaN / ±Inf inside the TIFF. Left alone, the
        # NaN propagates through `(img - mean) / std` and poisons every
        # downstream feature map, so both the CE loss and the MoE
        # balance loss become NaN from the very first training step.
        # Replace non-finite pixels with the per-channel mean so they
        # normalize to 0 (a neutral value the network will learn to
        # ignore alongside the 255-ignored label pixels).
        if not np.all(np.isfinite(img)):
            img = np.where(
                np.isfinite(img),
                img,
                np.broadcast_to(mean_b, img.shape),
            ).astype(np.float32)

        img = (img - mean_b) / std_b

        # Final safety net: some SAR products use a finite sentinel
        # (e.g. -9999) instead of NaN for nodata, which would otherwise
        # survive normalization as a huge outlier. Clipping to ±10σ is
        # well outside any legitimate value for the modalities
        # configured above, so real data is untouched.
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(img, -10.0, 10.0, out=img)

        results['img'] = img
        results['img_norm_cfg'] = dict(
            mean=mean.tolist(),
            std=std.tolist(),
            to_rgb=self.to_rgb,
        )

        return results


@TRANSFORMS.register_module()
class GenerateBoundary(BaseTransform):
    """Generate boundary map from segmentation label.

    Required Keys:
        - gt_seg_map (np.ndarray)

    Added Keys:
        - gt_boundary_map (np.ndarray)
    """

    def __init__(self, thickness=3, ignore_index=255):
        self.thickness = thickness
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' not in results:
            return results

        seg = results['gt_seg_map']

        if len(seg.shape) == 3:
            seg = seg.squeeze(-1)

        boundary = self._generate_boundary(seg)

        results['gt_boundary_map'] = boundary
        if 'seg_fields' in results:
            results['seg_fields'].append('gt_boundary_map')

        return results

    def _generate_boundary(self, seg_mask):
        import cv2

        boundary = np.zeros_like(seg_mask, dtype=np.uint8)

        unique_labels = np.unique(seg_mask)
        unique_labels = unique_labels[unique_labels != self.ignore_index]

        kernel = np.ones((self.thickness, self.thickness), np.uint8)

        for label in unique_labels:
            class_mask = (seg_mask == label).astype(np.uint8)
            dilated = cv2.dilate(class_mask, kernel, iterations=1)
            eroded = cv2.erode(class_mask, kernel, iterations=1)
            class_boundary = dilated - eroded
            boundary = np.maximum(boundary, class_boundary)

        boundary[seg_mask == self.ignore_index] = self.ignore_index

        return boundary


@TRANSFORMS.register_module()
class MultiModalPad(BaseTransform):
    """Pad images with arbitrary channel counts using numpy.

    mmcv.transforms.Pad uses cv2.copyMakeBorder which only supports up to
    4 channels. This transform uses numpy padding instead, so it works
    with SAR (8ch), GF (5ch), etc.

    Required Keys:
        - img (np.ndarray)

    Modified Keys:
        - img (np.ndarray)
        - img_shape (tuple)

    Added Keys:
        - pad_shape (tuple)
        - padding_size (tuple)

    Args:
        size (tuple): Target (H, W).
        pad_val (float): Padding value for images. Default: 0.
        seg_pad_val (float): Padding value for seg maps. Default: 255.
    """

    def __init__(self, size, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = img.shape[:2]
        target_h, target_w = self.size

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            if len(img.shape) == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            else:
                pad_width = ((0, pad_h), (0, pad_w))

            img = np.pad(img, pad_width, mode='constant',
                         constant_values=self.pad_val)
            results['img'] = img

            # Pad seg maps
            for key in results.get('seg_fields', []):
                if key in results:
                    seg = results[key]
                    if len(seg.shape) == 3:
                        seg_pad = ((0, pad_h), (0, pad_w), (0, 0))
                    else:
                        seg_pad = ((0, pad_h), (0, pad_w))
                    results[key] = np.pad(
                        seg, seg_pad, mode='constant',
                        constant_values=self.seg_pad_val)

            if 'gt_seg_map' in results and 'gt_seg_map' not in results.get(
                    'seg_fields', []):
                seg = results['gt_seg_map']
                if len(seg.shape) == 3:
                    seg_pad = ((0, pad_h), (0, pad_w), (0, 0))
                else:
                    seg_pad = ((0, pad_h), (0, pad_w))
                results['gt_seg_map'] = np.pad(
                    seg, seg_pad, mode='constant',
                    constant_values=self.seg_pad_val)

        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]
        results['padding_size'] = (0, pad_w, 0, pad_h)

        return results


@TRANSFORMS.register_module()
class PackMultiModalSegInputs(BaseTransform):
    """Pack multi-modal data into SegDataSample format.

    This replaces CollectMultiModalData + MultiModalFormatBundle from 0.x.

    Required Keys:
        - img (np.ndarray)

    Optional Keys:
        - gt_seg_map (np.ndarray)
        - gt_boundary_map (np.ndarray)
        - dataset_name (str)
        - modal_type (str)
        - actual_channels (int)

    Added Keys:
        - inputs (torch.Tensor)
        - data_samples (SegDataSample)
    """

    def __init__(self,
                 meta_keys=('img_path', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor',
                            'flip', 'flip_direction',
                            'modal_type', 'actual_channels',
                            'dataset_name', 'img_norm_cfg',
                            'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            if len(gt_seg_map.shape) == 2:
                data = to_tensor(gt_seg_map[None, ...].astype(np.int64))
            else:
                data = to_tensor(gt_seg_map.astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_boundary_map' in results:
            gt_boundary_data = dict(
                data=to_tensor(
                    results['gt_boundary_map'][None, ...].astype(np.int64)))
            data_sample.set_data(
                dict(gt_boundary_map=PixelData(**gt_boundary_data)))

        # Set meta info
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
