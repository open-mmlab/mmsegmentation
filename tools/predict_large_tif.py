"""
Large TIF Inference for Multi-Modal Segmentor.

Reads a large GeoTIFF (e.g. 20803x36986), tiles it into patches,
runs inference with the multi-modal model, and stitches results
back into a full-size GeoTIFF. Flood pixels are colored red (255,0,0),
non-flood pixels are black (0,0,0).

Usage:
    python tools/predict_large_tif.py \
        configs/floodnet/finetune_single_modal.py \
        work_dirs/generalization/LY-train-station/best_mIoU_epoch_30.pth \
        --input data/luoyuan/result.tif \
        --output data/luoyuan/prediction.tif \
        --tile-size 512 \
        --overlap 64 \
        --modal rgb \
        --bands 0 1 2 \
        --batch-size 4
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Large TIF inference with tile stitching')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--input', required=True, help='input TIF file')
    parser.add_argument('--output', default=None,
                        help='output TIF file (default: input_pred.tif)')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='tile size for inference (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='overlap between tiles (default: 64)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size for inference (default: 4)')
    parser.add_argument('--modal', default='rgb',
                        help='modality type for model (default: rgb)')
    parser.add_argument('--bands', type=int, nargs='+', default=[0, 1, 2],
                        help='band indices to read (0-indexed, default: 0 1 2)')
    parser.add_argument('--flood-class', type=int, default=1,
                        help='class index for flood (default: 1)')
    parser.add_argument('--device', default='cuda:0',
                        help='device for inference (default: cuda:0)')
    return parser.parse_args()


def build_model(cfg, checkpoint_path, device):
    """Build model and load checkpoint."""
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model = revert_sync_batchnorm(model)
    model.eval()
    return model


def read_tif_info(tif_path):
    """Read TIF metadata."""
    if HAS_RASTERIO:
        with rasterio.open(tif_path) as src:
            return {
                'height': src.height,
                'width': src.width,
                'bands': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'profile': src.profile.copy(),
            }
    elif HAS_GDAL:
        ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        return {
            'height': ds.RasterYSize,
            'width': ds.RasterXSize,
            'bands': ds.RasterCount,
            'geo_transform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
        }
    else:
        raise ImportError('Neither rasterio nor GDAL available. '
                          'Install with: pip install rasterio')


def read_tile_rasterio(tif_path, x, y, w, h, band_indices):
    """Read a tile from TIF using rasterio."""
    with rasterio.open(tif_path) as src:
        window = rasterio.windows.Window(x, y, w, h)
        # rasterio bands are 1-indexed
        bands = [b + 1 for b in band_indices]
        data = src.read(bands, window=window)  # (C, H, W)
        return data.astype(np.float32)


def read_tile_gdal(tif_path, x, y, w, h, band_indices):
    """Read a tile from TIF using GDAL."""
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    data = []
    for bi in band_indices:
        band = ds.GetRasterBand(bi + 1)  # GDAL is 1-indexed
        arr = band.ReadAsArray(x, y, w, h)
        data.append(arr.astype(np.float32))
    return np.stack(data, axis=0)  # (C, H, W)


def generate_tiles(img_h, img_w, tile_size, overlap):
    """Generate tile coordinates with overlap."""
    stride = tile_size - overlap
    tiles = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Clamp to image boundary
            x1 = min(x, img_w - tile_size)
            y1 = min(y, img_h - tile_size)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            w = min(tile_size, img_w - x1)
            h = min(tile_size, img_h - y1)
            tiles.append((x1, y1, w, h))

    # Deduplicate (edge tiles may repeat)
    seen = set()
    unique_tiles = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique_tiles.append(t)

    return unique_tiles


def normalize_tile(tile_data):
    """Simple normalization: scale to [0, 1] range per channel."""
    # tile_data: (C, H, W)
    for c in range(tile_data.shape[0]):
        band = tile_data[c]
        bmin, bmax = band.min(), band.max()
        if bmax - bmin > 1e-6:
            tile_data[c] = (band - bmin) / (bmax - bmin)
        else:
            tile_data[c] = 0.0
    return tile_data


def predict_batch(model, batch_imgs, modal, device):
    """Run inference on a batch of tile images.

    Args:
        model: segmentor model
        batch_imgs: list of (C, H, W) numpy arrays
        modal: modality string
        device: torch device

    Returns:
        list of (H, W) numpy prediction masks
    """
    imgs = []
    data_samples = []

    for img_np in batch_imgs:
        img_tensor = torch.from_numpy(img_np).float().to(device)
        imgs.append(img_tensor)

        ds = SegDataSample()
        h, w = img_np.shape[1], img_np.shape[2]
        ds.set_metainfo(dict(
            img_shape=(h, w),
            ori_shape=(h, w),
            pad_shape=(h, w),
            scale_factor=(1.0, 1.0),
            flip=False,
            flip_direction=None,
            modal_type=modal,
            actual_channels=img_np.shape[0],
            dataset_name=modal,
            reduce_zero_label=False,
        ))
        data_samples.append(ds)

    with torch.no_grad():
        results = model(imgs, data_samples, mode='predict')

    preds = []
    for r in results:
        pred = r.pred_sem_seg.data.cpu().numpy()[0]  # (H, W)
        preds.append(pred)

    return preds


def main():
    args = parse_args()

    if not HAS_RASTERIO and not HAS_GDAL:
        raise ImportError('Install rasterio or GDAL: pip install rasterio')

    read_tile = read_tile_rasterio if HAS_RASTERIO else read_tile_gdal

    # Output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f'{base}_pred{ext}'

    # Load config and model
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # Override test_cfg to use 'whole' mode (we handle tiling ourselves)
    cfg.model.test_cfg = dict(mode='whole')

    print('=' * 60)
    print('Large TIF Inference')
    print(f'Input:      {args.input}')
    print(f'Output:     {args.output}')
    print(f'Tile size:  {args.tile_size}')
    print(f'Overlap:    {args.overlap}')
    print(f'Modality:   {args.modal}')
    print(f'Bands:      {args.bands}')
    print(f'Batch size: {args.batch_size}')
    print('=' * 60)

    # Read TIF info
    info = read_tif_info(args.input)
    img_h, img_w = info['height'], info['width']
    print(f'\nImage size: {img_w} x {img_h} (W x H)')
    print(f'Bands:      {info["bands"]}')

    # Build model
    print('\nBuilding model...')
    model = build_model(cfg, args.checkpoint, args.device)
    print('Model loaded.')

    # Generate tiles
    tiles = generate_tiles(img_h, img_w, args.tile_size, args.overlap)
    num_tiles = len(tiles)
    num_batches = math.ceil(num_tiles / args.batch_size)
    print(f'\nTotal tiles: {num_tiles}')
    print(f'Batches:     {num_batches}')

    # Allocate output: prediction mask + count for overlap averaging
    pred_sum = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.float32)

    # Inference
    print('\nStarting inference...')
    t_start = time.time()

    for batch_idx in range(num_batches):
        start_i = batch_idx * args.batch_size
        end_i = min(start_i + args.batch_size, num_tiles)
        batch_tiles = tiles[start_i:end_i]

        # Read tiles
        batch_imgs = []
        for (x, y, w, h) in batch_tiles:
            tile_data = read_tile(args.input, x, y, w, h, args.bands)
            # Handle edge tiles smaller than tile_size: pad
            if h < args.tile_size or w < args.tile_size:
                padded = np.zeros(
                    (len(args.bands), args.tile_size, args.tile_size),
                    dtype=np.float32)
                padded[:, :h, :w] = tile_data
                tile_data = padded
            tile_data = normalize_tile(tile_data)
            batch_imgs.append(tile_data)

        # Predict
        preds = predict_batch(model, batch_imgs, args.modal, args.device)

        # Stitch predictions
        for (x, y, w, h), pred in zip(batch_tiles, preds):
            pred_sum[y:y+h, x:x+w] += pred[:h, :w].astype(np.float32)
            count_map[y:y+h, x:x+w] += 1.0

        # Progress
        done = end_i
        elapsed = time.time() - t_start
        eta = elapsed / done * (num_tiles - done) if done > 0 else 0
        print(f'\r  [{done}/{num_tiles}] '
              f'{done/num_tiles*100:.1f}% | '
              f'Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s',
              end='', flush=True)

    print()

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1.0)
    pred_avg = pred_sum / count_map
    pred_mask = (pred_avg >= 0.5).astype(np.uint8)  # threshold

    # Count statistics
    flood_pixels = (pred_mask == 1).sum()
    total_pixels = pred_mask.size
    print(f'\nFlood pixels:     {flood_pixels:,} '
          f'({flood_pixels/total_pixels*100:.2f}%)')
    print(f'Non-flood pixels: {total_pixels - flood_pixels:,}')

    # Write output TIF (3-band RGB: flood=red, non-flood=black)
    print(f'\nWriting output: {args.output}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if HAS_RASTERIO:
        profile = info['profile'].copy()
        profile.update(
            count=3,
            dtype='uint8',
            compress='lzw',
        )
        with rasterio.open(args.output, 'w', **profile) as dst:
            # Red channel: 255 for flood
            red = (pred_mask * 255).astype(np.uint8)
            # Green and Blue: 0
            green = np.zeros_like(red)
            blue = np.zeros_like(red)
            dst.write(red, 1)
            dst.write(green, 2)
            dst.write(blue, 3)
    elif HAS_GDAL:
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            args.output, img_w, img_h, 3, gdal.GDT_Byte,
            options=['COMPRESS=LZW'])
        out_ds.SetGeoTransform(info['geo_transform'])
        out_ds.SetProjection(info['projection'])
        red = (pred_mask * 255).astype(np.uint8)
        out_ds.GetRasterBand(1).WriteArray(red)
        out_ds.GetRasterBand(2).WriteArray(np.zeros_like(red))
        out_ds.GetRasterBand(3).WriteArray(np.zeros_like(red))
        out_ds.FlushCache()
        out_ds = None

    elapsed_total = time.time() - t_start
    print(f'\nDone! Total time: {elapsed_total:.1f}s')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
