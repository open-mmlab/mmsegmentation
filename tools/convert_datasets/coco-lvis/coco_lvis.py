import argparse
import pickle
import os
import os.path as osp
import shutil
import mmcv
import numpy as np
import imageio as io
from lvis.lvis import LVIS
from pycocotools.coco import COCO
from mmseg.utils import get_root_logger
logger = get_root_logger()


def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Convert coco lvis annotations to TrainIds')
    parser.add_argument('coco2017_path', help='coco2017 data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    """main process function"""
    args = parse_args()
    root = args.coco2017_path

    files = os.listdir(root)
    assert 'train2017' in files, 'missing train images'
    # assert 'test2017' in files, 'missing test images'
    assert 'val2017' in files, 'missing val images'
    assert 'stuff_annotations_trainval2017' in files, \
        'missing stuff annotations of coco2017 trainval'
    assert 'lvis_v1_train.json' in files and 'lvis_v1_val.json' in files, \
        'missing lvis annotations'

    lvis_train_ann = f'{root}/lvis_v1_train.json'
    lvis_val_ann = f'{root}/lvis_v1_val.json'
    coco_train_ann = f'{root}/stuff_annotations_trainval2017/'\
        'annotations/stuff_train2017.json'
    coco_val_ann = f'{root}/stuff_annotations_trainval2017/'\
        'annotations/stuff_val2017.json'
    coco2lvis = 'tools/convert_datasets/coco-lvis/coco2lvis.pkl'

    img2info = pickle.load(
        open('tools/convert_datasets/coco-lvis/img2info.pkl', 'rb'))
    img2info_val = pickle.load(
        open('tools/convert_datasets/coco-lvis/img2info_val.pkl', 'rb'))

    lvis_train_ann = LVIS(lvis_train_ann)
    lvis_val_ann = LVIS(lvis_val_ann)
    coco_train_ann = COCO(coco_train_ann)
    coco_val_ann = COCO(coco_val_ann)
    coco2lvis = pickle.load(open(coco2lvis, 'rb'))

    train_process = Processor(
        lvis_train_ann, coco_train_ann, img2info, args.nproc, root, coco2lvis)
    val_process = Processor(
        lvis_val_ann, coco_train_ann, img2info_val, args.nproc, root,
        coco2lvis, coco_val_ann)
    train_process()
    val_process()

    new_images_path = f'{root}/images'
    if not osp.exists(new_images_path):
        os.makedirs(new_images_path)
    shutil.move(f'{root}/train2017', f'{new_images_path}/train2017')
    shutil.move(f'{root}/val2017', f'{new_images_path}/val2017')
    shutil.move(f'{root}/test2017', f'{new_images_path}/test2017')


class Processor:
    def __init__(self,
                 lvis: LVIS,
                 coco: COCO,
                 img2info,
                 nproc,
                 root,
                 coco2lvis,
                 coco_val: COCO = None):
        self.lvis = lvis
        self.coco = coco
        self.lvis_images = list(lvis.imgs.keys())
        self.coco_images = list(coco.imgs.keys())
        self.root = root
        self.coco_val = coco_val
        self.coco2lvis = coco2lvis
        if coco_val:
            self.coco_val_images = list(coco_val.imgs.keys())
        else:
            self.coco_val_images = None
        self.not_in_coco = []

        self.no_annotations = []
        self.img2info = img2info
        self.nproc = nproc

    def __call__(self):
        if self.nproc > 1:
            mmcv.track_parallel_progress(
                self.process_one_img, self.lvis_images, self.nproc)
        else:
            mmcv.track_progress(self.process_one_img, self.lvis_images)

    def process_one_img(self, img):
        lvis_mask_dir = f'{self.root}/lvis_mask'
        path = self.img2info[img]['path'].replace('jpg', 'png')
        if osp.exists(f'{lvis_mask_dir}/{path}'):
            return

        if not self.coco_val_images:
            if img not in self.coco_images:
                self.not_in_coco.append(img)
                return
        else:
            if img not in self.coco_images and img not in self.coco_val_images:
                self.not_in_coco.append(img)
                return
        ann_ids = self.lvis.get_ann_ids(img_ids=[img])
        has_lvis = False
        for i, idx in enumerate(ann_ids):
            has_lvis = True
            if i == 0:
                mask = self.lvis.anns[idx]['category_id'] * \
                    self.lvis.ann_to_mask(self.lvis.anns[idx])
                add_mask = (mask == 0)
            else:
                mask = mask + add_mask * \
                    self.lvis.anns[idx]['category_id'] * \
                    self.lvis.ann_to_mask(self.lvis.anns[idx])
                add_mask = (mask == 0)
        if not self.coco_val_images:
            coco_ann_ids = self.coco.getAnnIds(imgIds=[img])
        else:
            if img not in self.coco_images:
                coco_ann_ids = self.coco_val.getAnnIds(imgIds=[img])
                in_train = False
            else:
                coco_ann_ids = self.coco.getAnnIds(imgIds=[img])
                in_train = True
        if len(coco_ann_ids) == 0:
            logger.error(img)

        has_coco = False
        for i, idx in enumerate(coco_ann_ids):
            has_coco = True
            if i == 0:
                if not self.coco_val_images or in_train:
                    cmask = self.coco.anns[idx]['category_id'] * \
                        self.coco.annToMask(self.coco.anns[idx])
                    c_add_mask = (cmask == 0)
                else:
                    cmask = self.coco_val.anns[idx]['category_id'] * \
                        self.coco_val.annToMask(self.coco_val.anns[idx])
                    c_add_mask = (cmask == 0)
            else:
                if not self.coco_val_images or in_train:
                    cmask = cmask + c_add_mask * \
                        self.coco.anns[idx]['category_id'] * \
                        self.coco.annToMask(self.coco.anns[idx])
                    c_add_mask = (cmask == 0)
                else:
                    cmask = cmask + c_add_mask * \
                        self.coco_val.anns[idx]['category_id'] * \
                        self.coco_val.annToMask(self.coco_val.anns[idx])
                    c_add_mask = (cmask == 0)

        if has_coco:
            for k, val in self.coco2lvis.items():
                cmask[cmask == k] = val
        if has_coco and has_lvis:
            final_mask = mask + add_mask * cmask
        elif has_coco:
            final_mask = cmask
        elif has_lvis:
            final_mask = mask
        else:
            self.no_annotations.append(img)
            return

        mask_array = final_mask.astype(np.uint16)
        lvis_mask_dir = f'{self.root}/lvis_mask'
        if not osp.exists(lvis_mask_dir):
            os.makedirs(lvis_mask_dir)

        if not self.coco_val_images:
            path = self.img2info[img]['path'].replace('jpg', 'png')
            if not osp.exists(f'{lvis_mask_dir}/train2017'):
                os.makedirs(f'{lvis_mask_dir}/train2017')
            io.imwrite(f'{lvis_mask_dir}/{path}', mask_array)
        else:
            path = self.img2info[img]['path'].replace('jpg', 'png')
            if not osp.exists(f'{lvis_mask_dir}/val2017'):
                os.makedirs(f'{lvis_mask_dir}/val2017')
            io.imwrite(f'{lvis_mask_dir}/{path}', mask_array)


if __name__ == '__main__':
    main()
