# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from glob import glob
import os.path as osp
import tempfile
import zipfile
import numpy as np
import matplotlib.pyplot as plt

import mmcv
from mmengine.utils import mkdir_or_exist

HRF_LEN = 15
TRAINING_LEN = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('--raw_data_root', help='the root path of raw data')
    
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args



def extract_img(root: str,cur_dir: str,out_dir: str,mode: str='train',file_type: str='img')-> None:
    """_summary_

    Args:
       Args:
        root (str): root where the extracted data is saved
        cur_dir (cur_dir): dir where the zip_file exists
        out_dir (str): root dir where the data is saved
        
        mode (str, optional): _description_. Defaults to 'train'.
        file_type (str, optional): _description_. Defaults to 'img',else to 'mask'.
    """
    print('mode',mode)
    zip_file = zipfile.ZipFile(cur_dir)
    zip_file.extractall(root)
    for cur_dir,dirs,files in os.walk(root):
        #filter child dirs and directories with "Illustration" and "MACOSX"
        if len(dirs) == 0 and cur_dir.split('\\')[-1].find('Illustration') == -1 \
        and cur_dir.find('MACOSX') == -1:
            
            file_names = [file for  file in files if file.endswith('.jpg') or file.endswith('.bmp') ]
            for filename in sorted(file_names):
                img = mmcv.imread(osp.join(cur_dir, filename))
                
                if file_type == 'annotations':
                    img = img[:,:,0]
                    img[np.where(img==0)] = 1
                    img[np.where(img==128)] = 2
                    img[np.where(img==255)] = 0
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, file_type, mode,
                             osp.splitext(filename)[0] + '.png'))    

def main():
    args = parse_args()
   
    raw_data_root = args.raw_data_root
    if args.out_dir is None:
        out_dir = osp.join('./data', 'REFUGE')
        print('out_dir:',out_dir)
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'images'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mkdir_or_exist(osp.join(out_dir, 'images', 'test'))
    mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))
    mkdir_or_exist(osp.join(out_dir, 'annotations', 'test'))

    print('Generating images and annotations...')
    cur_dir,dirs,files = list(os.walk(raw_data_root))[0]
    print("====================")
    
    
    files = list(filter(lambda x: x.endswith('.zip'),files))
    files.remove('Disc_Cup_Masks.zip')
    print(files)

    
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for file in files:
            mode = list(filter(lambda x:file.lower().find(x)!= -1,['training','test','validation']))[0]
            file_root = osp.join(tmp_dir,file[:-4])
            file_type = 'images' if file.find('Anno') == -1 and file.find('GT')== -1 else 'annotations'
            # if file_type == 'annotations':
            extract_img(file_root,osp.join(cur_dir,file),out_dir,mode,file_type)
    
    
    print('Done!')




if __name__ == '__main__':
    main()
    # raw_data_root = 'D:\OpenMMlab\Refuge'
    # print(os.listdir('../../'))
    # tmp_dir = '../mmsegmentation/data/refuge'
    # print(os.listdir(tmp_dir))
    # print(glob(tmp_dir))
    # # for cur_dir,dirs,files in list(os.walk(raw_data_root))[:1]:
    # cur_dir,dirs,files = list(os.walk(raw_data_root))[0]
    # print("====================")
    
    # # print('cur:',cur_dir,dirs,files)
    # # for cur_dir,dirs,files in list(os.walk(raw_data_root)):
    # #     child_dir = cur_dir.split('\\')[-1]
    # #     if child_dir not in ['Glaucoma','Non-Glaucoma']:
    # #         print(child_dir)
    # #         print("现在的目录：" + cur_dir)
    # #         print("该目录下包含的子目录：" + str(dirs))
    # #         print("该目录下包含的文件：" + str(files))
    # files = list(filter(lambda x: x.endswith('.zip'),files))
    # print(files)

    # # print(cur_dir)
    # # print(child_dir)
    # with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dir:
    #     for file in files:
    #         mode = list(filter(lambda x:file.lower().find(x)!= -1,['training','test','validation']))[0]
    #         file_root = osp.join(tmp_dir,file[:-4])
    #         file_type = 'img' if file.find('Anno') == -1 and file.find('GT')== -1 else 'annotations'
    #         if file_type == 'annotations':
    #             extract_img(file_root,osp.join(cur_dir,file),mode,file_type)
            # if file.lower().find('train') != -1:
            #     train_root = osp.join(tmp_dir,file[:-4])
            #     file_type = 'img' if file.find('Anno') == -1 and file.find('GT')== -1 else 'mask'
            #     zip_file = zipfile.ZipFile(osp.join(cur_dir,file))
            #     zip_file.extractall(train_root)


        # if child_dir in ['Glaucoma','Non-Glaucoma']:
        #     print(child_dir)
        #     print("现在的目录：" + cur_dir)
        #     print("该目录下包含的子目录：" + str(dirs))
        #     print("该目录下包含的文件：" + str(files))
    # mmcv.mkdir_or_exist(tmp_dir)
    # print(os.listdir(tmp_dir))
    # now_path = os.path.join(raw_data_root,'REFUGE-Validation400.zip')
    # out_dir = osp.join('data', 'HRF')
    
    # print('Making directories...')
    # mkdir_or_exist(out_dir)
    # mkdir_or_exist(osp.join(out_dir, 'images'))
    # mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    # mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    # mkdir_or_exist(osp.join(out_dir, 'annotations'))
    # mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    # mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    # with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dir:
    #         print(tmp_dir)
    #         zip_file = zipfile.ZipFile(now_path)
    #         zip_file.extractall(tmp_dir)
    #         for filename in sorted(os.listdir(tmp_dir)):
    #             img = mmcv.imread(osp.join(tmp_dir, filename))
