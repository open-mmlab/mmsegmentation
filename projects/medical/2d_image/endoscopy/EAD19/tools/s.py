#!/usr/bin/env python3
"""Created on Fri Mar  1 18:18:45 2019.

@author: shariba
"""

import os


def unzipFiles(path_to_zip_file, directory_to_extract_to):
    import zipfile
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()


def download(url, fileName, blocksize=200):
    import math

    import requests
    from tqdm import tqdm
    response = requests.get(url + fileName, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = blocksize
    wrote = 0
    with open(fileName, 'wb') as f:
        #        for data in tqdm(response.iter_content()):
        for data in tqdm(
                response.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit='KB',
                unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    # meta-data


#    print(response.status_code)
#    print(response.headers['content-type'])
#    print(response.encoding)


def createSemanticTestImages(txtFileName, src, dst):
    import os
    from shutil import copyfile
    with open(txtFileName) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines)):
        copyfile(os.path.join(src, lines[i]), os.path.join(dst, lines[i]))


if __name__ == '__main__':
    from shutil import move, rmtree

    # download flag
    flag_d = 0

    url = 'https://s3.amazonaws.com/ead2019-test-detection/'
    testImages = 'testImages_EAD2019.zip'
    """============== Prepare data for test detection ================="""
    # check if testImage exists
    if (os.path.isfile(testImages) or os.path.isdir(testImages.split('.')[0])):
        print('folder already exists..., delete it to re-download data')
    else:
        print('downloading test data for detection task-1')
        download(url, testImages)
        flag_d = 1

    # save images for detection
    folderDetection_Images = 'test_images_detection'
    if os.path.isdir(testImages.split('.')[0] + '/' + folderDetection_Images):
        print('folder already exists...')
    else:
        if (os.path.isfile(testImages) != 1):
            print('downloading test data for detection task-1')
            download(url, testImages)
            flag_d = 1
        unzipFiles(testImages, testImages.split('.')[0])

    if flag_d:
        os.remove(testImages)
#    move(os.path.join(testImages.split('.')[0],testImages.split('.')[0]),
#  os.path.join(testImages.split('.')[0], folderDetection_Images))
    """============== Prepare data for test semantic ================="""
    folderSemantic_Images = 'test_images_semantic'
    os.makedirs(
        os.path.join(testImages.split('.')[0], folderSemantic_Images),
        exist_ok=True)
    fileName = 'semanticTestImageList_EAD2019.txt'
    download(url, fileName, blocksize=1)

    createSemanticTestImages(
        fileName,
        os.path.join(testImages.split('.')[0], folderDetection_Images),
        os.path.join(testImages.split('.')[0], folderSemantic_Images))
    """============== Prepare data for generalization =================
    This data is from the 6th center which is not included in the training set
    """
    fileName = 'testGeneralization_EAD2019.zip'
    print('downloading test data for generalization task-3')
    download(url, fileName, blocksize=52)

    # Note: only bounding box detection is needed to be submitted
    folderGeneralization_Images = 'test_images_generalization'

    if os.path.isdir(fileName.split('.')[0]):
        print('folder already exists, delete it unzipping')
    else:
        unzipFiles(fileName, fileName.split('.')[0])
        move(
            os.path.join(
                fileName.split('.')[0], 'test_generalization_EAD2019'),
            os.path.join(testImages.split('.')[0], 'test_generalization'))

    rmtree(fileName.split('.')[0])
    os.remove(fileName)
    """============== Check if all folders and files exists =============="""
    import glob
    l_detection = len(
        glob.glob(
            os.path.join(
                testImages.split('.')[0], 'test_images_detection', '*.jpg'),
            recursive=True))
    l_semantic = len(
        glob.glob(
            os.path.join(
                testImages.split('.')[0], 'test_images_semantic', '*.jpg'),
            recursive=True))
    l_generalization = len(
        glob.glob(
            os.path.join(
                testImages.split('.')[0], 'test_generalization', '*.jpg'),
            recursive=True))

    if (l_detection == 195 and l_semantic == 121 and l_generalization == 51):
        print('SUCCESS!!!')
    else:
        print('something went wrong, please check your folder...')
