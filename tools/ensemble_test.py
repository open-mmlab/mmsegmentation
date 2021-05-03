import numpy as np
from PIL import Image
import os.path as osp
import os
# from scipy import stats
from tqdm import tqdm
import torch

# ensemble learning vote


def save_img(result, imgfile_prefix, basename):
    if not os.path.exists(imgfile_prefix):
        os.makedirs(imgfile_prefix)
    output = Image.fromarray(result.astype(np.uint8))
    png_filename = osp.join(imgfile_prefix, f'{basename}.png')
    output.save(png_filename)


result_dir = ['pup_trainval_160k_test_results_1', 'pup_trainval_160k_MS_test_results',
              'pup_trainval_200k_ade20k_test_results', 'mla_trainval_320k_test_results']
file_list = [os.listdir(result_dir[0]), os.listdir(
    result_dir[1]), os.listdir(result_dir[2]), os.listdir(result_dir[3])]
imgfile_prefix = './ensemble_learning_vote_result'
print('save result to : ', os.path.abspath(imgfile_prefix))

for i in tqdm(range(len(file_list[0]))):
    im2arr_concat = None
    basename = osp.splitext(osp.basename(file_list[0][i]))[0]

    for j in range(len(result_dir)):
        im2arr = np.array(Image.open(osp.join(result_dir[j], file_list[j][i])))
        if im2arr_concat is None:
            im2arr_concat = im2arr[np.newaxis, :]
        else:
            im2arr_concat = np.concatenate(
                (im2arr_concat, im2arr[np.newaxis, :]))

    mode_arr = np.zeros_like(im2arr)

    # use torch.mode
    im2arr_concat_tensor = torch.from_numpy(im2arr_concat)
    mode_tensor = torch.mode(im2arr_concat_tensor, dim=0)[0]
    mode_arr = mode_tensor.numpy()

    save_img(mode_arr, imgfile_prefix, basename)

print('Done!')


# ensemble learning average seg_logit
# seg_logit_dir = ['pup_trainval_160k_test_results_1', 'pup_trainval_160k_MS_test_results', 'pup_trainval_200k_ade20k_test_results', 'mla_trainval_320k_test_results']
# file_list = [os.listdir(result_dir[0]), os.listdir(result_dir[1]), os.listdir(result_dir[2]), os.listdir(result_dir[3])]

# for i in range(len(file_list[0])):
#     logit_npy_sum = None
#     basename = osp.splitext(osp.basename(file_list[0][i]))[0]

#     for j in range(len(seg_logit_dir)):
#         logit_npy = np.load(osp.join(seg_logit_dir[j], file_list[j][i]))
#         if logit_npy_sum is None:
#             logit_npy_sum = logit_npy
#         else:
#             logit_npy_sum = logit_npy_sum + logit_npy

#     logit_npy_mean = logit_npy_sum / len(seg_logit_dir)

#     seg_pred = logit_npy_mean.argmax(axis=0)
#     seg_pred = seg_pred + 1
#     save_img(seg_pred,'.', basename)
