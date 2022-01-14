import numpy as np
import matplotlib.pyplot as plt
import os

# def show_confMat(confusion_mat, classes_name, set_name, out_dir,ax):
#     """
#     可视化混淆矩阵，保存png格式
#     :param confusion_mat: nd-array
#     :param classes_name: list,各类别名称
#     :param set_name: str, eg: 'valid', 'train'
#     :param out_dir: str, png输出的文件夹
#     :return:
#     """

#     # 归一化
#     confusion_mat_N = confusion_mat.copy()
#     # confusion_mat_N = confusion_mat_N.astype('float') / confusion_mat_N.sum(axis=1)[:, np.newaxis]

#     # 获取颜色
#     cmap = plt.cm.get_cmap(
#         'BuPu'
#     )  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
#     im=ax.imshow(confusion_mat_N, cmap=cmap)


#     # 设置文字
#     xlocations = np.array(range(len(classes_name)))

#     ax.xaxis.set_ticks_position('top') 
#     plt.xticks(xlocations, classes_name, rotation=60,)
#     plt.yticks(xlocations, classes_name)

#     # 保存
#     plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))
    
def norm(mat):
    confusion_mat_N = mat.copy()
    confusion_mat_N = confusion_mat_N.astype('float') / confusion_mat_N.sum(axis=1)[:, np.newaxis]
    return confusion_mat_N



data1=np.array([[21075636,   447976,   849189,   183406,    92624,    22837],
        [  799558, 20719389,   371593,    38434,    15779,     4655],
        [  649032,   203086, 14621448,  1887284,     4500,     5435],
        [  149387,    45380,  1501524, 16811874,     1009,     2924],
        [  106488,    16553,     5764,     1426,   664722,      480],
        [  183227,   148438,     5197,      910,    19850,   334929]])

data2=np.array([[21237527,   422226,   774217,   171876,    57830,     7992],
        [  548330, 21131585,   212216,    44021,     9822,     3434],
        [  669343,   190216, 14381328,  2127453,     2389,       56],
        [  164992,    27690,  1271714, 17045566,      533,     1603],
        [   91922,    16639,     3882,     1876,   680915,      199],
        [  243447,   130210,     8483,     1004,     7799,   301608]])

data3=np.array([[21213732,   455588,   748926,   178289,    65023,    10110],
        [  497497, 21232005,   164006,    43077,    11223,     1600],
        [  730607,   227176, 14492509,  1912634,     3506,     4353],
        [  172472,    37015,  1327723, 16971992,      861,     2035],
        [   78748,    11804,     2126,     1464,   700958,      333],
        [  200536,   187836,     3441,      592,    10572,   289574]])
data=[data1,data2,data3]
# # show_confMat(data1, ['imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter'], "mit-b0", "./",1)
# # show_confMat(data2, ['imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter'], "mit-b4", "./",2)

    
# fig, ax =plt.subplots(2,2, frameon = False)
# ax = ax.flatten() 
# im=show_confMat(data2-data1, ['imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter'], "mit-b4-fuse", "./",ax[0])
# im=show_confMat(data3-data1, ['imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter'], "mit-b4-fuse", "./",ax[1])
# fig.colorbar(im, ax=[ax[0],ax[1]], fraction=0.03, pad=0.05)
# plt.show()

fig, ax = plt.subplots(1, 3)
ax = ax.flatten()
 

cmap = plt.cm.get_cmap('BuPu') 
classes_name=['imp surf', 'building', 'low_veg', 'tree', 'car', 'clutter']
xlocations = np.array(range(len(classes_name)))


# for i in range(2):
#     # im = ax[i].imshow(norm(data[i]),cmap=cmap)
#     im = ax[i].imshow(data[i+1]-data[0],cmap=cmap)
#     ax[i].xaxis.set_ticks_position('top')
#     ax[i].set_xticks(xlocations)
#     ax[i].set_xticklabels(classes_name, rotation=60)
#     ax[i].set_yticks(xlocations)
#     ax[i].set_yticklabels(classes_name)

for i in range(3):
    # im = ax[i].imshow(norm(data[i]),cmap=cmap)
    im = ax[i].imshow(norm(data[i]),cmap=cmap)
    majorFormatter = plt.FormatStrFormatter('%1.1f') 
    for j in range(data[i].shape[0]):
        for k in range(data[i].shape[0]):
            ax[i].annotate('%.3f'%norm(data[i])[k, j],xy=(j, k), horizontalalignment='center', verticalalignment='center')
    ax[i].xaxis.set_ticks_position('top')
    ax[i].set_xticks(xlocations)
    ax[i].set_xticklabels(classes_name, rotation=60)
    ax[i].set_yticks(xlocations)
    ax[i].set_yticklabels(classes_name)
 
fig.colorbar(im, ax=[ax[0], ax[1],ax[2]], fraction=0.015, pad=0.05)




# plt.imshow(data1, cmap=cmap)
# ax=plt.gca()
# ax.xaxis.set_ticks_position('top') 
# plt.xticks(xlocations, classes_name, rotation=60,)
# plt.yticks(xlocations, classes_name)
# plt.colorbar()

plt.savefig('tjn.png', bbox_inches='tight')
plt.show()