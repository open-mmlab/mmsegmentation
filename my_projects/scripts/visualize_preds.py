import argparse
import os
from copy import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import plotting_utils as p_utils
from mmengine.structures import PixelData
from mmseg.utils import (
    hots_v1_classes, 
    hots_v1_palette,
    irl_vision_sim_classes,
    irl_vision_sim_palette,
    ade_classes, 
    ade_palette,
    hots_v1_cat_classes, 
    hots_v1_cat_palette,
    irl_vision_sim_cat_classes,
    irl_vision_sim_cat_palette,
    arid20cat_classes,
    arid20cat_palette,
    arid10cat_classes,
    arid10cat_palette,
    sodhots_c_classes,
    sodhots_c_palette,
    cocostuff_classes,
    cocostuff_palette
)
from mmseg.visualization.local_visualizer_custom import SegLocalVisualizerCustom as SegVis

from math import ceil

# TODO temp, could be taken from testloaders etc
DATASET_TEST_IMG_PATH = {
    "HOTS"              :           "/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/img_dir/test",
    "HOTS-C"            :           "/media/ids/Ubuntu files/data/HOTS_v1_cat/SemanticSegmentation/img_dir/test",
    "SOD"               :           "/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/img_dir/test",
    "SOD-C"             :           "/media/ids/Ubuntu files/data/irl_vision_sim_cat/SemanticSegmentation/img_dir/test", 
    "ARID20"            :           "/media/ids/Ubuntu files/data/ARID20_CAT/img_dir/test",
    "ARID10"            :           "/media/ids/Ubuntu files/data/ARID10_CAT/img_dir/test",
    "SODHOTS-C"         :           "/media/ids/Ubuntu files/data/SODHOTS-C/SemanticSegmentation/img_dir/test"
}

DATASET_TEST_ANN_PATH = {
    "HOTS"              :           "/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/ann_dir/test",
    "HOTS-C"            :           "/media/ids/Ubuntu files/data/HOTS_v1_cat/SemanticSegmentation/ann_dir/test",
    "SOD"               :           "/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/ann_dir/test",
    "SOD-C"             :           "/media/ids/Ubuntu files/data/irl_vision_sim_cat/SemanticSegmentation/ann_dir/test", 
    "ARID20"            :           "/media/ids/Ubuntu files/data/ARID20_CAT/ann_dir/test",
    "ARID10"            :           "/media/ids/Ubuntu files/data/ARID10_CAT/ann_dir/test",
    "SODHOTS-C"         :           "/media/ids/Ubuntu files/data/SODHOTS-C/SemanticSegmentation/ann_dir/test"
}

DATASET_PALETTE = {
    "HOTS"              :           hots_v1_palette,
    "HOTS-C"            :           hots_v1_cat_palette,
    "SOD"               :           irl_vision_sim_palette,
    "SOD-C"             :           irl_vision_sim_cat_palette, 
    "ARID20"            :           arid20cat_palette,
    "ARID10"            :           arid10cat_palette,
    "ADE20K"            :           ade_palette,
    "SODHOTS-C"         :           sodhots_c_palette
}

DATASET_CLASSES = {
    "HOTS"              :           hots_v1_classes,
    "HOTS-C"            :           hots_v1_cat_classes,
    "SOD"               :           irl_vision_sim_classes,
    "SOD-C"             :           irl_vision_sim_cat_classes, 
    "ARID20"            :           arid20cat_classes,
    "ARID10"            :           arid10cat_classes,
    "ADE20K"            :           ade_classes,
    "SODHOTS-C"         :           sodhots_c_classes
}

DATASET_DEFAULT_PRED_LIST = {
    "HOTS"              :       [
        "kitchen_6_top_raw_0.png",
        "mix_2_top_raw_8.png",
        "office_7_top_raw_4.png",
        "table_8_top_raw_0.png"
    ],
    "HOTS-C"            :       [
        "kitchen_6_top_raw_0.png",
        "mix_2_top_raw_8.png",
        "office_7_top_raw_4.png",
        "table_8_top_raw_0.png"
    ],
    "SOD"               :       [
        "scene_75.png",
        "scene_99.png",
        "scene_109.png",
        "scene_273.png"
    ],
    "SOD-C"             :       [
        "scene_75.png",
        "scene_99.png",
        "scene_109.png",
        "scene_273.png"
    ],
    "ARID20"            :       [
        "floor_bottom_seq03_17.png",
        "table_bottom_seq01_19.png",
        "table_top_seq02_16.png",
        "table_top_seq06_16.png"
    ],
    "ARID10"            :       [
        "floor_bottom_box_seq33_5.png",
        "floor_top_non-fruits_seq39_7.png",
        "table_bottom_mixed_seq13_5.png",
        "table_top_non-fruits_seq21_3.png"
    ],
    "ARID20_CLUTTER"    :       [
        "table_top_seq03_7.png",
        "table_top_seq03_14.png",
        "table_top_seq03_19.png"
    ],
    "SODHOTS-C"         :       [
        "mix_2_top_raw_8.png",
        "scene_99.png",
        "office_6_top_raw_5.png",
        "scene_395.png"
    ]   
}

CONVERSION_IDX = 255
CONVERSION_COLOR = [255, 255, 255]
CONVERSION_LABEL = "unknown"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_path',
        help='dir containing model dirs',
        type=str
    )
    parser.add_argument(
        'dataset_name',
        type=str
    )
    parser.add_argument(
        '--pred_names',
        '-pn',
        type=str,
        nargs='+',
        default=[]
    )
    
    parser.add_argument(
        '--save_path',
        '-sp',
        type=str,
        default="my_projects/images_plots/prediction_images"
    )
    parser.add_argument(
        '--set_name',
        '-sn',
        type=str,
        default=None
    )
    parser.add_argument(
        '--save_separate',
        '-sep',
        action='store_true'
    )
    parser.add_argument(
        '--show',
        action='store_true'
    )
    parser.add_argument(
        '--legend_mode',
        choices=[
            "none", "full", 
            "interesting", "separate",
            "gt_only"
        ],
        default="interesting"
    )
    parser.add_argument(
        '--legend_pos',
        choices=[
            "side", "low"
        ],
        default="side"
    )
    parser.add_argument(
        '--ignore_background',
        '-ibg',
        action='store_true'
    )
    parser.add_argument(
        '--alpha',
        '-a',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--include_label',
        '-il',
        action='store_true'
    )
    parser.add_argument(
        '--label_scale',
        '-ls',
        type=float,
        default=0.04
    )
    parser.add_argument(
        '--include_raw',
        '-raw',
        action='store_true'
    )
    parser.add_argument(
        '--include_gt',
        '-gt',
        action='store_true'
    )
    parser.add_argument(
        '--clutter',
        '-cl',
        action='store_true'
    )
    parser.add_argument(
        '--test_dataset',
        '-tds',
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_dataset',
        '-ods',
        type=str,
        default=None
    )
    # parser.add_argument(
    #     '--with_iou',
    #     '-iou',
    #     action='store_true'
    # )
    
    args = parser.parse_args()
    # # TODO temp defaults
    # args.include_raw = True
    # args.include_gt = True
    
    args.dataset_name = p_utils.map_dataset_name(args.dataset_name)
    if args.test_dataset is None:
        args.test_dataset = args.dataset_name
    else:
        args.test_dataset = p_utils.map_dataset_name(args.test_dataset)
        
    if args.output_dataset is None:
        args.output_dataset = args.dataset_name
    else:
        args.output_dataset = p_utils.map_dataset_name(args.output_dataset)
        
    if not args.pred_names:
        ds_name = args.test_dataset
        if ds_name == "ARID20" and args.clutter:
            ds_name = "ARID20_CLUTTER"
        args.pred_names = DATASET_DEFAULT_PRED_LIST[ds_name]
    if not args.set_name:
        pred_name = args.pred_names[0] if len(args.pred_names) == 1 else "set"
        if ds_name == "ARID20" and args.clutter:
            pred_name = f"clutter_{pred_name}"
        pred_name = f"_{pred_name}"
        args.set_name = f"{args.dataset_name}{pred_name}".replace(
            ".png", ""
        )
    print("args")
    print(args)
    print()
    return args


def get_gt_img(test_dataset, pred_name):
    gt_path = os.path.join(
        DATASET_TEST_ANN_PATH[test_dataset],
        pred_name
    )
    try:
        gt_img = np.array(Image.open(gt_path)).astype(np.uint8)
        return gt_img
    except:
        print(f"failed to open gt image at {gt_path}")
        return None
        
def get_pred_label(pred_name, model_dir_path):
    pred_label_path = os.path.join(
        model_dir_path,
        "pred_results",
        pred_name
    )
    try:
        pred_label_img = np.array(Image.open(pred_label_path)).astype(np.uint8)
        return pred_label_img
    except:
        print(f"failed to open pred image at {pred_label_path}")
        return None

def get_rgb_img(test_dataset, pred_name):
    rgb_path = os.path.join(
        DATASET_TEST_IMG_PATH[test_dataset],
        pred_name
    )
    try:
        rgb_img = np.array(Image.open(rgb_path))
        return rgb_img
    except Exception as ex:
        print(f"failed to open rgb image at {rgb_path}\n{ex}") 
        return None

def label_to_pixeldata(label: np.ndarray) -> PixelData:
    return PixelData(data=label)

def get_model_pred_labels(results_path, pred_name):
    pred_dict = {}
    for model_dir_name in os.listdir(results_path):
        if model_dir_name == "data":
            continue
        model_name = p_utils.map_model_name(
            model_name=model_dir_name.split("_")[0]
        )
        model_dir_path = os.path.join(
            results_path,
            model_dir_name
        )
        pred_label = get_pred_label(pred_name=pred_name, model_dir_path=model_dir_path)
        pred_dict[model_name] = pred_label
    return pred_dict

def get_all_figures_dict(args):
    figures_dict = {}
    for pred_name in args.pred_names:
        figures_dict[pred_name] = {}
        if args.include_raw:
            figures_dict[pred_name]["Input"] = get_rgb_img(
                test_dataset=args.test_dataset, 
                pred_name=pred_name
            )
        if args.include_gt:
            figures_dict[pred_name]["Ground Truth"] = get_gt_img(
                test_dataset=args.test_dataset, 
                pred_name=pred_name
            )
        
        pred_dict = get_model_pred_labels(
            results_path=args.results_path,
            pred_name=pred_name
        )
        for model_name, model_data in pred_dict.items():
            figures_dict[pred_name][model_name] = model_data
    
    return figures_dict



def get_ids_interesting(figures_dict):
    gt_label = figures_dict["Ground Truth"]
    gt_label_ids = []
    pred_label_ids = []
    for fig_name, pred_label in figures_dict.items():
        mismatch_pred = pred_label[pred_label != gt_label]
        mismatch_gt = gt_label[gt_label != pred_label]
        for pred_id in np.unique(mismatch_pred):
            if len(mismatch_pred[mismatch_pred == pred_id]) < 10:
                continue
            if pred_id not in pred_label_ids:
                pred_label_ids.append(pred_id)
        for gt_id in np.unique(mismatch_gt):
            if len(mismatch_gt[mismatch_gt == gt_id]) < 10:
                continue
            if gt_id not in gt_label_ids:
                gt_label_ids.append(gt_id)
    pred_label_ids = [
        pred_id for pred_id in pred_label_ids 
            if pred_id not in gt_label_ids
    ]
    if 0 in gt_label_ids:
        gt_label_ids = sorted(gt_label_ids)[1:]
    
    
    return gt_label_ids, pred_label_ids

def get_ids_full(figures_dict):
    gt_label_ids = np.unique(figures_dict["Ground Truth"])
    pred_label_ids = []
    for fig_name, pred_label in figures_dict.items():
        for pred_id_ in np.unique(pred_label):
            # skip very small areas
            if len(pred_label[pred_label == pred_id_]) < 10:
                continue
            if pred_id_ not in gt_label_ids and pred_id_ not in pred_label_ids:
                pred_label_ids.append(pred_id_)
    
    return gt_label_ids, pred_label_ids
        
def get_ids_gt_only(figures_dict):
    return np.unique(figures_dict["Ground Truth"]), []



def generate_set_figure(args):
    p_utils.reset_params()
    figures_dict = get_all_figures_dict(args=args)
    n_rows = len(figures_dict.keys())
    n_cols = len(tuple(figures_dict.items())[0][1].keys())
    # set params
    param_dict = copy(p_utils.PRED_VISUALIZATION_FIGURE_PARAMS)
    # (fig_w, fig_h) = param_dict['figure.figsize']
    # fig_h *= n_rows
    param_dict['figure.figsize'] = (2*n_cols, 1.6*n_rows)
    p_utils.set_params(param_dict=param_dict)
    
    figure = plt.figure()
    subplots = figure.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        gridspec_kw = {
            'wspace':0.01, 
            'hspace':0.001
        }
    )
    print(f"drawing and labeling using {args.output_dataset}")
    classes = copy(DATASET_CLASSES[args.output_dataset]())
    palette = copy(DATASET_PALETTE[args.output_dataset]())
    print(f"getting ann and rgb from {args.test_dataset}")     
    visualizer = SegVis(alpha=args.alpha)
    
    for row_idx, (pred_name, pred_dict) in enumerate(figures_dict.items()):
        figures_item_list = list(pred_dict.items())
        rgb_img = get_rgb_img(
            test_dataset=args.test_dataset, 
            pred_name=pred_name
        )
        if rgb_img is None:
            return
        for col_idx in range(n_cols):
            (fig_name, label) = figures_item_list.pop(0)
            if n_rows == 1:
                axes = subplots[col_idx]
            else:
                axes = subplots[row_idx][col_idx]
            if not fig_name == "Input":
                classes_ = classes
                palette_ = palette
                if "bise" in fig_name.lower() and args.output_dataset == "ADE20K":
                    classes_ = cocostuff_classes()
                    palette_ = cocostuff_palette()
                img = visualizer._draw_sem_seg(
                    image=rgb_img.copy(),
                    sem_seg=label_to_pixeldata(label=label),
                    classes=classes_,
                    palette=palette_,
                    with_labels=args.include_label,
                    label_scale=args.label_scale,
                    ignore_idx=0,
                    ignore_background=args.ignore_background
                )
            else:
                img = label
            axes.imshow(img)
            axes.axis('off')
            if row_idx == 0:
                axes.set_title(fig_name)
    plt.subplots_adjust(hspace=0)
    figure.tight_layout()
    save_path = os.path.join(
        args.save_path,
        f"{args.set_name}.png"
    )
    print(f"saved fig in : {save_path}")
    figure.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight'
    )
    
    
def main():
    args = parse_args()
    generate_set_figure(args=args)

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
# def generate_set_figure_old(args):
#     p_utils.reset_params()
#     p_utils.set_params(param_dict=p_utils.PRED_VISUALIZATION_FIGURE_PARAMS_2ROWS)
#     rgb_img = get_rgb_img(args=args)
#     if rgb_img is None:
#         return
#     figure = plt.figure()
#     subplots  = figure.subplots(
#         nrows=2, 
#         ncols=3, 
#         gridspec_kw = {
#             'wspace':0.05, 
#             'hspace':0.20
#         }
#     )
#     figures_dict = get_all_figures_dict(args=args)
#     classes = copy(DATASET_CLASSES[args.dataset_name]())
#     palette = copy(DATASET_PALETTE[args.dataset_name]())
    
        
#     figures_item_list = list(figures_dict.items())
#     visualizer = SegVis(alpha=args.alpha)
#     for row_idx in range(len(subplots)):
#         for col_idx in range(len(subplots[row_idx])):
#             (fig_name, label) = figures_item_list.pop(0)
#             axes = subplots[row_idx][col_idx]
            
                
#             img = visualizer._draw_sem_seg(
#                 image=rgb_img.copy(),
#                 sem_seg=label_to_pixeldata(label=label),
#                 classes=classes,
#                 palette=palette,
#                 with_labels=False
#             )
#             axes.imshow(img)
#             axes.axis('off')
#             axes.set_title(fig_name)
#     if (
#         args.legend_mode == "full"
#         or 
#         args.legend_mode == "interesting"
#         or 
#         args.legend_mode == "gt_only" 
        
#     ):
#         legend_handles = generate_legend_handles(
#             figures_dict=figures_dict,
#             classes=classes,
#             palette=palette,
#             legend_mode=args.legend_mode,
#             ignore_background=args.ignore_background
#         )  
#         fontsize = min(
#             max(
#                 300/len(legend_handles),
#                 15
#             ),
#             20
#         )
#         if args.legend_pos == "low":
#             ncols = 4
#             bbheight = ceil(len(legend_handles) / ncols)
#             bby = -1 * ((bbheight * 2) + (bbheight / 2.0)) / 100
#             print(bbheight)
#             print(bby)
            
#             figure.legend(
#                 handles=legend_handles,
#                 # bbox_to_anchor=(0.12, 0.72, 1, 0.2),
#                 bbox_to_anchor=(0.5, bby),
#                 ncols=ncols,
#                 loc = 'lower center',
#                 fontsize=fontsize
#             )
#         else:
#             figure.legend(
#                 handles=legend_handles,
#                 bbox_to_anchor=(0.12, 0.72, 1, 0.2),
#                 fontsize=fontsize
#             )
#     figure.tight_layout()
#     save_path = os.path.join(
#         args.save_path,
#         f"{args.set_name}.png"
#     )
#     print(f"saved fig in : {save_path}")
#     figure.savefig(
#         save_path,
#         dpi=100,
#         bbox_inches='tight'
#     )
#     if args.legend_mode == "separate":
#         figure.clf()
#         figure = plt.figure(figsize=(8,8))
#         legend_handles = generate_legend_handles(
#             figures_dict=figures_dict,
#             classes=classes,
#             palette=palette,
#             legend_mode="full",
#             ignore_background=args.ignore_background
#         )
#         figure.legend(handles=legend_handles, ncols=2)
#         figure.savefig(
#         os.path.join(
#             args.save_path,
#             f"{args.set_name}_legend.png"
#         ),
#         dpi=100,
#         bbox_inches='tight'
#     )

# def generate_legend_handles(
#     figures_dict,
#     classes,
#     palette,
#     legend_mode,
#     ignore_background
# ):
#     gt_label_ids = []
#     pred_label_ids = []
#     if legend_mode == "full":
#         gt_label_ids, pred_label_ids = get_ids_full(
#             figures_dict=figures_dict
#         )
#     if legend_mode == "interesting":
#         gt_label_ids, pred_label_ids = get_ids_interesting(
#             figures_dict=figures_dict
#         )
#     if legend_mode == "gt_only":
#         gt_label_ids, pred_label_ids = get_ids_gt_only(
#             figures_dict=figures_dict
#         )  
        
#     legend_handles = []

#     for gt_label_id in gt_label_ids:
#         if gt_label_id == CONVERSION_IDX:
#             class_name = CONVERSION_LABEL
#             class_color = CONVERSION_COLOR
#         else: 
#             class_name = classes[gt_label_id]
#             class_color = palette[gt_label_id]
#         if class_name[0] == "_":
#             class_name = class_name[1:]
#         if ignore_background and "background" in class_name:
#             continue
        
        
#         class_color = tuple([color/255 for color in class_color])
#         legend_handles.append(
#             mlines.Line2D(
#                 [], [],
#                 color=class_color,
#                 label=class_name,
#                 linestyle='-',
#                 linewidth=10
#             )
#         )
#     if pred_label_ids:
#         legend_handles.append(
#             mlines.Line2D(
#                 [], [],
#                 color='k',
#                 label="pred not present",
#                 marker='o',
#                 linewidth=0,
#                 markersize=16
#             )
#         )
#     for pred_label_id in pred_label_ids:
        
#         if pred_label_id == CONVERSION_IDX:
#             class_name = CONVERSION_LABEL
#             class_color = CONVERSION_COLOR
#         else:
#             class_name = classes[pred_label_id]
#             class_color = palette[pred_label_id]
#         class_color = tuple([color/255 for color in class_color])
#         legend_handles.append(
#             mlines.Line2D(
#                 [], [],
#                 color=class_color,
#                 label=class_name,
#                 marker='o',
#                 linewidth=0,
#                 markersize=16
#             )
#         )
#     return legend_handles