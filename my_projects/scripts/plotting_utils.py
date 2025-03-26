
from matplotlib import rcParams, rcParamsDefault

double_col_readable = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   25,
    'axes.titlesize'    :   25,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

scenes_plot_global = {
    'figure.figsize'    :   (16, 8),
    'legend.fontsize'   :   30,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   28,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

scenes_plot_global_per_scene_type = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   30,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   20,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}
scenes_plot_model = {
    'figure.figsize'    :   (18, 7),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   30,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

clutter_plot = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   8
}
LEARNING_PLOT_PARAMS = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   8
}
TRADEOFF_PLOT_PARAMS = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   8,
    'lines.markersize'  :   30
}
PRED_VISUALIZATION_FIGURE_PARAMS_2ROWS = {
    'figure.figsize'    :   (16, 8),
    'figure.constrained_layout.use' : True,
    'legend.fontsize'   :   20,
    # 'axes.labelsize'    :   30,
    'axes.titlesize'    :   25,
    # 'xtick.labelsize'   :   30,
    # 'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}
PRED_VISUALIZATION_FIGURE_PARAMS_3ROWS = {
    'figure.figsize'    :   (10, 6),
    'figure.constrained_layout.use' : True,
    'legend.fontsize'   :   20,
    # 'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    # 'xtick.labelsize'   :   30,
    # 'ytick.labelsize'   :   25,
    'lines.linewidth'   :   4
}
PRED_VISUALIZATION_FIGURE_PARAMS = {
    'figure.figsize'    :   (16, 4),
    'figure.constrained_layout.use' : True,
    'legend.fontsize'   :   20,
    # 'axes.labelsize'    :   30,
    'axes.titlesize'    :   12,
    # 'xtick.labelsize'   :   30,
    # 'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}
MODEL_COLORS = {
    "BiSeNet"       :           'tab:blue', 
    "Mask2Former"   :           'tab:orange', 
    "MaskFormer"    :           'tab:green', 
    "SegFormer"     :           'tab:red', 
    "SegNeXt"       :           'tab:purple'
}

def set_params(param_dict=None):
    if param_dict is None:
        param_dict = double_col_readable
    param_dict_ = {}
    for key, val in param_dict.items():
        if key in rcParams.keys():
            param_dict_[key] = val
        else:
            print(f"key {key} not in rcParams")
    rcParams.update(
        param_dict_
    )
    
def reset_params():
    rcParams.update(rcParamsDefault)
    
# def fix_metric_name(metric_name):
#     return metric_name.split(".")[0]

METRIC_KEY_MAP = {
    "mPr@50"        :    "mPr@50",
    "mPr@60"        :    "mPr@60",
    "mPr@70"        :    "mPr@70",
    "mPr@80"        :    "mPr@80",
    "mPr@90"        :    "mPr@90",
    "mPr@50.0"      :    "mPr@50",
    "mPr@60.0"      :    "mPr@60",
    "mPr@70.0"      :    "mPr@70",
    "mPr@80.0"      :    "mPr@80",
    "mPr@90.0"      :    "mPr@90",  
    "mIoU"          :    "mIoU",
    "average_fps"   :    "FPS",
    "average_mem"   :    "Mem (MB)",
    "FPS"           :    "FPS",
    "Mem (MB)"      :    "Mem (MB)"  
}

def map_metric_key_strict(key):
    if key in METRIC_KEY_MAP.keys():
        return METRIC_KEY_MAP[key]
    return False 

def map_metric_key(key):
    if key in METRIC_KEY_MAP.keys():
        return METRIC_KEY_MAP[key]
    return key
    # return key.split(".")[0]


METRIC_KEY_MAP_PLOT = {
    "mPr@50"      :    "$\mathregular{mPr_{50}}$",
    "mPr@60"      :    "$\mathregular{mPr_{60}}$",
    "mPr@70"      :    "$\mathregular{mPr_{70}}$",
    "mPr@80"      :    "$\mathregular{mPr_{80}}$",
    "mPr@90"      :    "$\mathregular{mPr_{90}}$",
    "mPr@50.0"    :    "$\mathregular{mPr_{50}}$",
    "mPr@60.0"    :    "$\mathregular{mPr_{60}}$",
    "mPr@70.0"    :    "$\mathregular{mPr_{70}}$",
    "mPr@80.0"    :    "$\mathregular{mPr_{80}}$",
    "mPr@90.0"    :    "$\mathregular{mPr_{90}}$"     
}

def map_metric_key_plot(key):
    if key in METRIC_KEY_MAP_PLOT.keys():
        return METRIC_KEY_MAP_PLOT[key]
    return key

MODEL_NAME_MAP = {
    "bisenet"       :       "BiSeNet",
    "bisenetv1"     :       "BiSeNet",
    "mask2former"   :       "Mask2Former",
    "maskformer"    :       "MaskFormer",
    "segformer"     :       "SegFormer",
    "segnext"       :       "SegNeXt",
}

def map_model_name(model_name):
    if model_name.lower() in MODEL_NAME_MAP.keys():
        return MODEL_NAME_MAP[model_name.lower()]
    return model_name

DATASET_NAME_MAP = {
    "hots"              :           "HOTS",
    "hots cat"          :           "HOTS-C",
    "irl vision"        :           "SOD",
    "irl vision cat"    :           "SOD-C",
    "arid20"            :           "ARID20",
    "arid10"            :           "ARID10",
    "sodhots"           :           "SODHOTS-C",
    "sodhots-c"         :           "SODHOTS-C",
    "HOTS"              :           "HOTS",
    "HOTS-C"            :           "HOTS-C",
    "SOD"               :           "SOD",
    "SOD-C"             :           "SOD-C",
    "ARID20"            :           "ARID20",
    "ARID10"            :           "ARID10",
    "SODHOTS-C"         :           "SODHOTS-C"
}

def map_dataset_name(ds_name):
    if ds_name in DATASET_NAME_MAP.keys():
        return DATASET_NAME_MAP[ds_name]
    ds_name_ = ds_name.lower()
    if ds_name_ in DATASET_NAME_MAP.keys():
        return DATASET_NAME_MAP[ds_name_]
    if "sodhots" in ds_name_:
        return DATASET_NAME_MAP["sodhots"]
    if "hots" in ds_name_:
        if "cat" in ds_name_ or "c" in ds_name_:
            return DATASET_NAME_MAP["hots cat"]
        else:
            return DATASET_NAME_MAP["hots"]
    if "irl" in ds_name_:
        if "cat" in ds_name_:
            
            return DATASET_NAME_MAP["irl vision cat"]
        else:
            return DATASET_NAME_MAP["irl vision"]
    if "arid20" in ds_name_:
        return DATASET_NAME_MAP["arid20"]
    
    if "arid10" in ds_name_:
        return DATASET_NAME_MAP["arid10"]
    return ds_name