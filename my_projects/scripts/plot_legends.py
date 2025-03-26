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
)

dataset_dict = {
    "HOTS"      :       (
                            hots_v1_classes(), 
                            hots_v1_palette()
                        ),
    "HOTS-C"    :       (
                            hots_v1_cat_classes(), 
                            hots_v1_cat_palette()
                        ),
    "SOD"       :       (
                            irl_vision_sim_classes(), 
                            irl_vision_sim_palette()
                        ),
    "SOD-C"     :       (
                            irl_vision_sim_cat_classes(), 
                            irl_vision_sim_cat_palette()
                        ),
    "ARID10"    :       (
                            arid10cat_classes(),
                            arid10cat_palette()
                        ),
    "ARID20"    :       (
                            arid20cat_classes(),
                            arid20cat_palette()
                        ),
    "SODHOTS-C" :       (
                            sodhots_c_classes(),
                            sodhots_c_palette()
                        )  
}


def plot_legend(
    name: str,
    classes,
    palette
):
    
    fig, ax = plt.subplots()

    legend_handles = generate_legend_handles(classes=classes, palette=palette)

    # Add the legend
    ax.legend(handles = legend_handles, ncol=3, loc='center')


    # Remove the axes ticks and labels (optional, for a cleaner look)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False) # Hide border lines
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Or, even simpler to remove everything related to the axes:
    # ax.axis('off')

    # Adjust the figure size if needed to fit the legend nicely
    fig.set_figheight(2)
    fig.set_figwidth(2)
    fig.tight_layout() # Ensures the legend fits within the figure bounds.  Important!

    # Display the figure
    # plt.show()
    save_path = "my_projects/images_plots/legends"
    fig.savefig(
        os.path.join(
            save_path,
            f"{name}_legend.png"
        ),
        bbox_inches='tight'
    )
    

def generate_legend_handles(
    classes,
    palette
):
    legend_handles = []
    for class_name, class_color in zip(classes, palette): 
        class_color = tuple([color/255 for color in class_color])
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color=class_color,
                label=class_name.strip("_"),
                linestyle='-',
                linewidth=10
            )
        )
    return legend_handles


for name, (classes, palette) in dataset_dict.items():
    plot_legend(name, classes=classes, palette=palette)  