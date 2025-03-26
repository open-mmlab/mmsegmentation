from my_projects.conversion_tests.converters.conversion_dicts import (
    IRL_VISION2IRL_VISION_CAT, HOTS2HOTS_CAT 
)
from mmseg.utils import irl_vision_sim_cat_classes, hots_v1_cat_classes
import numpy as np
from PIL import Image
import os

DATASET_DICT = {
    "HOTS_v1_cat"                 :    HOTS2HOTS_CAT,
    "irl_vision_sim_cat"          :    IRL_VISION2IRL_VISION_CAT       
}


def convert_annotation_label(
    label: np.ndarray, 
    conversion_dict: dict
) -> np.ndarray:
    label_shape = label.shape
    label = label.flatten()
    for label_idx, label_val in enumerate(label):
        label[label_idx] = conversion_dict[label_val]
    return label.reshape(label_shape)


def save_annotation_label(
    label: np.ndarray,
    annotation_file_path: str
) -> None:
    im = Image.fromarray(label.astype(np.uint8))
    im.save(annotation_file_path)

def load_annotation_label(annotation_file_path: str) -> np.ndarray:
    file = Image.open(annotation_file_path)
    return np.array(file)



def traverse_annotation_data(
    annotation_dir_path: str, 
    conversion_dict: dict
) -> None:
    unique_labels = np.asarray([0])
    for ann_phase_dir in os.listdir(annotation_dir_path):
        ann_phase_dir_path = os.path.join(
            annotation_dir_path,
            ann_phase_dir
        )
        for file_name in os.listdir(ann_phase_dir_path):
            annotation_file_path = os.path.join(
                ann_phase_dir_path,
                file_name
            )
            label = load_annotation_label(
                annotation_file_path=annotation_file_path
            )
            unique_labels = np.unique(
                np.concatenate(
                    (unique_labels, np.unique(label))
                )
            )
            label = convert_annotation_label(
                label=label, conversion_dict=conversion_dict
            )
            save_annotation_label(
                label=label, annotation_file_path=annotation_file_path
            )
    print("unique labels: ")
    print(unique_labels)


def main():
    data_path = "/media/ids/Ubuntu files/data"
    annotation_path = "SemanticSegmentation/ann_dir"
    for dataset_name, conversion_dict in DATASET_DICT.items():
        dataset_name = os.path.join(dataset_name)
        annotation_dir_path=os.path.join(
            data_path,
            dataset_name,
            annotation_path
        )
        print(dataset_name)
        traverse_annotation_data(
            annotation_dir_path=annotation_dir_path,
            conversion_dict=conversion_dict
        ) 
    print(f"hots_cat: (len == {len(hots_v1_cat_classes())})")
    for class_name in hots_v1_cat_classes():
        print(class_name)
        
    print(f"irl_cat: (len == {len(irl_vision_sim_cat_classes())})")
    for class_name in irl_vision_sim_cat_classes():
        print(class_name)
           
if __name__ == '__main__':
    main()
    