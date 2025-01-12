import numpy as np
from .conversion_dicts import (
    SOURCE_TARGET_MAP as CONV_MAP,
    DATASET_CLASSES as GET_CLASSES,
    DATASET_PALETTE as GET_PALETTE
)
import torch

# this is very hard coded: converting only works when 1 to 1
# ensure test to target is always set2set_cat
class DatasetConverter:
    def __init__(
        self, 
        test_dataset: str = "HOTS",
        output_dataset: str = "ADE20K",
        target_dataset: str = "HOTS_CAT",
        unknown_idx: int = 255
    ) -> None:
        self.test_dataset = test_dataset
        self.output_dataset = output_dataset
        self.target_dataset = target_dataset
        self.unknown_idx = unknown_idx
        self.gt_conversions = CONV_MAP[test_dataset][target_dataset]
        self.pred_conversions = CONV_MAP[output_dataset][target_dataset]

        self.gt_conv_direct = DatasetConverter.conversions_are_direct(
            conversions=self.gt_conversions
        )
        self.pred_conv_direct = DatasetConverter.conversions_are_direct(
            conversions=self.pred_conversions
        )
        
        
    
    def convert_labels(
        self,
        pred_label,
        gt_label
    ):
        
        gt_new = self.convert_gt_label(label=gt_label)
        pred_new = self.convert_pred_label(
            pred_label=pred_label, 
            gt_label_converted=gt_new
        )
        return pred_new, gt_new
        
    def get_intersect_and_union(
        self,
        pred_label,
        gt_label
    ):
        
        pred_new, gt_new = self.convert_labels(
            pred_label=pred_label,
            gt_label=gt_label
        )
        assert len(gt_label) == len(pred_label), \
            print(f"invalid gt and pred len: \
                {len(gt_label)} vs {len(pred_label)} respectively")
            
        return self.get_areas(gt_converted=gt_new, pred_converted=pred_new)
    
    def get_areas(self, gt_converted, pred_converted):
        if self.gt_conv_direct and self.pred_conv_direct:
            return self.get_areas_direct(
                gt_converted=gt_converted,
                pred_converted=pred_converted
            )
        # area_intersect is of shape (n_classes_target_dataset)
        area_intersect = np.zeros(
            len(GET_CLASSES[self.target_dataset]())
        ).astype(np.uint16)
        area_gt = np.zeros_like(area_intersect)
        area_pred_label = np.zeros_like(area_intersect)
        
        for (gt_val, pred_val) in zip(gt_converted, pred_converted):
            if pred_val == self.unknown_idx and gt_val == self.unknown_idx:
                continue
            if gt_val != self.unknown_idx:
                area_gt[gt_val] += 1
            if type(pred_val) is list:
                for pred_val_idx in pred_val:
                    # TODO pred vals are unique but this could be extreme
                    if gt_val == pred_val_idx and gt_val != self.unknown_idx:
                        area_intersect[gt_val] += 1
                    # TODO devided due to it being a list and it could be either
                        area_pred_label[pred_val_idx] += 1                 
            else:
                if pred_val == gt_val:
                    area_intersect[gt_val] += 1
                if pred_val != self.unknown_idx:
                    area_pred_label[pred_val] += 1
        area_union = area_pred_label + area_gt - area_intersect
       
        return area_intersect, area_union, area_pred_label, area_gt
    
    
    def get_areas_direct(self, gt_converted, pred_converted):
        
        area_intersect = np.zeros(
            len(GET_CLASSES[self.target_dataset]())
        ).astype(np.uint16)
        area_gt = np.zeros_like(area_intersect)
        area_pred_label = np.zeros_like(area_intersect)
        
        for (gt_val, pred_val) in zip(gt_converted, pred_converted):
            if pred_val == self.unknown_idx and gt_val == self.unknown_idx:
                continue
            if gt_val != self.unknown_idx:
                area_gt[gt_val] += 1
            if pred_val == gt_val:
                area_intersect[gt_val] += 1
            if pred_val != self.unknown_idx:
                area_pred_label[pred_val] += 1
        
        area_union = area_pred_label + area_gt - area_intersect
        return area_intersect, area_union, area_pred_label, area_gt
    
    def convert_direct_label(
        self, 
        label: np.ndarray, 
        conversions: list
    ) -> np.ndarray:
        new_label = np.zeros_like(label)
        for idx, value in enumerate(label):
            val = value
            for conversion in conversions:
                if type(val) is list:
                    print(f"dataconverter.convert_gt_label: \
                            dict value should not be a list")
                if val in conversion.keys():
                    val = conversion[val]
                else:
                    val = self.unknown_idx
            new_label[idx] = val
        return new_label
                
            
         
    # always known_dataset to known_dataset_cat (unless cat dataset)
    def convert_gt_label(
        self, 
        label: np.ndarray, 
    ) -> np.ndarray:
        label_shape = label.shape
        if type(label) is torch.Tensor:
            label = np.asarray(label.cpu())
        label = label.flatten()
        if self.gt_conv_direct:
            return self.convert_direct_label(
                label=label,
                conversions=self.gt_conversions
            ).reshape(label_shape)
    
        return self.convert_any_label(
            label=label,
            conversions=self.gt_conversions 
        ) 
    
    def convert_pred_label(
        self, 
        pred_label: np.ndarray,
        gt_label_converted: np.ndarray = None
    ):
        label_shape = pred_label.shape
        if type(pred_label) is torch.Tensor:
            pred_label = np.asarray(pred_label.cpu())
        pred_label = pred_label.flatten()
        if self.pred_conv_direct:
            return self.convert_direct_label(
                label=pred_label,
                conversions=self.pred_conversions
            ).reshape(label_shape)
        if gt_label_converted is None:
            return self.convert_any_label_random_choice(
                label=pred_label,
                conversions=self.pred_conversions
            ).reshape(label_shape)
        pred_label_tmp = self.convert_any_label(
            label=pred_label,
            conversions=self.pred_conversions
        )
        return self.convert_pred_label_list(
            pred_label=pred_label_tmp,
            gt_label_converted=gt_label_converted
        ).reshape(label_shape)
        
    def convert_pred_label_rand_choice(
        self,
        label: np.ndarray
    ):
        return self.convert_any_label_random_choice(
            label=label, 
            conversions=self.pred_conversions
        )
    
    def convert_pred_label_list(
        self, 
        pred_label,
        gt_label_converted: np.ndarray
    ):
        if type(pred_label) is torch.Tensor:
            pred_label = np.asarray(pred_label.cpu())
        new_pred_label = np.zeros_like(gt_label_converted)
        for label_idx, (pred_val, gt_val) in enumerate(
            zip(pred_label, gt_label_converted)
        ):
            if isinstance(pred_val, (list, np.ndarray)):
                if len(pred_val) == 1:
                    new_pred_label[label_idx] = pred_val[0]
                    continue
                for pred_val_item in pred_val:
                    if isinstance(gt_val, (list, np.ndarray)):
                        if len(gt_val) == 1:   
                            new_pred_label[label_idx] = gt_val[0]
                            continue
                        if pred_val_item in gt_val:
                            new_pred_label[label_idx] = pred_val_item
                            continue
                    elif gt_val == pred_val_item and gt_val != self.unknown_idx:
                        new_pred_label[label_idx] = pred_val_item
                        break
            elif isinstance(gt_val, (list, np.ndarray)):
                if len(gt_val) == 1:   
                    new_pred_label[label_idx] = gt_val[0]
                    continue
                if pred_val in gt_val:
                    new_pred_label[label_idx] = pred_val
                    continue
            elif pred_val == gt_val:
                new_pred_label[label_idx] = pred_val
            else:
                new_pred_label[label_idx] = self.unknown_idx
        return new_pred_label
            
        
        
                    
        
    def convert_any_label(self, label: np.ndarray, conversions: list):
        if type(label) is torch.Tensor:
            label = np.asarray(label.cpu())
        new_label = list(np.zeros_like(label))
        for idx, value in enumerate(label):
            # ititial val is not a list bc label is straight from pred
            val = value
            for conversion in conversions:
                # converting
                if val in conversion.keys():
                    val = conversion[val]
                    
                elif type(val) is list and len(val) == 1:
                    val = conversion[val[0]] 
                elif type(val) is list and len(val) > 1:
                    converted_vals = []
                    for val_item in val:
                        if val_item in conversion.keys():
                            converted_vals.append(conversion[val_item])
                    val = converted_vals if converted_vals else self.unknown_idx
                else:
                    val = self.unknown_idx
            new_label[idx] = val
        return new_label
    
    def convert_any_label_random_choice(
        self, 
        label: np.ndarray, 
        conversions: list
    ) -> np.ndarray:
        if type(label) is torch.Tensor:
            label = np.asarray(label.cpu())
        new_label = np.zeros_like(label)
        # print(f"unique label: {np.unique(label)}")
        for idx, value in enumerate(label):
            val = value
            for conv_idx, conversion in enumerate(conversions):
                # print(f"conv idx: {conv_idx}")
                 
                if type(val) is list:
                    val = self.get_random_converted_val(
                        val=val, 
                        conversions=conversions,
                        conv_idx=conv_idx
                    )
                    # print(f"val after choice: {val}")
                if val in conversion.keys():
                    # print(f"in keys: {val}")
                    val = conversion[val]
                    
                    # print(f" val converted: {val}")
                else:
                    # print(f"unknown: {val}")
                    val = self.unknown_idx
            if type(val) is list:
                val = self.get_random_converted_val(
                    val=val, 
                    conversions=conversions,
                    conv_idx=conv_idx
                )
            # print(f"val assigned at idx: {val} at {idx}")
            new_label[idx] = val
        return new_label
    
    def get_random_converted_val(self, val, conversions, conv_idx):
        converted_vals = []
        for val_ in val:
            if val_ in conversions[conv_idx].keys():
                conv_ = conversions[conv_idx][val_]
                if isinstance(conv_, list): 
                    for conv_item in conv_:
                        converted_vals.append(conv_item) 
                else:
                    converted_vals.append(conv_)
        
        # check forward
        if conv_idx < len(conversions) - 1:
            converted_vals = [
                c_val for c_val in converted_vals
                if c_val in conversions[conv_idx + 1].keys()
            ]
        # print(f"val is list: val, converted = {val}, {converted_vals}")
        return np.random.choice(converted_vals) if converted_vals else self.unknown_idx

    @staticmethod
    def datasets_are_direct_map(
        test_dataset,
        output_dataset,
        target_dataset
    ):
        gt_conversions = CONV_MAP[test_dataset][target_dataset]
        
        pred_conversions = CONV_MAP[output_dataset][target_dataset]
        
        return (
            DatasetConverter.conversions_are_direct(
                conversions=gt_conversions
            )
            and 
            DatasetConverter.conversions_are_direct(
                conversions=pred_conversions
            )
        )

    @staticmethod
    def conversions_are_direct(
        conversions
    ):
        for conversion in conversions:
            for key, val in conversion.items():
                if isinstance(key, list) or isinstance(val, list):
                    return False
        return True
    
    @staticmethod
    def get_class_names(dataset_name):
        return GET_CLASSES[dataset_name]()
    
    @staticmethod
    def get_palette(dataset_name):
        return GET_PALETTE[dataset_name]()
    
    @staticmethod
    def get_dataset(dataset_name):
        return DATASETS[dataset_name]
    
    
# def test():
#     gt_label = np.zeros(shape=30)
#     for idx, val in enumerate(gt_label):
#         gt_label[idx] = np.random.choice(list(ADE20K2HOTS_CAT.keys()))
#     print(f"gt_label:\n{gt_label}")
#     ds_convert = DatasetConverter()
#     conversions = CONV_MAP["ADE20K"]["HOTS_CAT"]
#     new_gt = ds_convert.convert_label(label=gt_label, conversions=conversions)
#     print(f"new gt:\n{new_gt}")
#     print(f'type index: {type(CONV_MAP["ADE20K"]["HOTS_CAT"][0][75])}')

# # test()