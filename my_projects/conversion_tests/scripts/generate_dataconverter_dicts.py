from mmseg.utils import irl_vision_sim_classes, hots_v1_classes
from mmseg.utils import cocostuff_classes, ade_classes
from mmseg.utils import irl_vision_sim_cat_classes, hots_v1_cat_classes
from mmseg.utils import arid20cat_classes, arid20cat_palette
from my_projects.conversion_tests.converters.conversion_dicts import (
    HOTS2HOTS_CAT, IRL_VISION2IRL_VISION_CAT, 
    IRL_VISION_CAT2HOTS_CAT, HOTS_CAT2IRL_VISION_CAT,
    ADE20K2HOTS_CAT_CLASS_NAMES, ARID20CAT2IRL_VISION_CAT_CLASS_NAMES,
    ARID20CAT2HOTS_CAT_CLASS_NAMES
)


print(f"{'#' * 10} ARID20K vs IRL {'#' * 10}")
print("not in arid:")
for irl_name in irl_vision_sim_cat_classes():
    if irl_name not in ARID20CAT2IRL_VISION_CAT_CLASS_NAMES.values():
        print(irl_name)

print("not in irl:")
for arid_name in arid20cat_classes():
    if arid_name not in ARID20CAT2IRL_VISION_CAT_CLASS_NAMES.keys():
        print(arid_name)
        


print(f"{'#' * 10} ARID20K vs HOTS {'#' * 10}")
print("not in arid:")
for hots_name in hots_v1_cat_classes():
    if hots_name not in ARID20CAT2HOTS_CAT_CLASS_NAMES.values():
        print(hots_name)

print("not in hots:")
for arid_name in arid20cat_classes():
    if arid_name not in ARID20CAT2HOTS_CAT_CLASS_NAMES.keys():
        print(arid_name)


print(f"{'#' * 10} HOTS vs IRL {'#' * 10}")
print("not in hots:")
for irl_name in irl_vision_sim_cat_classes():
    if irl_vision_sim_cat_classes().index(irl_name) not in HOTS_CAT2IRL_VISION_CAT.values():
        print(irl_name)

print("not in irl:")
for hots_name in hots_v1_cat_classes():
    if hots_v1_cat_classes().index(hots_name) not in HOTS_CAT2IRL_VISION_CAT.keys():
        print(hots_name)
   
# for irl_name in irl_vision_sim_cat_classes():
#     for arid_name in arid20cat_classes():
#         if (
#             irl_name in arid_name
#             or 
#             arid_name in irl_name
#             or 
#             irl_name == arid_name 
#         ):
#             if arid_name not in ARID20CAT2IRL_VISION_CAT_CLASS_NAMES.keys():
#                 ARID20CAT2IRL_VISION_CAT_CLASS_NAMES[arid_name] = []
#             ARID20CAT2IRL_VISION_CAT_CLASS_NAMES[arid_name].append(
#                 irl_name
#             )

# for arid_name, irl_name in ARID20CAT2IRL_VISION_CAT_CLASS_NAMES.items():
#     if isinstance(irl_name, list) and len(irl_name) == 1:
#         ARID20CAT2IRL_VISION_CAT_CLASS_NAMES[arid_name] = irl_name[0]

# for arid_name, irl_name in ARID20CAT2IRL_VISION_CAT_CLASS_NAMES.items():
#     print(f'   "{arid_name}" : "{irl_name}", ')


# items in ade20k that matter: 
# "box", "table", "book", "computer", "apparel", "bottle",
# "plaything", "bag", "ball"(?), "food", "screen"
# "vase", "tray", "crt screen", "plate", "monitor", "glass"
# ADE20K2HOTS_CAT = {}
# for ade_name, hots_names in ADE20K2HOTS_CAT_CLASS_NAMES.items():
#     ade_idx=ade_classes().index(ade_name)
#     if ade_idx not in ADE20K2HOTS_CAT.keys():
#         ADE20K2HOTS_CAT[ade_idx] = []
#     for hots_name in hots_names:
#         ADE20K2HOTS_CAT[ade_idx].append(
#             hots_v1_cat_classes().index(hots_name)
#         )
        
# print("ADE20K2HOTS_CAT")
# for key, val in ADE20K2HOTS_CAT.items():
#     print(f"{key} : {val},")

# HOTS_CAT2ADE20K = {}
# for ade_idx, hots_idxs in ADE20K2HOTS_CAT.items():
#     for hot_idx in hots_idxs:
#         if hot_idx not in HOTS_CAT2ADE20K.keys():
#             HOTS_CAT2ADE20K[hot_idx] = []
#         HOTS_CAT2ADE20K[hot_idx].append(ade_idx)
        
# print("\nHOTS_CAT2ADE20K")
# for key, val in HOTS_CAT2ADE20K.items():
#     print(f"{key} : {val},")
# hots to hots_simple
#  dict["unique_name"] = [idxs]
# hots_to_cat = {}
# for class_idx, class_name in enumerate(hots_v1_classes()):
#     class_cat_name = class_name.split("_")[0]
#     if "juice_box" in class_name:
#         class_cat_name = "juice_box"
#     if "background" in class_name:
#         class_cat_name = "_background_"
#     if class_cat_name not in hots_to_cat.keys():
#         hots_to_cat[class_cat_name] = []
#     hots_to_cat[class_cat_name].append(class_idx)


# for idx, (key, val) in enumerate(hots_to_cat.items()):
    
#     print(f"cat name {key}, idx {idx}:\n members: {[hots_v1_classes()[item] for item in val]}")

# sum_ = sum(len(val) for key, val in hots_to_cat.items())
# print(f"total classes accounted for: {sum_}")

  
# print("IRL CAT")  
# irl_vision_to_cat = {}
# for class_idx, class_name in enumerate(irl_vision_sim_classes()):
#     class_cat_name = class_name.split("_")[0]
#     if "juice_box" in class_name:
#         class_cat_name = "juice_box"
#     if "background" in class_name:
#         class_cat_name = "_background_"
#     if class_cat_name not in irl_vision_to_cat.keys():
#         irl_vision_to_cat[class_cat_name] = []
#     irl_vision_to_cat[class_cat_name].append(class_idx)


# for idx, (key, val) in enumerate(irl_vision_to_cat.items()):
    
#     print(f"cat name {key}, idx {idx}:\n members: {[irl_vision_sim_classes()[item] for item in val]}")

# sum_ = sum(len(val) for key, val in irl_vision_to_cat.items())
# print(f"total classes accounted for: {sum_}")

    

# print("\nkeys not found in hots: ")
# for idx_irl, (key_irl, val_irl) in enumerate(irl_vision_to_cat.items()):
#     if not key_irl in hots_to_cat.keys(): 
#         print(key_irl)

# print(f"\n keys not found in irl_vision")
# for key, val in hots_to_cat.items():
#     if key not in irl_vision_to_cat.keys():
#         print(key)
    

# print(f"len irl,hots: {len(irl_vision_to_cat)}, {len(hots_to_cat)}")

    
# #### HERE THE DICTS ARE CREATED ####
# hots2hots_cat = {}
# for cat_idx, (cat_name, item_idx_list) in enumerate(hots_to_cat.items()):
#     for item_idx in item_idx_list:
#         hots2hots_cat[item_idx] = cat_idx

# irl_vision2irl_vision_cat = {}
# for cat_idx, (cat_name, item_idx_list) in enumerate(irl_vision_to_cat.items()):
#     for item_idx in item_idx_list:
#         irl_vision2irl_vision_cat[item_idx] = cat_idx

# irl_vision_cat2hots_cat = {}
# for idx_irl_cat, (name_irl_cat, item_idx_list_irl) in enumerate(irl_vision_to_cat.items()):
#     for idx_hots_cat, (name_hots_cat, item_idx_list_hots) in enumerate(hots_to_cat.items()):
#         if name_irl_cat == name_hots_cat:
#             irl_vision_cat2hots_cat[idx_irl_cat] = idx_hots_cat

# hots_cat2irl_vision_cat = {
#     val : key for key, val in irl_vision_cat2hots_cat.items()   
# }
# print("idx map:\n")
# print("HOTS2HOTS_CAT = {")
# for item_idx, cat_idx in hots2hots_cat.items():
#     print(f"\t\t{item_idx}  :  {cat_idx},")
# print("}")

# print("")

# print("IRL_VISION2IRL_VISION_CAT = {")
# for item_idx, cat_idx in irl_vision2irl_vision_cat.items():
#     print(f"\t\t{item_idx}  :  {cat_idx},")
# print("}")

# print("")
# print("IRL_VISION_CAT2HOTS_CAT = {")
# for item_idx, cat_idx in irl_vision_cat2hots_cat.items():
#     print(f"\t\t{item_idx}  :  {cat_idx},")
# print("}")

# print("")
# print("HOTS_CAT2IRL_VISION_CAT = {")
# for item_idx, cat_idx in hots_cat2irl_vision_cat.items():
#     print(f"\t\t{item_idx}  :  {cat_idx},")
# print("}")