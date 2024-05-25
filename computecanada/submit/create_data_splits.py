import os
import random

# Path to the folder containing the nc files
folder_path = "/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3"
save_folder_pretrain = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
save_folder_finetune = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
# List all the nc files in the folder
nc_files = [file for file in os.listdir(folder_path) if file.endswith(".nc")]

# Define different splits
# splits = [(0.9,0.1),(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
splits = [(0.95,0.05)]

# Iterate over splits
for split_index, (train_ratio, test_ratio) in enumerate(splits, start=1):
    # Calculate the number of files for train and test sets
    num_train_files = int(len(nc_files) * train_ratio)
    num_test_files = int(len(nc_files) * test_ratio)

    # Randomly shuffle the list of nc files
    random.shuffle(nc_files)

    # Divide files into train and test sets
    train_files = nc_files[:num_train_files]
    test_files = nc_files[num_train_files:num_train_files + num_test_files]

    # Write filenames to train.txt
    with open(os.path.join(save_folder_pretrain, f"pretrain_{int(train_ratio*100)}.txt"), "w") as train_txt:
        for file in train_files:
            train_txt.write(file + "\n")

    # Write filenames to test.txt
    with open(os.path.join(save_folder_finetune, f"finetune_{int(test_ratio*100)}.txt"), "w") as test_txt:
        for file in test_files:
            test_txt.write(file + "\n")

    print(
        f"pretrain_{int(train_ratio*100)}.txt and finetune_{int(test_ratio*100)}.txt files created successfully.")
