import json
import os
import argparse


def find_json_file(n_objects_dir_path):
    for root, dirs, files in os.walk(os.path.join(n_objects_dir_path)):
        json_file = [file for file in files if ".json" in file]
        if json_file:
            return os.path.join(root, json_file[0])
    return None

def collect_json_files(clutter_path):
    clutter_dict = {}
    for n_objects_dir in os.listdir(clutter_path):
        json_file_path = find_json_file(
            n_objects_dir_path=os.path.join(
                clutter_path,
                n_objects_dir
            )
        )
        if json_file_path is None:
            continue
        with open(json_file_path, 'r') as json_file:
            clutter_dict[int(n_objects_dir)] = json.load(json_file)
    clutter_dump_path = os.path.join(clutter_path, "clutter_data.json")
    with open(clutter_dump_path, 'w') as dump_file:
        json.dump(clutter_dict, dump_file, indent=4)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'clutter_path', 
        type=str
    )
    args = parser.parse_args()
    collect_json_files(args.clutter_path)
    
if __name__ == '__main__':
    main()
