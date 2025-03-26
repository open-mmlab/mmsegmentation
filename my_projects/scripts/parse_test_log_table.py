import argparse
import json

def save_json(data_dict: dict, dump_file_path: str):
    print(f"saving to : {dump_file_path}")
    with open(dump_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'log_path',
        type=str
    )
    parser.add_argument(
        'save_path',
        type=str
    )
    
    args = parser.parse_args()
    
    return args

def extract_items_from_line(line):
    items = line.split("|")
    return [
        item.strip() for item in items 
            if len(item.strip()) != 0
    ]

def parse_table(log_path: str) -> dict:
    with open(log_path, 'r') as log_file:
        line = log_file.readline()
        # skip until table
        while line:
            # if "Class" in line and "IoU":
                # break
            if "+---" in line:
                line = log_file.readline()
                break 
            line = log_file.readline() 
        if not line:
            print(f"table not found in {log_path}")
            return 
        # start parsing
        
        keys = extract_items_from_line(line=line)
        # split(".") is temp due to false keys in some logs
        keys = [
            key.split(".")[0] for key in keys
        ][1:]
        
        print(keys)
        # line to skip
        line = log_file.readline()
        # start parsing vals
        line = log_file.readline()
        table_dict = {}
        while "+---" not in line:
            vals = extract_items_from_line(line=line)
            class_name = vals[0]
            vals = [
                float(val) for val in vals[1:]
            ]
            table_dict[class_name] = {
                key : val for key, val in zip(keys, vals)
            }            
            line = log_file.readline()
        return table_dict
        
            
            
def main():
    args = parse_args()
    table_dict = parse_table(log_path=args.log_path)
    save_json(
        data_dict=table_dict,
        dump_file_path=args.save_path
    )

if __name__ == '__main__':
    main()