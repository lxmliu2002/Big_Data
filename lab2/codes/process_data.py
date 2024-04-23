import pickle
import numpy as np

def set_idx(data_path):
    nodes = set()
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            line_index = 0
            while line_index < len(lines):
                try:
                    line = lines[line_index].strip()
                    if line == "" or '|' not in line:
                        line_index += 1
                        continue
                    _, items_cnt = int(line.split('|'))
                    for _ in range(items_cnt):
                        item_line = lines[line_index].strip().split()
                        if not item_line:
                            raise ValueError("Unexpected end of file")
                        item_id = int(item_line[0])
                        nodes.add(item_id)
                        line_index += 1
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return {}
    nodes = sorted(nodes)
    node_idx = {}
    for idx, node in enumerate(nodes):
        node_idx[node] = idx
    
    return node_idx




if __name__ == '__main__':
    print("Processing data...")
    train_data_path = "./data/train.txt"
    test_data_path = "./data/test.txt"
    attribute_data_path = "./data/itemAttribute.txt"
    
    node_idx = set_idx(train_data_path)
    