import pickle
import NerualCF
import torch
import warnings
from data_process import DatasetMapper
warnings.filterwarnings('ignore')

# 读取测试数据
def read_test_data(filepath):
    test_data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            user_id, num_ratings = lines[i].strip().split('|')
            i += 1
            for _ in range(int(num_ratings)):
                item_id = lines[i].strip()
                test_data.append((int(user_id),int(item_id)))
                i += 1
    return test_data

def read_attribute_data(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            attr1 = int(parts[1]) if parts[1] != 'None' else -1
            attr2 = int(parts[2]) if parts[2] != 'None' else -1
            # 检查并初始化 data[item_id] 为空列表
            if item_id not in data:
                data[item_id] = []
            data[item_id].append(attr1)
            data[item_id].append(attr2)
            
    return data


mapper = DatasetMapper()
user_map, item_map = mapper.load_mappings()

test_data = read_test_data('../data/test.txt')

atrr_data = read_attribute_data('../data/itemAttribute.txt')

# 加载模型
user_map, item_map, attr1_map, attr2_map = mapper.load_attr_mappings()
attrmap = NerualCF.map_attr_data(atrr_data, attr1_map, attr2_map, item_map)
model = NerualCF.NeuMF(num_users=19835, num_items=456722,num_attr1=52188,num_attr2=19692)
model.load_model(load_dir='model/NeuMF/model_NeuMF_3_epochs.pt')
NerualCF.test(model,user_map, item_map,  test_data, 'result/predictions.txt',attrmap)
