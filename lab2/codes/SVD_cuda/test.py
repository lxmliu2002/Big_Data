# import pickle
import SVD, SVDattr
# import torch
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
model_type = "SVDbias"
# 更改SVD/SVDbias来选择模型，这里参数保持和train中一致
if model_type == "SVD":
    model = SVD.SVDModel(num_users=19835, num_items=456722, latent_dim=10)
    model.load_model(load_dir='model/SVD/model_SVD_6_epochs.pt')
    model.eval()  # 将模型设置为评估模式
    SVD.test(model,user_map, item_map,  test_data, 'result/SVD/predictions.txt')
elif model_type == "SVDbias":
    model = SVD.SVDbiasModel(num_users=19835, num_items=456722, latent_dim=10)
    model.load_model(load_dir='model/SVDbias/model_SVD_28_epochs.pt')
    SVD.test(model,user_map, item_map,  test_data, 'result/SVDbias/predictions.txt')
elif model_type == "SVDattr":
    user_map, item_map, attr1_map, attr2_map = mapper.load_attr_mappings()
    attrmap = SVDattr.map_attr_data(atrr_data, attr1_map, attr2_map, item_map)
    model = SVDattr.SVDattrModel(num_users=19835, num_items=456722,num_attr1=52188,num_attr2=19692, latent_dim=10)
    model.load_model(load_dir='model/SVDattr/model_SVD.pt')
    SVDattr.test(model,user_map, item_map,  test_data, 'result/SVDattr/predictions.txt',attrmap)
