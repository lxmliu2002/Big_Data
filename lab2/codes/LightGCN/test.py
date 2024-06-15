import pickle
import SVD
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


mapper = DatasetMapper()
user_map, item_map = mapper.load_mappings()

test_data = read_test_data('../data/test.txt')

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
