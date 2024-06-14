import pickle
import random
import numpy as np
import SVD 
import warnings
from data_process import DatasetMapper
warnings.filterwarnings('ignore')


def read_train_data(filepath):
    train_data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            user_id, num_ratings = lines[i].strip().split('|')
            i += 1
            for _ in range(int(num_ratings)):
                item_id, score = lines[i].strip().split()
                train_data.append((int(user_id), int(item_id), float(score)))
                i += 1
    return train_data

def split_train_test(data, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
    random.shuffle(data)
    return data[:train_size], data[train_size:]

# 该划分方法在每个用户集合中随机选一个用户评价记录作为测试集
def split_train_test2(data):
    # 使用字典存储每个用户的评价记录
    user_ratings = {}
    for user_id, item_id, score in data:
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        user_ratings[user_id].append((item_id, score))
    
    # 分割数据集
    train_data = []
    test_data = []
    for user_id, ratings in user_ratings.items():
        if ratings:  # 确保用户有评价记录
            # 随机选择一条作为测试集，剩余的加入训练集
            random_rating = random.choice(ratings)
            test_data.append((user_id,) + random_rating)
            # 移除已选为测试集的评价
            ratings.remove(random_rating)
            train_data.extend([(user_id, item_id, score) for item_id, score in ratings])
    
    return train_data, test_data

# 读取训练数据
train_data = read_train_data('../data/train.txt')
mapper = DatasetMapper()
user_map, item_map = mapper.load_mappings()

#参数
latent_dim = 10
num_epochs= 50
lr=0.05
weight_decay=1e-6
model_type = "SVDbias"
data_divide_method = 2


# 划分训练集和测试集
if data_divide_method == 1:
    train_data, test_data = split_train_test(train_data, train_ratio=0.9)
elif data_divide_method == 2:
    train_data, test_data = split_train_test2(train_data)

train_data = SVD.prepare_train_data(user_map, item_map, train_data)
test_data = SVD.prepare_train_data(user_map, item_map, test_data)

num_users = len(user_map)
num_items = len(item_map)

print(len(train_data))
print(len(test_data))

# 更改SVD/SVDbias来选择模型
if model_type == "SVD":
    model = SVD.SVDModel(num_users, num_items, latent_dim)
elif model_type == "SVDbias":
    model = SVD.SVDbiasModel(num_users, num_items, latent_dim)

SVD.train(model, train_data, test_data, num_epochs, lr, weight_decay)

model.save_model()