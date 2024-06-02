<<<<<<< HEAD
import pickle
import random
import numpy as np
import SVD 

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


# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item_id, attr1, attr2 = line.strip().split('|')
            item_data[item_id] = (attr1, attr2)
    return item_data


def build_index(train_data):
    users = set(user for user, _, _ in train_data)
    items = set(item for _, item, _ in train_data)
    user_map = {user: i for i, user in enumerate(users)}
    item_map = {item: i for i, item in enumerate(items)}

    return user_map, item_map

def split_train_test(data, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
    random.shuffle(data)
    return data[:train_size], data[train_size:]

# 读取训练数据
train_data = read_train_data('../data/train.txt')
item_data = read_item_data('../data/itemAttribute.txt')

user_map, item_map = build_index(train_data)

# 划分训练集和测试集
train_data, test_data = split_train_test(train_data, train_ratio=0.9)

#参数
latent_dim = 10
num_epochs=20
lr=0.05
weight_decay=1e-6

model = SVD.SVDModel(user_map, item_map, latent_dim)

SVD.train(model, train_data, test_data, num_epochs=5, lr=0.01, weight_decay=1e-6)

=======
import pickle
import random
import numpy as np
import SVD 

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


# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item_id, attr1, attr2 = line.strip().split('|')
            item_data[item_id] = (attr1, attr2)
    return item_data


def build_index(train_data):
    users = set(user for user, _, _ in train_data)
    items = set(item for _, item, _ in train_data)
    user_map = {user: i for i, user in enumerate(users)}
    item_map = {item: i for i, item in enumerate(items)}

    return user_map, item_map

def split_train_test(data, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
    random.shuffle(data)
    return data[:train_size], data[train_size:]

# 读取训练数据
train_data = read_train_data('../data/train.txt')
item_data = read_item_data('../data/itemAttribute.txt')

user_map, item_map = build_index(train_data)

# 划分训练集和测试集
train_data, test_data = split_train_test(train_data, train_ratio=0.9)

#参数
latent_dim = 10
num_epochs=20
lr=0.05
weight_decay=1e-6

model = SVD.SVDModel(user_map, item_map, latent_dim)

SVD.train(model, train_data, test_data, num_epochs=5, lr=0.01, weight_decay=1e-6)

>>>>>>> c835e526528528651acdd235d7c009149cef1cca
model.save_model()