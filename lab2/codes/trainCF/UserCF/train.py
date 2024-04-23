import pickle
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 读取训练数据
def read_train_data(filepath):
    train_data = defaultdict(list)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            user_id, num_ratings = lines[i].strip().split('|')
            i += 1
            for _ in range(int(num_ratings)):
                item_id, score = lines[i].strip().split()
                train_data[int(user_id)].append((item_id, float(score)))
                i += 1
    return train_data

# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Reading item data"):
            item_id, attr1, attr2 = line.strip().split('|')
            item_data[item_id] = (attr1, attr2)
    return item_data

# 构建评分矩阵
def build_rating_matrix(train_data):
    users = list(train_data.keys())
    items = list(set(item for ratings in train_data.values() for item, _ in ratings))
    user_map = {user: i for i, user in enumerate(users)}
    item_map = {item: i for i, item in enumerate(items)}

    data, row, col = [], [], []
    for user, ratings in train_data.items():
        for item, score in ratings:
            data.append(score)
            row.append(user_map[user])
            col.append(item_map[item])
    
    rating_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return rating_matrix, user_map, item_map

# 假设rating_matrix是输入的用户-物品评分矩阵
def calculate_user_similarity(rating_matrix, batch_size=5000):
    num_users = rating_matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))
    
    for start in tqdm(range(0, num_users, batch_size), desc="Calculating similarity"):
        end = min(start + batch_size, num_users)
        batch_similarity = cosine_similarity(rating_matrix[start:end], rating_matrix)
        similarity_matrix[start:end] = batch_similarity
    
    return similarity_matrix

# 保存模型
def save_model(model, matrix, user_map, item_map, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump((model, matrix, user_map, item_map), f)

print("start...")

# 主流程
train_data = read_train_data('../../data/train.txt')
item_data = read_item_data('../../data/itemAttribute.txt')

rating_matrix, user_map, item_map = build_rating_matrix(train_data)

similarity_matrix = calculate_user_similarity(rating_matrix)

# print(similarity_matrix)

save_model(similarity_matrix, rating_matrix, user_map, item_map, '../model/model_userCF.pkl')

print("finished!")
