import pickle
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import math

# 读取训练数据
def read_train_data(filepath):
    train_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            user_id, num_ratings = line.strip().split('|')
            for _ in range(int(num_ratings)):
                item_id, score = f.readline().strip().split()
                train_data[int(user_id)].append((item_id, float(score)))
    return train_data

# 划分训练集和测试集
def split_train_test(train_data, test_size=0.1):
    train_split = defaultdict(list)
    test_split = defaultdict(list)

    for user, ratings in train_data.items():
        train_ratings, test_ratings = train_test_split(ratings, test_size=test_size)
        train_split[user].extend(train_ratings)
        test_split[user].extend(test_ratings)

    return train_split, test_split

# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Reading item data"):
            item_id, attr1, attr2 = line.strip().split('|')
            item_data[item_id] = (attr1, attr2)
    return item_data

# 构建评分矩阵
def build_rating_matrix(train_data):
    users = list(train_data.keys())
    items = list({item for ratings in train_data.values() for item, _ in ratings})
    user_map = {user: i for i, user in enumerate(users)}
    item_map = {item: i for i, item in enumerate(items)}

    num_ratings = sum(len(ratings) for ratings in train_data.values())
    data, row, col = np.zeros(num_ratings), np.zeros(num_ratings, dtype=int), np.zeros(num_ratings, dtype=int)

    idx = 0
    for user, ratings in train_data.items():
        for item, score in ratings:
            data[idx] = score
            row[idx] = user_map[user]
            col[idx] = item_map[item]
            idx += 1

    rating_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return rating_matrix, user_map, item_map

# 计算用户相似度
def calculate_user_similarity(rating_matrix, batch_size=5000):
    num_users = rating_matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))

    for start in tqdm(range(0, num_users, batch_size), desc="Calculating similarity"):
        end = min(start + batch_size, num_users)
        batch_similarity = cosine_similarity(rating_matrix[start:end], rating_matrix)
        similarity_matrix[start:end] = batch_similarity

    return similarity_matrix

# 读取测试数据
def read_test_data(filepath):
    test_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            user_id, num_ratings = line.strip().split('|')
            num_ratings = int(num_ratings)
            for _ in range(num_ratings):
                item_id = f.readline().strip()
                test_data[int(user_id)].append(item_id)
    return test_data

# 进行用户评分预测
def predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_data, top_k=10):
    predictions = defaultdict(list)

    for user, items in tqdm(test_data.items(), desc="Processing users", total=len(test_data)):
        if user not in user_map:
            continue
        user_idx = user_map[user]

        # 获取与当前用户最相似的前top_k个用户的索引
        similar_users = np.argsort(similarity_matrix[user_idx])[::-1][1:top_k + 1]

        for item in items:
            if item not in item_map:
                continue
            item_idx = item_map[item]

            # 收集相似用户对该物品的评分及其相似度
            sim_scores = rating_matrix[similar_users, item_idx].toarray().flatten()
            weights = similarity_matrix[user_idx, similar_users]

            # 只考虑评分和相似度都不为0的情况
            valid_mask = (sim_scores != 0) & (weights != 0)
            if np.any(valid_mask):
                # 计算加权平均评分
                predicted_score = np.dot(sim_scores[valid_mask], weights[valid_mask]) / np.sum(weights[valid_mask])
            else:
                predicted_score = 0  # 如果没有相似用户评分该物品，则预测评分为0

            # 保留4位小数并四舍五入
            predicted_score = round(predicted_score, 4)
            predictions[user].append((item, predicted_score))

    return predictions

# 计算RMSE
def calculate_rmse(predictions, test_split, user_map, item_map):
    error_sum = 0
    count = 0

    for user, items in test_split.items():
        if user not in user_map:
            continue
        user_idx = user_map[user]

        for item, actual_score in items:
            if item not in item_map:
                continue
            item_idx = item_map[item]

            predicted_score = next((score for i, score in predictions[user] if i == item), 0)
            error_sum += (predicted_score - actual_score) ** 2
            count += 1

    rmse = math.sqrt(error_sum / count) if count != 0 else 0
    return rmse

# 写入预测结果
def write_predictions(predictions, filepath):
    with open(filepath, 'w') as f:
        for user, items in predictions.items():
            f.write(f"{user}|{len(items)}\n")
            for item, score in items:
                f.write(f"{item} {score}\n")

# 主流程
print("Starting...")

train_data_path = '../../data/train.txt'
item_data_path = '../../data/itemAttribute.txt'
test_data_path = '../../data/test.txt'
result_path = '../result/result_userCF.txt'
top_k = 500

train_data = read_train_data(train_data_path)
train_split, test_split = split_train_test(train_data, test_size=0.1)

item_data = read_item_data(item_data_path)
rating_matrix, user_map, item_map = build_rating_matrix(train_split)
similarity_matrix = calculate_user_similarity(rating_matrix)

print("Training Finished!")

# 对test_split进行预测并计算RMSE
split_predictions = predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_split, top_k)
rmse = calculate_rmse(split_predictions, test_split, user_map, item_map)
print(f"RMSE: {rmse:.4f}")

# 对test.txt中的所有数据进行预测并写入文件
test_data = read_test_data(test_data_path)
final_predictions = predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_data, top_k)
write_predictions(final_predictions, result_path)

print("Finished!")
