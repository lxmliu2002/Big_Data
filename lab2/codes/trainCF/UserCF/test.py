import pickle
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

# 读取测试数据
def read_test_data(filepath):
    test_data = defaultdict(list)
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            user_id, num_ratings = lines[i].strip().split('|')
            i += 1
            for _ in range(int(num_ratings)):
                item_id = lines[i].strip()
                test_data[int(user_id)].append(item_id)
                i += 1
    return test_data

# 加载模型
def load_model(filepath):
    with open(filepath, 'rb') as f:
        model, rating_matrix, user_map, item_map = pickle.load(f)
    return model, rating_matrix, user_map, item_map

# 进行用户评分预测
def predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_data, top_k=10):
    predictions = defaultdict(list)
    
    for user, items in tqdm(test_data.items(), desc="Processing users", total=len(test_data)):
        if user not in user_map:
            continue
        user_idx = user_map[user]
        
        # 获取与当前用户最相似的前top_k个用户的索引
        similar_users = np.argsort(similarity_matrix[user_idx])[::-1][1:top_k+1]
        
        for item in items:
            if item not in item_map:
                continue
            item_idx = item_map[item]
            
            # 收集相似用户对该物品的评分及其相似度
            sim_scores = []
            weights = []
            for sim_user_idx in similar_users:
                sim_score = rating_matrix[sim_user_idx, item_idx]
                weight_score = similarity_matrix[user_idx, sim_user_idx]
                if sim_score != 0 and weight_score !=0:  # 只考虑评分不为0的情况
                    sim_scores.append(sim_score)
                    weights.append(weight_score)
            
            if weights:
                # 计算加权平均评分
                predicted_score = np.dot(sim_scores, weights) / np.sum(weights)
            else:
                predicted_score = 0  # 如果没有相似用户评分该物品，则预测评分为0

            # 保留4位小数并四舍五入
            predicted_score = round(predicted_score, 4)
            predictions[user].append((item, predicted_score))
    
    return predictions

# 写入预测结果
def write_predictions(predictions, filepath):
    with open(filepath, 'w') as f:
        for user, items in predictions.items():
            f.write(f"{user}|6\n")  # 这里写死了每个用户6个评分，可能需要根据实际项目调整
            for item, score in items:
                f.write(f"{item} {score}\n")

# 超参数
top_k = 500

test_data = read_test_data('../../data/test.txt')
model, rating_matrix, user_map, item_map = load_model('../model/model_userCF.pkl')
predictions = predict_user_ratings(model, rating_matrix, user_map, item_map, test_data, top_k)
write_predictions(predictions, '../result/result_userCF.txt')

print("finished!")
