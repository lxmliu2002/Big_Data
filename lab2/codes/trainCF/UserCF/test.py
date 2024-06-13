import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm


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


# 写入预测结果
def write_predictions(predictions, filepath):
    with open(filepath, 'w') as f:
        for user, items in predictions.items():
            f.write(f"{user}|{len(items)}\n")  # 根据实际预测数量写入
            for item, score in items:
                f.write(f"{item} {score}\n")


# 超参数
top_k = 500

# 主要流程
test_data = read_test_data('../../data/test.txt')
model, rating_matrix, user_map, item_map = load_model('../model/model_userCF.pkl')
predictions = predict_user_ratings(model, rating_matrix, user_map, item_map, test_data, top_k)
write_predictions(predictions, '../result/result_userCF.txt')

print("finished!")
