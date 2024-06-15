import heapq
from collections import defaultdict
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# 读取训练数据
def read_train_data(filepath):
    train_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            user_id, num_ratings = line.strip().split('|')
            num_ratings = int(num_ratings)
            for _ in range(num_ratings):
                item_id, score = next(f).strip().split()
                train_data[int(user_id)].append((item_id, float(score)))
    return train_data


# 读取测试数据
def read_test_data(filepath):
    test_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            user_id, num_ratings = line.strip().split('|')
            num_ratings = int(num_ratings)
            for _ in range(num_ratings):
                item_id = next(f).strip()
                test_data[user_id].append(item_id)
    return test_data


# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        for line in f:
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

    rating_matrix = csc_matrix((data, (row, col)), shape=(len(users), len(items)))
    return rating_matrix, user_map, item_map


# 计算目标物品与用户购买过的物品的相似度
def calculate_similarity(target_item_id, user_id, user_purchases, rating_matrix, item_map, user_map):
    if target_item_id not in item_map:
        return 0.0

    target_index = item_map[target_item_id]
    similarities = []

    target_vector = rating_matrix[:, target_index].toarray().T
    for purchase in user_purchases:
        if purchase in item_map:
            purchase_index = item_map[purchase]
            purchase_vector = rating_matrix[:, purchase_index].toarray().T
            sim = cosine_similarity(purchase_vector, target_vector)[0, 0]
            similarities.append((purchase, sim))

    # 取前十高相似度的物品
    top_similar_items = heapq.nlargest(10, similarities, key=lambda x: x[1])
    weighted_sum = sum(
        score * rating_matrix[user_map[int(user_id)], item_map[item]] for item, score in top_similar_items)
    sim_sum = sum(score for _, score in top_similar_items)
    predicted_score = weighted_sum / sim_sum if sim_sum != 0 else 0.0
    return round(predicted_score, 4)


# 预测评分
def predict_ratings(test_data, train_data, rating_matrix, user_map, item_map):
    predictions = defaultdict(list)
    for user_id, item_ids in tqdm(test_data.items(), desc="predicting"):
        user_ratings = train_data[int(user_id)]
        user_purchases = [item for item, _ in user_ratings]
        for item_id in item_ids:
            if item_id not in user_purchases:
                score = calculate_similarity(item_id, user_id, user_purchases, rating_matrix, item_map, user_map)
                predictions[user_id].append((item_id, score))
    return predictions


# 写入预测结果
def write_predictions(predictions, filepath):
    with open(filepath, 'w') as f:
        for user, items in predictions.items():
            f.write(f"{user}|{len(items)}\n")
            for item, score in items:
                f.write(f"{item} {score}\n")


def calculate_rmse(predictions, train_data):
    mse = np.mean([(train_data[user][item] - score)**2 
                    for user, items in predictions.items() 
                    for item, score in items if item in train_data[user]])
    return np.sqrt(mse)

print("starting...")

# 主流程
train_data = read_train_data('../data/train.txt')
item_data = read_item_data('../data/itemAttribute.txt')
test_data = read_test_data('../data/test.txt')

rating_matrix, user_map, item_map = build_rating_matrix(train_data)

# 以部分测试集为例进行展示
predict_matrix = predict_ratings(test_data, train_data, rating_matrix, user_map, item_map)

write_predictions(predict_matrix, './result/result_itemCF.txt')

rmse = calculate_rmse(predict_matrix, train_data)
print(f"RMSE: {rmse:.4f}")

print("finished!")
