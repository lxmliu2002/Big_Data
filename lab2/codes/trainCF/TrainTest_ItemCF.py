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
                test_data[user_id].append(item_id)
                i += 1
    return test_data

# 读取项目数据
def read_item_data(filepath):
    item_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
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
def calculate_similarity(target_item_id,user_id, user_purchases, rating_matrix, item_map):
    if target_item_id not in item_map:
        return 0

    target_index = item_map[target_item_id]
    similarities = []
    
    for purchase in user_purchases:
        if purchase in item_map:
            purchase_index = item_map[purchase]
            sim = cosine_similarity(rating_matrix[:, purchase_index].toarray().T,
                                    rating_matrix[:, target_index].toarray().T)[0, 0]
            similarities.append((purchase, sim))
    
    # 取前五高相似度的物品
    top_similar_items = heapq.nlargest(10, similarities, key=lambda x: x[1])
    # print(top_similar_items )
    weighted_sum = sum(score * rating_matrix[user_map[int(user_id)], item_map[item]] for item, score in top_similar_items)
    sim_sum = sum(score for _, score in top_similar_items)
    predicted_score = weighted_sum / sim_sum if sim_sum != 0 else 0.0
    return predicted_score


# 预测评分
def predict_ratings(test_data, train_data, rating_matrix, user_map, item_map):
    predictions = defaultdict(list)
    for user_id, item_ids in tqdm(test_data.items(), desc="predictng"):
        user_ratings = train_data[int(user_id)]
        user_purchases = [item for item, _ in user_ratings]

        for item_id in item_ids:
            if item_id not in user_purchases:
                scores = calculate_similarity(item_id, user_id, user_purchases, rating_matrix, item_map)
                scores = round(scores, 4)
                predictions[user_id].append((item_id, scores))

    return predictions


# 写入预测结果
def write_predictions(predictions, filepath):
    with open(filepath, 'w') as f:
        for user, items in predictions.items():
            f.write(f"{user}|6\n")
            for item, score in items:
                f.write(f"{item} {score}\n")

print("starting...")

# 主流程
train_data = read_train_data('../data/train.txt')
item_data = read_item_data('../data/itemAttribute.txt')
test_data = read_test_data('../data/test.txt')
# test_data = read_test_data('../data/test_light.txt')

rating_matrix, user_map, item_map = build_rating_matrix(train_data)

# 以部分测试集为例进行展示
predict_matrix = predict_ratings(test_data, train_data, rating_matrix, user_map, item_map)

write_predictions(predict_matrix, './result/result_itemCF.txt')
# write_predictions(predict_matrix, './result/result_itemCF_light.txt')

print("finished!")
