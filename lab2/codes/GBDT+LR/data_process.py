import pickle
import random
import numpy as np

class DatasetMapper:
    def __init__(self):
        self.user_map = {}
        self.item_map = {}

    def build_mappings(self, train_data, test_data):
        """从训练集和测试集数据中构建user_map和item_map"""
        train_users = {user for user, _, _ in train_data}
        test_users = {user for user, _ in test_data}
        all_users = train_users.union(test_users)
        
        train_items = {item for _, item, _ in train_data}
        test_items = {item for _, item in test_data}
        all_items = train_items.union(test_items)
        
        self.user_map = {user: idx for idx, user in enumerate(all_users)}
        self.item_map = {item: idx for idx, item in enumerate(all_items)}

    def save_mappings(self, filepath_user='index/user_mapping.pkl', filepath_item='index/item_mapping.pkl'):
        """将映射保存到pickle文件，以便高效读取"""
        with open(filepath_user, 'wb') as f:
            pickle.dump(self.user_map, f)
        with open(filepath_item, 'wb') as f:
            pickle.dump(self.item_map, f)

    @staticmethod
    def load_mappings(filepath_user='index/user_mapping.pkl', filepath_item='index/item_mapping.pkl'):
        """从pickle文件加载映射"""
        with open(filepath_user, 'rb') as f:
            user_map = pickle.load(f)
        with open(filepath_item, 'rb') as f:
            item_map = pickle.load(f)
        return user_map, item_map

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


if __name__ == "__main__":
    train_data = read_train_data('../data/train.txt')
    test_data = read_test_data('../data/test.txt')
    
    mapper = DatasetMapper()
    mapper.build_mappings(train_data, test_data)
    mapper.save_mappings()

    print("index build success")