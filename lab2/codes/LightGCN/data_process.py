import pickle

class DatasetMapper:
    def __init__(self):
        self.user_map = {}
        self.item_map = {}
        self.attr1_map = {}
        self.attr2_map = {}

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

    def build_attr_mappings(self, attr_data):
        """构建属性映射"""
        attr1_set = set()
        attr2_set = set()
        
        # 遍历attr_data，收集所有的属性值
        for attrs in attr_data.items():
            attr1_set.add(attrs[1][0])
            attr2_set.add(attrs[1][1])
        
        # 构建属性映射
        self.attr1_map = {attr: idx for idx, attr in enumerate(attr1_set)}
        self.attr2_map = {attr: idx for idx, attr in enumerate(attr2_set)}

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

    def save_attr_mappings(self, 
                     filepath_user='index/user_mapping.pkl', 
                     filepath_item='index/item_mapping.pkl',
                     filepath_attr1='index/attr1_mapping.pkl',
                     filepath_attr2='index/attr2_mapping.pkl'):
        """将映射保存到pickle文件，以便高效读取"""
        with open(filepath_user, 'wb') as f:
            pickle.dump(self.user_map, f)
        with open(filepath_item, 'wb') as f:
            pickle.dump(self.item_map, f)
        with open(filepath_attr1, 'wb') as f:
            pickle.dump(self.attr1_map, f)
        with open(filepath_attr2, 'wb') as f:
            pickle.dump(self.attr2_map, f)

    @staticmethod
    def load_attr_mappings(filepath_user='index/user_mapping.pkl', 
                      filepath_item='index/item_mapping.pkl',
                      filepath_attr1='index/attr1_mapping.pkl',
                      filepath_attr2='index/attr2_mapping.pkl'):
        """从pickle文件加载映射"""
        with open(filepath_user, 'rb') as f:
            user_map = pickle.load(f)
        with open(filepath_item, 'rb') as f:
            item_map = pickle.load(f)
        with open(filepath_attr1, 'rb') as f:
            attr1_map = pickle.load(f)
        with open(filepath_attr2, 'rb') as f:
            attr2_map = pickle.load(f)
        return user_map, item_map, attr1_map, attr2_map


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

def read_attribute_data(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            attr1 = int(parts[1]) if parts[1] != 'None' else -1
            attr2 = int(parts[2]) if parts[2] != 'None' else -1
            # 检查并初始化 data[item_id] 为空列表
            if item_id not in data:
                data[item_id] = []
            data[item_id].append(attr1)
            data[item_id].append(attr2)
            
    return data

if __name__ == "__main__":
    train_data = read_train_data('../data/train.txt')
    test_data = read_test_data('../data/test.txt')
    atrr_data = read_attribute_data('../data/itemAttribute.txt')
    
    mapper = DatasetMapper()
    mapper.build_mappings(train_data, test_data)
    mapper.build_attr_mappings(atrr_data)
    mapper.save_attr_mappings()

    print("index build success")