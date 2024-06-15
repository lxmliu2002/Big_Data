import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm
import data_process
import data_analysis
import SVD

# 假设 SVDbiasModel 已经定义在 SVD 模块中

# 准备用于 GBDT+LR 模型的训练数据
def prepare_gbdt_lr_data(user_map, item_map, data, item_attributes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long).to(device)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long).to(device)
    
    # 加载训练好的 SVD 模型
    svd_model = SVD.SVDbiasModel(num_users=len(user_map), num_items=len(item_map), latent_dim=10)
    svd_model.load_model('../trainSVD_cuda/model/SVDbias/model_SVDbias_method2_best.pt')
    svd_model.eval().to(device)
    
    # 获取用户和物品的 embedding
    with torch.no_grad():
        user_embeddings = svd_model.user_factors(users).detach().cpu().numpy()
        item_embeddings = svd_model.item_factors(items).detach().cpu().numpy()
    
    # 构建物品属性特征
    item_features = []
    for item_id, attr1, attr2 in item_attributes:
        item_features.append([item_id, attr1, attr2])
    
    # 合并 embedding 和物品属性特征
    X = []
    y = []
    for i, (u, item, r) in enumerate(data):
        user_idx = user_map[u]
        item_idx = item_map[item]
        user_emb = user_embeddings[i]
        item_emb = item_embeddings[i]
        item_attr = item_features[item_idx]
        combined_features = list(user_emb) + list(item_emb) + item_attr
        X.append(combined_features)
        y.append(r)
    
    return X, y

# 训练 GBDT+LR 模型的函数
def train_gbdt_lr(X_train, y_train):
    # 初始化 LightGBM GBDT 模型
    gbdt_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'device': 'cpu' 
    }
    print("开始训练 GBDT 模型...")
    gbdt_model = lgb.LGBMRegressor(**gbdt_params)
    gbdt_model.fit(X_train, y_train)
    print("GBDT 模型训练完成")
    
    # 使用 GBDT 输出作为 LR 的特征
    X_gbdt = gbdt_model.predict(X_train, pred_leaf=True)
    lr_model = LinearRegression()
    print("开始训练 LR 模型...")
    lr_model.fit(X_gbdt, y_train)
    print("LR 模型训练完成")
    
    return gbdt_model, lr_model

# 测试 GBDT+LR 模型的函数
def test_gbdt_lr(model, user_map, item_map, test_data,  target_data,item_attributes, output_filepath):
    X_test, _ = prepare_gbdt_lr_data(user_map, item_map, test_data, item_attributes)
    gbdt_model, lr_model = model
    
    # 使用 GBDT 获取预测
    X_gbdt_test = gbdt_model.predict(X_test, pred_leaf=True)
    
    # 使用 LR 进行最终预测
    lr_predictions = lr_model.predict(X_gbdt_test)
    
    # 组织预测结果并写入文件
    predictions = defaultdict(list)
    for (u, i, _), pred in zip(test_data, lr_predictions):
        predictions[u].append((i, pred))
    
    with open(output_filepath, 'w') as f:
        for user_id, preds in predictions.items():
            num_ratings = len(preds)
            f.write(f"{user_id}|{num_ratings}\n")
            for item_id, score in preds:
                f.write(f"{item_id}\t{score:.6f}\n")
    
    print(f"预测结果已保存到 {output_filepath}")


if __name__ == "__main__":
    # 读取训练数据
    train_data = data_process.read_train_data('../data/train.txt')
    mapper = data_process.DatasetMapper()
    user_map, item_map = mapper.load_mappings()
    
    # 拆分训练数据和测试数据
    train_data, test_data = data_process.split_train_test(train_data, train_ratio=0.9)
    target_data = data_process.read_test_data('../data/test.txt')

    # 读取物品属性
    item_attributes = data_analysis.read_attribute_data("../data/itemAttribute.txt")

    # 准备并训练 GBDT+LR 模型
    X_train, y_train = prepare_gbdt_lr_data(user_map, item_map, train_data, item_attributes)
    gbdt_model, lr_model = train_gbdt_lr(X_train, y_train)
    
    # 测试 GBDT+LR 模型
    test_gbdt_lr((gbdt_model, lr_model), user_map, item_map, test_data, target_data,item_attributes, 'result/predictions.txt')
