import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from tqdm import tqdm
import data_process
import data_analysis
import SVD
import numpy as np


# 准备用于 GBDT+LR 模型的训练数据
def prepare_gbdt_lr_data(user_map, item_map, data, item_attributes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long).to(device)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long).to(device)
    
    # 加载训练好的 SVD 模型
    svd_model = SVD.SVDbiasModel(num_users=len(user_map), num_items=len(item_map), latent_dim=10)
    svd_model.load_model('../trainSVD_cuda/model/SVDbias/model_SVDbias_method1_best.pt')
    svd_model.eval().to(device)
    
    # 获取用户和物品的 embedding 以及bias的
    with torch.no_grad():
        user_embeddings = svd_model.user_factors(users).detach().cpu().numpy()
        item_embeddings = svd_model.item_factors(items).detach().cpu().numpy()
        ubs = svd_model.user_bias(users).detach().cpu().numpy()
        ibs = svd_model.item_bias(items).detach().cpu().numpy()
    
    # 构建物品属性特征
    item_features = []
    for item_id, attr1, attr2 in item_attributes:
        item_features.append([ attr1, attr2])
    
    # 合并 embedding 和物品属性特征
    X = []
    y = []
    for i, (u, item, r) in enumerate(data):
        user_idx = user_map[u]
        item_idx = item_map[item]
        user_emb = user_embeddings[i]
        item_emb = item_embeddings[i]
        ub = ubs[i]
        ib = ibs[i]
        item_attr = item_features[item_idx]
        combined_features = list(user_emb) + list(item_emb) + item_attr + list(ub) + list(ib)
        X.append(combined_features)
        y.append(r)
    
    return X, y

# 训练 GBDT+LR 模型的函数
def train_gbdt_lr(X_train, y_train, model_save_path):
    # 初始化 LightGBM GBDT 模型
    gbdt_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'feature_fraction': 0.9,
        'verbose': 2, # 设置 verbose 参数
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
    
    # 保存模型
    torch.save({
        'gbdt_model': gbdt_model,
        'lr_model': lr_model
    }, model_save_path)
    print(f"模型已保存到 {model_save_path}")
    return  gbdt_model,lr_model
    

# 测试 GBDT+LR 模型的函数
def test_gbdt_lr(model_path, user_map, item_map, test_data, item_attributes):
    # 加载模型
    checkpoint = torch.load(model_path)
    gbdt_model = checkpoint['gbdt_model']
    lr_model = checkpoint['lr_model']
    
    # 准备测试数据
    X_test,Y_test = prepare_gbdt_lr_data(user_map, item_map, test_data, item_attributes)
    
    # 使用 GBDT 获取预测
    X_gbdt_test = gbdt_model.predict(X_test, pred_leaf=True)
    
    # 使用 LR 进行最终预测
    lr_predictions = lr_model.predict(X_gbdt_test)
    
    # 计算并输出 RMSE
    true_ratings = np.array([r for _, _, r in test_data])
    rmse = mean_squared_error(true_ratings, lr_predictions, squared=False)
    print(f"在测试集上的RMSE为: {rmse:.4f}")


# 示例用法
if __name__ == "__main__":
    is_train = True

    # 读取训练数据
    train_data = data_process.read_train_data('../data/train.txt')
    mapper = data_process.DatasetMapper()
    user_map, item_map = mapper.load_mappings()

    # 拆分训练数据和测试数据
    train_data, test_data = data_process.split_train_test(train_data, train_ratio=0.9)

    # 读取物品属性
    item_attributes = data_analysis.read_attribute_data("../data/itemAttribute.txt")

    if is_train == True:
        # 准备并训练 GBDT+LR 模型
        X_train, y_train = prepare_gbdt_lr_data(user_map, item_map, train_data, item_attributes)
        gbdt_model, lr_model = train_gbdt_lr(X_train, y_train, 'model/gbdt_lr_model.pth')
        test_gbdt_lr('model/gbdt_lr_model.pth', user_map, item_map, test_data, item_attributes)
    else :
        # 测试 GBDT+LR 模型并输出预测结果
        test_gbdt_lr('model/gbdt_lr_model.pth', user_map, item_map, test_data, item_attributes)