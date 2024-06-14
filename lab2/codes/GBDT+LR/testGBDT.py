import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm
import data_process
import data_analysis
import SVD

# 假设 SVDbiasModel 已经定义在 SVD 模块中

# 准备用于 GBDT+LR 模型的测试数据
def prepare_gbdt_lr_test_data(user_map, item_map, data, item_attributes):
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
    
    for i, (u, item) in enumerate(data):
        user_idx = user_map[u]
        item_idx = item_map[item]
        user_emb = user_embeddings[i]
        item_emb = item_embeddings[i]
        
        # 获取物品的属性信息
        item_attr = item_attributes[item_idx]
        
        # 使用标签编码处理物品属性
        label_encoder = LabelEncoder()
        encoded_attr = label_encoder.fit_transform(item_attr)
        
        # 合并 embedding 和物品属性特征
        combined_features = list(user_emb) + list(item_emb) + list(encoded_attr)
        X.append(combined_features)
    
    return X

# 测试 GBDT+LR 模型的函数
def test_gbdt_lr(model_path, user_map, item_map, test_data, item_attributes, output_filepath):
    # 加载模型
    checkpoint = torch.load(model_path)
    gbdt_model = checkpoint['gbdt_model']
    lr_model = checkpoint['lr_model']
    
    # 准备测试数据
    X_test = prepare_gbdt_lr_test_data(user_map, item_map, test_data, item_attributes)
    
    # 使用 GBDT 获取预测
    X_gbdt_test = gbdt_model.predict(X_test, pred_leaf=True)
    
    # 使用 LR 进行最终预测
    lr_predictions = lr_model.predict(X_gbdt_test)
    
    # 组织预测结果并写入文件
    predictions = defaultdict(list)
    for (u, i), pred in zip(test_data, lr_predictions):
        predictions[u].append((i, pred))
    
    with open(output_filepath, 'w') as f:
        for user_id, preds in predictions.items():
            num_ratings = len(preds)
            f.write(f"{user_id}|{num_ratings}\n")
            for item_id, score in preds:
                f.write(f"{item_id}\t{score:.6f}\n")
    
    print(f"预测结果已保存到 {output_filepath}")

# 示例用法
if __name__ == "__main__":
    # 读取测试数据
    test_data = data_process.read_test_data('../data/test.txt')
    mapper = data_process.DatasetMapper()
    user_map, item_map = mapper.load_mappings()
    
    # 读取物品属性
    item_attributes = data_analysis.read_attribute_data("../data/itemAttribute.txt")
    
    # 测试 GBDT+LR 模型并输出预测结果
    test_gbdt_lr('model/gbdt_lr_model.pth', user_map, item_map, test_data, item_attributes, 'result/predictions.txt')
