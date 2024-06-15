import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm

class SVDattrModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, num_attr1, num_attr2):
        super(SVDattrModel, self).__init__()
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 新增两个物品属性的embedding
        self.item_attributes = nn.ModuleList([
            nn.Embedding(num_attr1, latent_dim),  # 假设第一个属性的类别数量为 num_item_attributes[0]
            nn.Embedding(num_attr2, latent_dim)   # 假设第二个属性的类别数量为 num_item_attributes[1]
        ])
        
        # 初始化所有 embedding 和偏置
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(0, 0.05)
        self.item_bias.weight.data.uniform_(0, 0.05)
        for attr_embedding in self.item_attributes:
            attr_embedding.weight.data.uniform_(0, 0.05)

    def forward(self, user_idx, item_idx, item_attr1_idx, item_attr2_idx):
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        user_bias = self.user_bias(user_idx).squeeze()
        item_bias = self.item_bias(item_idx).squeeze()
        
        # 获取物品属性的embedding
        item_attr1_factors = self.item_attributes[0](item_attr1_idx)
        item_attr2_factors = self.item_attributes[1](item_attr2_idx)
        
        # 计算预测评分
        dot = (user_factors * item_factors).sum(1)
        attr1_term = (item_attr1_factors * user_factors).sum(1)
        attr2_term = (item_attr2_factors * user_factors).sum(1)
        
        prediction = dot +  attr1_term + attr2_term +user_bias + item_bias + self.global_bias
        
        return prediction

    def save_model(self, epoch=0, save_dir='model/SVDattr/model_SVD.pt'):
        if epoch != 0:
            save_dir = f'model/SVDattr/model_SVD_{epoch}_epochs.pt'
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}\n')
    
    def load_model(self, load_dir='model/SVDattr/model_SVDbias.pt'):
        self.load_state_dict(torch.load(load_dir))
        print(f'模型加载于 {load_dir}')


def map_attr_data(attr_data, attr1_map, attr2_map, item_map):
    mapped_data = {}
    for item_id, item_idx in item_map.items():
        if item_id in attr_data.keys():
            attr1 = attr_data[item_id][0]
            attr2 = attr_data[item_id][1]
            if attr1 in attr1_map and attr2 in attr2_map and item_id in item_map:
                attr1_idx = attr1_map[attr1]
                attr2_idx = attr2_map[attr2]
                mapped_data[item_idx] = (attr1_idx, attr2_idx)
        else :
            attr1 = -1
            attr2 = -1
            attr1_idx = attr1_map[attr1]
            attr2_idx = attr2_map[attr2]
            mapped_data[item_idx] = (attr1_idx, attr2_idx)

    return mapped_data

def prepare_train_data(user_map, item_map, data, attr_map):
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in data], dtype=torch.float32)
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long)
    # 获取物品属性列表
    item_attr1 = torch.tensor([attr_map[item_map[i]][0] for _, i, _ in data], dtype=torch.long)
    item_attr2 = torch.tensor([attr_map[item_map[i]][1] for _, i, _ in data], dtype=torch.long)
    return TensorDataset(users, items, ratings, item_attr1, item_attr2)

def prepare_test_data(user_map, item_map, data, attr_map, test =False):
    users = torch.tensor([user_map[u] for u, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i in data], dtype=torch.long)
    # 获取物品属性列表
    item_attr1 = torch.tensor([attr_map[item_map[i]][0] for _, i in data], dtype=torch.long)
    item_attr2 = torch.tensor([attr_map[item_map[i]][1] for _, i in data], dtype=torch.long)

    user_inverse_map = {v: k for k, v in user_map.items()}
    item_inverse_map = {v: k for k, v in item_map.items()}
    return TensorDataset(users, items, item_attr1, item_attr2), user_inverse_map, item_inverse_map


def train(model, train_data, test_data,  num_epochs=20, lr=0.01, weight_decay=1e-6,batch_size=512, device='cuda',gamma=0.5):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    
    # 早停
    best_auc = 0
    early_stopping_counter = 0
    early_stopping_patience = 4

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_iterator:
            for user, item_raw_id, rating, attr1, attr2 in train_iterator:
                user = user.to(device, non_blocking=True)
                rating = rating.to(device, non_blocking=True)
                item_raw_id = item_raw_id.to(device, non_blocking=True)
                attr1 = attr1.to(device, non_blocking =True)
                attr2 = attr2.to(device, non_blocking =True)

                optimizer.zero_grad()
                prediction = model(user, item_raw_id, attr1, attr2)
                loss = criterion(prediction, rating)
                loss.backward()
                optimizer.step()
        
        scheduler.step()
        test_loss = 0
        all_predictions = []
        all_ratings = []
        with torch.no_grad():
            model.eval()
            for user, item_raw_id, rating, attr1, attr2 in test_loader:
                user = user.to(device, non_blocking=True)
                item_raw_id = item_raw_id.to(device, non_blocking=True)
                rating = rating.to(device, non_blocking=True)
                attr1 = attr1.to(device, non_blocking =True)
                attr2 = attr2.to(device, non_blocking =True)

                prediction = model(user, item_raw_id, attr1, attr2)
                test_loss += criterion(prediction, rating).item() * len(user)

                all_predictions.extend(prediction.cpu().numpy())
                all_ratings.extend(rating.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        
        # Calculate AUC
        if len(set(all_ratings)) > 1:
            auc = roc_auc_score((torch.tensor(all_ratings) > torch.tensor(all_ratings).mean()).numpy(), all_predictions)
        else:
            auc = float('nan')

        print(f'\n======Epoch {epoch + 1}/{num_epochs}======\nMSE Loss: {test_loss} AUC: {auc} \n')

        if auc > best_auc:
            best_auc = auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        model.save_model(epoch + 1)

def test(model, user_map, item_map, test_data, output_filepath, attr_map=None, device='cuda'):
    test_dataset, user_inverse_map, item_inverse_map = prepare_test_data(user_map, item_map, test_data,attr_map)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) 

    predictions = defaultdict(list)
    model.to(device)
    # 将张量传递给模型进行预测
    with torch.no_grad():
        for users, items_raw_id, attr1, attr2 in test_loader:
            users = users.to(device, non_blocking=True)
            items_raw_id = items_raw_id.to(device, non_blocking=True)
            attr1 = attr1.to(device, non_blocking =True)
            attr2 = attr2.to(device, non_blocking =True)
            
            preds = model(users, items_raw_id, attr1, attr2)
            for i in range(len(users)):
                user_id = user_inverse_map[users[i].item()]
                item_id = item_inverse_map[items_raw_id[i].item()]
                predictions[user_id].append((item_id, preds[i].item()))

    # 组织预测结果并写入文件
    with open(output_filepath, 'w') as f:
        for user_id, preds in predictions.items():
            num_ratings = len(preds)
            f.write(f"{user_id}|{num_ratings}\n")
            for item_id, score in preds:
                f.write(f"{item_id}\t{score:.6f}\n")
    
    print(f"预测结果已经写入到 {output_filepath}")