import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_attr1, num_attr2, embedding_dim=32, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers

        # 初始化用户和项目嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 初始化属性嵌入
        self.item_attr1 = nn.Embedding(num_attr1, embedding_dim)
        self.item_attr2 = nn.Embedding(num_attr2, embedding_dim)

        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.item_attr1.weight)
        nn.init.xavier_uniform_(self.item_attr2.weight)

    def forward(self, user_idx, item_idx, item_attr1_idx, item_attr2_idx, edge_index):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # 获取属性嵌入
        attr1_emb = self.item_attr1(item_attr1_idx)
        attr2_emb = self.item_attr2(item_attr2_idx)

        # 初始嵌入
        all_emb = torch.cat([user_emb, item_emb, attr1_emb, attr2_emb], dim=0)
        embs = [all_emb]
        print(all_emb.size())
        print(edge_index.size())
        # 邻域聚合
        for layer in range(self.num_layers):
            # 使用torch_geometric的GCNConv进行邻域聚合
            conv = GCNConv(all_emb.size(1), all_emb.size(1))
            all_emb = conv(all_emb, edge_index)
            embs.append(all_emb)

        # 组合嵌入
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)

        # 获取最终的用户和项目嵌入
        user_final_emb = final_emb[user_idx]
        item_final_emb = final_emb[item_idx + len(user_idx)]

        # 预测输出
        score = (user_final_emb * item_final_emb).sum(dim=1)
        return score

    def save_model(self, epoch=0, save_dir='model/NeuMF/model_NeuMF.pt'):
        if epoch != 0:
            save_dir = f'model/NeuMF/model_NeuMF_{epoch}_epochs.pt'
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}\n')
    
    def load_model(self, load_dir='model/NeuMF/model_NeuMF.pt'):
        self.load_state_dict(torch.load(load_dir))
        print(f'Model loaded from {load_dir}')

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
        else:
            attr1 = -1
            attr2 = -1
            attr1_idx = attr1_map[attr1]
            attr2_idx = attr2_map[attr2]
            mapped_data[item_idx] = (attr1_idx, attr2_idx)

    return mapped_data

def prepare_train_data(user_map, item_map, data, attr_map):
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in data], dtype=torch.float32)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long)
    # 获取物品属性列表
    item_attr1 = torch.tensor([attr_map[item_map[i]][0] for _, i, _ in data], dtype=torch.long)
    item_attr2 = torch.tensor([attr_map[item_map[i]][1] for _, i, _ in data], dtype=torch.long)
    
    # 假设存在带权重的交互关系，例如评分
    weights = torch.tensor([r for _, _, r in data], dtype=torch.float32)

    # 构建带权重的边索引
    edge_index = torch.tensor([(user_map[u],item_map[i]) for u, i, _ in data])
    edge_attr = weights

    edge_index = edge_index.view(2, -1)
    edge_attr = weights.view(-1, 1)  # 将评分作为边属性

    # 构建 torch_geometric 的 Data 对象
    data = Data(x=None, edge_index=edge_index, edge_attr=edge_attr)

    return TensorDataset(users, items, ratings, item_attr1, item_attr2), data


def train(model, train_data,train_edge, test_data,test_edge,  num_epochs=20, lr=0.01, weight_decay=1e-6,batch_size=512, device='cuda',gamma=0.5):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)


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
                prediction = model(user, item_raw_id, attr1, attr2, train_edge.edge_index)
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

                prediction = model(user, item_raw_id, attr1, attr2,test_edge.edge_index)
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

def test(model, user_map, item_map, test_data, output_filepath, attr_map=None):

    test_dataset, user_inverse_map, item_inverse_map = prepare_test_data(user_map, item_map, test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) 

    predictions = defaultdict(list)

    # 将张量传递给模型进行预测
    with torch.no_grad():
        for users, items_raw_id in test_loader:
            users = users.to(device, non_blocking=True)
            items_raw_id = items_raw_id.to(device, non_blocking=True)
            
            # 获取物品属性列表
            item_attr1 = torch.tensor([attr_map[it][0] for it in items_raw_id]).to(device)
            item_attr2 = torch.tensor([attr_map[it][1] for it in items_raw_id]).to(device)

            preds = model(users, items_raw_id, item_attr1, item_attr2)
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
