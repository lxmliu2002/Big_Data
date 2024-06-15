import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_attr1, num_attr2, gmf_dim = 64, mlp_dim = 128, layers_dim=[256,128,64,32]):
        super(NeuMF, self).__init__()

        # GMF部分
        self.user_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_gmf = nn.Embedding(num_items, gmf_dim)
        self.gmf_out = nn.Linear(gmf_dim, 1)

        # MLP部分
        self.user_mlp = nn.Embedding(num_users, mlp_dim)
        self.item_mlp = nn.Embedding(num_items, mlp_dim)
        self.mlp_layers = nn.ModuleList()
        now_num = 4 * mlp_dim
        for layer_dim in layers_dim:
            self.mlp_layers.append(nn.Linear(now_num, layer_dim))  # Concatenate user and item embeddings
            now_num = layer_dim

        self.mlp_out = nn.Linear(layers_dim[-1], 1)
        # Fusion layer
        self.fusion_layer = nn.Linear(2, 1)  # Fusion layer to combine GMF and MLP outputs

        # Item属性部分
        self.item_attr1 = nn.Embedding(num_attr1, mlp_dim)
        self.item_attr2 = nn.Embedding(num_attr2, mlp_dim)

        self.init_weights()

    def init_weights(self):
        # 初始化所有 embedding 和偏置
        nn.init.uniform_(self.user_gmf.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_gmf.weight, -0.05, 0.05)
        nn.init.uniform_(self.user_mlp.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_mlp.weight, -0.05, 0.05)
        nn.init.xavier_uniform_(self.gmf_out.weight)
        nn.init.zeros_(self.gmf_out.bias)
        nn.init.xavier_uniform_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.uniform_(self.item_attr1.weight, -0.05, 0.05)
        nn.init.uniform_(self.item_attr2.weight, -0.05, 0.05)
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        
    def forward(self, user_idx, item_idx, item_attr1_idx, item_attr2_idx):
        # GMF部分
        user_gmf = self.user_gmf(user_idx)
        item_gmf = self.item_gmf(item_idx)
        gmf = user_gmf * item_gmf
        gmf_out = self.gmf_out(gmf)

        # MLP部分
        user_mlp = self.user_mlp(user_idx)
        item_mlp = self.item_mlp(item_idx)
        attr1 = self.item_attr1(item_attr1_idx)
        attr2 = self.item_attr2(item_attr2_idx)
        mlp_input = torch.cat([user_mlp, item_mlp, attr1, attr2], dim=1)
        for layer in self.mlp_layers:
            mlp_input = torch.relu(layer(mlp_input))

        mlp_out = self.mlp_out(mlp_input)

        fusion_input = torch.cat([gmf_out, mlp_out], dim=1)
        prediction = torch.sigmoid(self.fusion_layer(fusion_input)).squeeze() * 100

        return prediction

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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)


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