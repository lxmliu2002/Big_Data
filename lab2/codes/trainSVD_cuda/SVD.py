import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm

class SVDModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(SVDModel, self).__init__()
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, user_idx, item_idx):
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        dot = (user_factors * item_factors).sum(1)
        return dot

    def save_model(self, epoch=0, save_dir='model/SVD/model_SVD.pt'):
        if epoch != 0:
            save_dir = f'model/SVD/model_SVD_{epoch}_epochs.pt'
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}\n')

    def load_model(self, load_dir='model/SVD/model_SVD_8_epochs.pt'):
        self.load_state_dict(torch.load(load_dir))
        print(f'Model loaded from {load_dir}\n')

class SVDbiasModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(SVDbiasModel, self).__init__()
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)  # 用户偏置
        self.item_bias = nn.Embedding(num_items, 1)  # 物品偏置
        self.global_bias = nn.Parameter(torch.zeros(1))  # 全局偏置
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(0, 0.05)  # 初始化用户偏置
        self.item_bias.weight.data.uniform_(0, 0.05)  # 初始化物品偏置

    def forward(self, user_idx, item_idx):
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        user_bias = self.user_bias(user_idx).squeeze()
        item_bias = self.item_bias(item_idx).squeeze()
        dot = (user_factors * item_factors).sum(1)
        return dot + user_bias + item_bias + self.global_bias

    def save_model(self, epoch=0, save_dir='model/SVDbias/model_SVD.pt'):
        if epoch != 0:
            save_dir = f'model/SVDbias/model_SVD_{epoch}_epochs.pt'
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}\n')
    
    def load_model(self, load_dir='model/SVDbias/model_SVD_8_epochs.pt'):
        self.load_state_dict(torch.load(load_dir))
        print(f'模型加载于 {load_dir}')


def prepare_train_data(user_map, item_map, data):
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in data], dtype=torch.float32)
    return TensorDataset(users, items, ratings)

def prepare_test_data(user_map, item_map, data, test =False):
    users = torch.tensor([user_map[u] for u, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i in data], dtype=torch.long)
    

    user_inverse_map = {v: k for k, v in user_map.items()}
    item_inverse_map = {v: k for k, v in item_map.items()}
    return TensorDataset(users, items), user_inverse_map, item_inverse_map

def train(model, train_data, test_data, num_epochs=20, lr=0.01, weight_decay=1e-6, batch_size=512, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    
    # 早停
    best_auc = 0
    early_stopping_counter = 0
    early_stopping_patience = 4

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_iterator:
            for user, item, rating in train_iterator:
                user = user.to(device, non_blocking=True)
                item = item.to(device, non_blocking=True)
                rating = rating.to(device, non_blocking=True)

                optimizer.zero_grad()
                prediction = model(user, item)
                loss = criterion(prediction, rating)
                loss.backward()
                optimizer.step()
        
        scheduler.step()
        test_loss = 0
        all_predictions = []
        all_ratings = []
        with torch.no_grad():
            model.eval()
            for user, item, rating in test_loader:
                user = user.to(device, non_blocking=True)
                item = item.to(device, non_blocking=True)
                rating = rating.to(device, non_blocking=True)

                prediction = model(user, item)
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

        model.save_model(epoch+1)

# 进行预测并输出结果
def test(model, user_map, item_map, test_data, output_filepath):

    test_dataset, user_inverse_map, item_inverse_map = prepare_test_data(user_map, item_map, test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) 

    predictions = defaultdict(list)

    # 将张量传递给模型进行预测
    with torch.no_grad():
        for users, items in test_loader:
            preds = model(users, items)
            for i in range(len(users)):
                user_id = user_inverse_map[users[i].item()]
                item_id = item_inverse_map[items[i].item()]
                predictions[user_id].append((item_id, preds[i].item()))

    # 组织预测结果并写入文件
    with open(output_filepath, 'w') as f:
        for user_id, preds in predictions.items():
            num_ratings = len(preds)
            f.write(f"{user_id}|{num_ratings}\n")
            for item_id, score in preds:
                f.write(f"{item_id}\t{score:.6f}\n")
    
    print(f"预测结果已经写入到 {output_filepath}")
