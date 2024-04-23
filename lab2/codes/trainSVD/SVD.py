import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class SVDModel(nn.Module):
    def __init__(self, user_map, item_map, latent_dim):
        super(SVDModel, self).__init__()
        self.user_factors = nn.Embedding(len(user_map), latent_dim)
        self.item_factors = nn.Embedding(len(item_map), latent_dim)
        self.user_map = user_map
        self.item_map = item_map
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, user, item):
        user_idx = torch.tensor([self.user_map[int(u)] for u in user], dtype=torch.long)
        item_idx = torch.tensor([self.item_map[int(i)] for i in item], dtype=torch.long)
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        
        dot = (user_factors * item_factors).sum(1)
        return dot

    def save_model(self,save_dir='model/model_SVD.pt'):
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}')

def train(model, train_data, test_data, num_epochs=20, lr=0.01, weight_decay=1e-6, batch_size=256):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for user, item, rating in train_tqdm:
            rating = rating.float()
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_tqdm.set_postfix({'Loss': total_loss / len(user)})
        
        scheduler.step()

        test_loss = 0
        with torch.no_grad():
            model.eval()
            test_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} Testing")
            for user, item, rating in test_tqdm:
                rating = rating.float()
                prediction = model(user, item)
                loss = criterion(prediction, rating)
                test_loss += loss.item() * len(user)
                test_tqdm.set_postfix({'Test Loss': test_loss / len(test_loader.dataset)})

        print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss / len(test_loader.dataset)}')

# 以下代码部分是预测函数的示例，已注释掉
# def predict(model, user, item):
#     model.eval()
#     user_idx = torch.tensor([model.user_map[user]], dtype=torch.long)
#     item_idx = torch.tensor([model.item_map[item]], dtype=torch.long)
#     with torch.no_grad():
#         prediction = model(user_idx, item_idx)
#     return prediction.item()
