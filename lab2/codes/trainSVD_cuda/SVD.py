import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import retrieval_normalized_dcg as ndcg
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

    def save_model(self, epoch=0, save_dir='model/model_SVD.pt'):
        if epoch != 0:
            save_dir = f'model/model_SVD_{epoch}_epochs.pt'
        torch.save(self.state_dict(), save_dir)
        print(f'Model saved at {save_dir}\n')

def prepare_data(user_map, item_map, data):
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in data], dtype=torch.float32)
    return TensorDataset(users, items, ratings)

def hit_rate(predictions, ratings, top_k=10):
    _, indices = torch.topk(predictions, top_k)
    hits = torch.sum(ratings[indices] > 0).item()
    return hits / len(ratings)

def train(model, train_data, test_data, num_epochs=20, lr=0.01, weight_decay=1e-6, batch_size=512, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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

                total_loss += loss.item() * len(user)
        
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
        # Calculate NDCG@k
        ndcg_score10 = ndcg(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=10).item()
        ndcg_score5 = ndcg(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=5).item()
        ndcg_score3 = ndcg(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=3).item()
        # Calculate HR@k
        hr3 = hit_rate(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=3)
        hr5 = hit_rate(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=5)
        hr10 = hit_rate(torch.tensor(all_predictions), torch.tensor(all_ratings), top_k=10)

        print(f'\n======Epoch {epoch + 1}/{num_epochs}======\nTest Loss: {test_loss} AUC: {auc} \nNDCG@3: {ndcg_score3},NDCG@5: {ndcg_score5}, NDCG@10: {ndcg_score10}')
        print(f'HR@3: {hr3}, HR@5: {hr5}, HR@10: {hr10}')

        model.save_model(epoch+1)
