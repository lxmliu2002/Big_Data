import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm




def prepare_train_data(user_map, item_map, data):
    users = torch.tensor([user_map[u] for u, _, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i, _ in data], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in data], dtype=torch.float32)
    return TensorDataset(users, items, ratings)

def prepare_test_data(user_map, item_map, data):
    users = torch.tensor([user_map[u] for u, _ in data], dtype=torch.long)
    items = torch.tensor([item_map[i] for _, i in data], dtype=torch.long)
    return TensorDataset(users, items)

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
            model.save_model(epoch+1)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        model.save_model(epoch+1)
