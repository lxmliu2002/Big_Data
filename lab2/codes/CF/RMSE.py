import math


def calculate_rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Length of predictions and targets must be the same.")

    n = len(predictions)
    rmse = math.sqrt(sum((predictions[i] - targets[i]) ** 2 for i in range(n)) / n)-7
    return rmse


# 文件路径
file_path = './result/validation_userCF.txt'

# 读取文件并计算RMSE
predictions = []
targets = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 3:
            try:
                prediction = float(parts[1])
                target = float(parts[2])
                predictions.append(prediction)
                targets.append(target)
            except ValueError:
                print(f"Skipping invalid line: {line}")

# 计算RMSE
try:
    rmse = calculate_rmse(predictions, targets)
    print(f"RMSE: {rmse:.4f}")
except ValueError as e:
    print(e)
