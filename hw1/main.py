import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 生成数据
def generate_data(num_samples):
    X = torch.rand(num_samples, 1) * 15 + 1  # 从 [1, 16] 均匀采样
    y = torch.log2(X) + torch.cos(torch.tensor(np.pi) * X / 2)
    return X, y


# 划分数据集
def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# 定义模型
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.SiLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义训练函数
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=100):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # if (epoch+1) % 10 == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 定义测试函数
def test_model(model, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        return mse.item()


# 可视化验证集原始样本点和模型预测点
def visualize_results(model, X_val, y_val, n):
    with torch.no_grad():
        y_pred = model(X_val)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_val.cpu().numpy(), y_val.cpu().numpy(), label='Original Data')
    plt.scatter(X_val.cpu().numpy(), y_pred.cpu().numpy(), color='red', label='Predicted Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Validation Set: Original vs. Predicted with N={n}')
    plt.legend()
    plt.show()


# 数据量
N = [200, 2000]


# 模型参数
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 5
learning_rate = 0.01
num_epochs = 15000

# 调参范围
learning_rates = [0.01, 0.05, 0.09]
hidden_dims = [32, 64, 128]

# 测试性能的超参数
best_lr = 0.01
best_hidden_dim = 64

# 实验
for n in N:
    print(f"Experiment with N={n}:, num_epochs={num_epochs}")

    # 生成数据
    X, y = generate_data(n)

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
    y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

    # 构建模型
    model = FeedforwardNN(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, criterion, optimizer, X_train, y_train, num_epochs)

    # 测试模型性能
    mse = test_model(model, X_test, y_test)
    # print(f"Test MSE: {mse:.4f}")
    best_mse = mse

    # 调参分析
    for lr in learning_rates:
        for h_dim in hidden_dims:
            model = FeedforwardNN(input_dim, h_dim, output_dim, num_layers).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, criterion, optimizer, X_train, y_train, num_epochs)
            mse = test_model(model, X_val, y_val)
            print(f'Validation MSE with lr={lr}, hidden_dim={h_dim}: {mse:.4f}')
            if mse < best_mse:
                best_mse = mse
                best_lr = lr
                best_hidden_dim = h_dim

    print(f"Best hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}")

    # 使用最佳超参数重新训练模型
    model = FeedforwardNN(input_dim, best_hidden_dim, output_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    train_model(model, criterion, optimizer, X_train, y_train, num_epochs)

    # 可视化验证集
    visualize_results(model, X_val, y_val, n)

    # 在测试集上测试性能
    mse = test_model(model, X_test, y_test)
    print(f"Final Test MSE: {mse:.4f}")
