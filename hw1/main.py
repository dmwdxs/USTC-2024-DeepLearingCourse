import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


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


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.silu(self.fc1(x))  # 使用 SiLU 激活函数
        for layer in self.hidden_layers:
            out = F.silu(layer(out))  
        out = self.fc2(out)
        return out


# 定义训练函数
def train_model(model, criterion, optimizer, X_train, y_train):
    losses = []
    loss = 1.0
    epoch = 1
    while loss > 1e-4:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        epoch = epoch + 1
        if epoch % 2000 == 0:
            print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')
        if loss <= 1e-4:
            print(f"Final Epoch:{epoch}, with loss:{loss}")
        if epoch >=30000:
            break
    
    return losses


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
    plt.scatter(X_val.cpu().numpy(), y_val.cpu().numpy(), label='Original Data', s=8)
    plt.scatter(X_val.cpu().numpy(), y_pred.cpu().numpy(), color='red', label='Predicted Data', s=8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Validation Set: Original vs. Predicted with N={n}')
    plt.legend()
    plt.show()


# 数据量
N = [200, 2000, 10000]


# 模型参数
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 4
learning_rate = 0.01

# 实验
for n in N:
    print(f"Experiment with N={n}")
    # 生成数据
    X, y = generate_data(n)

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
    y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 模型输入
    model = FeedforwardNN(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = train_model(model, criterion, optimizer, X_train, y_train)
    mse = test_model(model, X_val, y_val)
    print(f'Validation MSE with lr={learning_rate}, hidden_dim={hidden_dim}: {mse:.4f}')

    # 重新训练模型
    model = FeedforwardNN(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = train_model(model, criterion, optimizer, X_train, y_train)

    # 可视化验证集
    visualize_results(model, X_val, y_val, n)

    # 在测试集上测试性能
    mse = test_model(model, X_test, y_test)
    print(f"Final Test MSE: {mse:.4f}")

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
