import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# 超参数
input_size = 28     # 输入特征维度（每行像素数）
hidden_size = 128   # LSTM隐藏层维度
num_layers = 1      # LSTM层数
num_classes = 10    # 输出类别数
num_epochs = 10     # 训练轮数
batch_size = 64     # 批大小
learning_rate = 0.001

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# 定义LSTM模型
class LSTM_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_MNIST, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 输入尺寸: (batch_size, 1, 28, 28)
        # 转换为序列: (batch_size, 28, 28)
        x = x.squeeze(1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        return out

class LSTM_Bidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_Bidirectional, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM，输出维度翻倍

    def forward(self, x):
        # 输入尺寸: (batch_size, 1, 28, 28)
        # 转换为序列: (batch_size, 28, 28)
        x = x.squeeze(1)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 双向LSTM，隐藏状态维度翻倍
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 双向LSTM，隐藏状态维度翻倍
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size * 2)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out

class FourWayLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()
        # 行方向双向LSTM
        self.row_lstm = nn.LSTM(input_dim, hidden_size, 
                              bidirectional=True, batch_first=True)
        # 列方向双向LSTM
        self.col_lstm = nn.LSTM(input_dim, hidden_size,
                              bidirectional=True, batch_first=True)
        
        # 特征融合层
        self.fc = nn.Linear(4*hidden_size, num_classes)

    def forward(self, x):
        # 原始输入形状: (batch, 1, 28, 28)
        x = x.squeeze(1)  # (batch,28,28)
        
        # 行方向处理（双向）
        row_out, _ = self.row_lstm(x)  # (batch,28,256)
        row_features = row_out[:, -1, :]  # (batch,256)
        
        # 列方向处理（需转置维度）
        x_transposed = x.transpose(1,2)  # (batch,28,28)
        col_out, _ = self.col_lstm(x_transposed)  # (batch,28,256)
        col_features = col_out[:, -1, :]  # (batch,256)
        
        # 特征拼接
        combined = torch.cat([row_features, col_features], dim=1)  # (batch,512)
        
        # 分类输出
        return self.fc(combined)
    
# 初始化模型
model = LSTM_MNIST(input_size, hidden_size, num_layers, num_classes).to(device)
model_bidirectional = LSTM_Bidirectional(input_size, hidden_size, num_layers, num_classes).to(device)
model_fourway = FourWayLSTM(input_size, hidden_size, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_bidirectional = optim.Adam(model_bidirectional.parameters(), lr=learning_rate)
optimizer_fourway = optim.Adam(model_fourway.parameters(), lr=learning_rate)

# 训练循环
total_step = len(train_loader)
losses = []
losses_bidirectional = []
losses_fourway = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        outputs_bidirectional = model_bidirectional(images)
        outputs_fourway = model_fourway(images)
        loss = criterion(outputs, labels)
        loss_bidirectional = criterion(outputs_bidirectional, labels)
        loss_fourway = criterion(outputs_fourway, labels)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_bidirectional.zero_grad()
        loss_bidirectional.backward()
        optimizer_bidirectional.step()

        optimizer_fourway.zero_grad()
        loss_fourway.backward()
        optimizer_fourway.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss_bidirectional: {loss_bidirectional.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss_fourway: {loss_fourway.item():.4f}')
            losses.append(loss.item())
            losses_bidirectional.append(loss_bidirectional.item())
            losses_fourway.append(loss_fourway.item())

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

model_bidirectional.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_bidirectional(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy Bidirectional: {100 * correct / total:.2f}%')

model_fourway.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_fourway(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy Fourway: {100 * correct / total:.2f}%')

plt.plot(losses,label='LSTM')
plt.plot(losses_bidirectional,label='Bidirectional LSTM')
plt.plot(losses_fourway,label='Fourway LSTM')
plt.xlabel('100 Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), f'LSTM_epoch{num_epochs}.pth')
torch.save(model_bidirectional.state_dict(), f'Bidirectional_LSTM_epoch{num_epochs}.pth')
torch.save(model_fourway.state_dict(), f'Fourway_LSTM_epoch{num_epochs}.pth')