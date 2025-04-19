import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(           
                in_channels=1,  # 输入通道数，灰度图像为1
                out_channels=8,  # 输出通道数
                kernel_size=3,  # 卷积核大小
                stride=1,  # 步长
                padding=1,  # 填充
                bias=True  # 是否添加偏置项
            ),nn.ReLU(),    # 28-3+1*2+1 = 28
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28-2+2)/2=14
            nn.Conv2d(8,16,kernel_size=5),nn.ReLU(),# 14-5+1=10
            nn.MaxPool2d(kernel_size=2,stride=2),# (10-2+2)/2=5
            nn.Flatten(),  # 展平层，将多维张量展平为一维
            nn.Linear(16*5*5,120),nn.Dropout(0.2),nn.ReLU(),  # 全连接层，输入特征数为16*5*5，输出特征数为120
            nn.Linear(120,84),nn.Dropout(0.2),nn.ReLU(),  # 全连接层，输入特征数为120，输出特征数为84
            nn.Linear(84,10)  # 全连接层，输入特征数为84，输出特征数为10，对应10个类别
        )
    
    def forward(self, X):
        return self.net(X)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式    
    losses = []  # ⽤于存储每个epoch的损失
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader): #每次load一个batch
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录损失            
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}]')
        
        # 记录每个epoch的平均损失        
        losses.append(epoch_loss / len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {losses[-1]:.4f}')
    
    return losses

# 测试模型
def test_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁⽤梯度计算        
          for images, labels in test_loader:    #每次load一个batch
            # 前向传播            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            
            # 统计正确预测的数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy


if __name__ == '__main__':
    # 下载并加载训练集和测试集
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 官方预计算值
])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    # 初始化模型
    model_le = LeNet()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_cls = optim.Adam(model_le.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 8
    losses_cls = train_model(model_le, train_loader, criterion, optimizer_cls, num_epochs=num_epochs)
    # 绘制损失曲线
    plt.plot(losses_cls, label='LeNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    # 测试模型
    acc = test_model(model_le, test_loader)
    print(f"Accuracy of LeNet: {acc:.2f}%")
    # 保存模型
    torch.save(model_le.state_dict(), f'LeNet_epoch{num_epochs}.pth')



