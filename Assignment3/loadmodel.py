import torch
from Assignment3.HW3_CNN import LeNet
from torchvision import datasets,transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
#加载模型
model = LeNet()
model.load_state_dict(torch.load('LeNet_epoch6.pth'))
#测试模型
model.eval()
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

