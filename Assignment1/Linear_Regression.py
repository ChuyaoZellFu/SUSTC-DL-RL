import torch
from torch.utils.data import random_split,DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, N, w1_ref, w2_ref, b_ref, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.w1_ref = w1_ref
        self.w2_ref = w2_ref
        self.b_ref = b_ref
        self.x_1 = torch.rand(N)
        self.x_2 = torch.rand(N)
        self.noise = torch.randn(N)*0.2
        self.y = w1_ref * self.x_1 + w2_ref * self.x_2 + b_ref + self.noise
        
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {'input1':self.x_1[idx], 'input2':self.x_2[idx], 'target':self.y[idx]}
    
class LinearRegression(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features=2, out_features=1, bias=True)
    def forward(self, x):
        x = self.fc1(x)
        return x
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mydataset = MyDataset(N=1000, w1_ref=2, w2_ref=3, b_ref=5)
    train_size = int(0.8 * len(mydataset))
    val_size = len(mydataset) - train_size
    train_dataset,val_dataset = random_split(mydataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = LinearRegression()
    nn.init.normal_(model.fc1.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc1.bias)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005)
    epochs = 70
    training_loss_list = []
    validation_loss_list = []
    for i in range(epochs):
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input1 = batch['input1'].to(device)
            input2 = batch['input2'].to(device)
            target = batch['target'].to(device)
            input = torch.stack([input1, input2], dim=1)
            prediction = model(input)
            loss = criterion(prediction, target)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'epoch {i+1}/{epochs}, training loss: {training_loss/len(train_loader)}')
        training_loss_list.append(training_loss/len(train_loader))
        
        model.eval()
        total_loss = 0.0
        for batch in val_loader:
            input1 = batch['input1'].to(device)
            input2 = batch['input2'].to(device)
            target = batch['target'].to(device)
            input = torch.stack([input1, input2], dim=1)
            with torch.no_grad():
                prediction = model(input)
                loss = criterion(prediction, target)
            total_loss += loss.cpu().item() * len(input)

        print(f'epoch {i+1}/{epochs}, validation loss: {total_loss/len(val_dataset)}')
        validation_loss_list.append(total_loss/len(val_dataset))
    
    print(f"w1:{model.fc1.weight[0][0].item()}, w2:{model.fc1.weight[0][1].item()}, b:{model.fc1.bias.item()}")
    y_true = []
    y_pred = []
    for batch in val_loader:
        input1 = batch['input1'].to(device)
        input2 = batch['input2'].to(device)
        target = batch['target'].to(device)
        input = torch.stack([input1, input2], dim=1)
        with torch.no_grad():
            prediction = model(input)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(prediction.cpu().numpy())
    r2 = r2_score(y_true, y_pred)
    print(f"R²值: {r2:.4f}")
    
    plt.plot(training_loss_list)
    plt.plot(validation_loss_list)
    plt.legend(['training loss', 'validation loss'])
    plt.show()

            

