import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataloader
import seaborn as sns

X_train,y_train,X_test,y_test=dataloader.read_bci_data()
dataset=TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
train_loader=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=4)
dataset=TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
test_loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=4)

sns.set()
plt.figure(figsize=(10,2))
plt.plot(X_train[0,0,0])
plt.figure(figsize=(10,2))
plt.plot(X_train[0,0,1])

class DeepConvNet(nn.Module):
    def __init__(self,activation=nn.ELU()):
        super(DeepConvNet,self).__init__()
        self.conv0=nn.Conv2d(1,25,kernel_size=(1,5))
        self.conv1 = nn.Sequential(
                nn.Conv2d(25,25,kernel_size=(2,1)),
                nn.BatchNorm2d(25,eps=1e-5,momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(25,50,kernel_size=(1,5)),
                nn.BatchNorm2d(50,eps=1e-5,momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
        self.conv3 = nn.Sequential(
                nn.Conv2d(50,100,kernel_size=(1,5)),
                nn.BatchNorm2d(100,eps=1e-5,momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
        self.conv4 = nn.Sequential(
                nn.Conv2d(100,200,kernel_size=(1,5)),
                nn.BatchNorm2d(200,eps=1e-5,momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(p=0.5)
            )
        self.classify=nn.Linear(8600,2)
    def forward(self,X):
        out=self.conv0(X)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=out.view(out.shape[0],-1)
        neurons=out.shape[0]
        out=self.classify(out)
        return out

lr=0.001
epochs=320

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
activations={'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(0.05),'ELU':nn.ELU(0.8)}

Loss=nn.CrossEntropyLoss()
df=pd.DataFrame()
df['epoch']=range(1,epochs+1)
best_model_wts={'ReLU':None,'LeakyReLU':None,'ELU':None}
best_evaluated_acc={'ReLU':0,'LeakyReLU':0,'ELU':0}
for name,activation in activations.items():
    model=DeepConvNet(activation)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.025)
    acc_train=list()
    acc_test=list()
    for epoch in range(1,epochs+1):
        model.train()
        total_loss=0
        correct=0
        for idx,(data,target) in enumerate(train_loader):
            data=data.to(device,dtype=torch.float)
            target=target.to(device,dtype=torch.long)
            predict=model(data)
            loss=Loss(predict,target)
            total_loss+=loss.item()
            correct+=predict.max(dim=1)[1].eq(target).sum().item()
            optimizer.zero_grad()
            loss.backward()  # bp
            optimizer.step()
        total_loss/=len(train_loader.dataset)
        correct=100.*correct/len(train_loader.dataset)
        acc_train.append(correct)
				
				
        model.eval()
        correct=0
        for idx,(data,target) in enumerate(test_loader):
            data=data.to(device,dtype=torch.float)
            target=target.to(device,dtype=torch.long)
            predict=model(data)
            correct+=predict.max(dim=1)[1].eq(target).sum().item()
        correct=100.*correct/len(test_loader.dataset)
        acc_test.append(correct)
        if correct>best_evaluated_acc[name]:
            best_evaluated_acc[name]=correct
            best_model_wts[name]=copy.deepcopy(model.state_dict())
    df[name+'_train']=acc_train
    df[name+'_test']=acc_test
		
for name,model_wts in best_model_wts.items():
    PATH = 'DeepConvNet_'+ name + '.pt'
    torch.save(model_wts,PATH)

for column in df.columns[1:]:
    print(f'{column} highest accuracy: {df[column].max()}')
		
		
fig=plt.figure(figsize=(10,6))
plt.title("DeepConvNet")
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
for name in df.columns[1:]:
    plt.plot('epoch',name,data=df)
plt.legend()
fig.savefig('DeepConvNet.png')
