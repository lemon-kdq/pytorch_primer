import torch
import torchvision 
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from torch import optim
from matplotlib import pyplot as plt
from utils import  plot_curve, plot_image,  one_hot

#Step1: Load Data
print("")
batch_size = 512
train_loader = data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))])),  #归一化分布,其中0.1307是mnist数据集的均值，0.3081是其标准差
                                       batch_size=batch_size,shuffle=True) #batch_size并行处理的图片数量，shuffle打散图片
print("Tip1:torchvision.transforms.Normalize——\
      可以对图片进行标准化处理，使其均值为0标准差为1，训练出来的模型精度高和泛化能力强")
test_loader = data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))])),
                                       batch_size=batch_size,shuffle=False)

x, y = next(iter(train_loader))
print(x.shape,y.shape,x.mean(),x.std())

plot_image(x,y,'image sample')
#Step2: Build Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # xw + b
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        # x : [b,1,28,28]
        # h1 = relu(w1x + b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(w2h1 + b2)
        x = F.relu(self.fc2(x))
        # h3 = w3h2 + b3
        x = self.fc3(x)

        return x
    
    



#Step3: Train

#Step4: Test
