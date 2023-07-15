import torch 
from torch.utils import data
import numpy as np
class TestDataset(data.Dataset):
    def __init__(self):
#一些由2维向量表示的数据集
        self.Data = np.asarray([[1,2],[3,4],[2,1],[3,4],[4,5]]) 
#这是数据集合对应的标签
        self.Label = np.asarray([0,1,0,1,2])  
    def __getitem__(self,index):
        #numpy 转换成 Tensor
        txt = torch.from_numpy(self.Data[index])
        label = torch.tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)
#获取数据集中的数据
Test = TestDataset()
#相当于调用__getitem__(2),输出[2,1]
print(Test[2])
print(Test.__len__())