import torch
import numpy as np
#Author:    KDQ
#Date:      2023-7-8
#Reference: https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter01_getting-started/1_3_1_tensor_tutorial/
#Purpose:   学习pytorch中的张量的概念以及常规操作
#Summary：  张量类似numpy中的ndarrays，但是张量可以使用GPU计算

##Step1: 创建和初始化张量
print("-------张量的创建和赋值-------")
#Method1: 创建一个5*3的张量，但不做初始化
x = torch.empty(5,3)
print(x)

#Method2：创建一个3*5的张量并随机初始化
x = torch.rand(3,5)
print(x)

#Method3: 创建一个0填充的张量且用long类型
x = torch.zeros(5,3,dtype=torch.long)
print(x)

#Method4: 用data直接创建张量
x = torch.tensor([[1,3],[4,6]],dtype=torch.float64)
print(x)

#Method5: 用现有的张量创建张量，那么该张量则具有现有张量的属性（size,dtype）,除非属性被覆盖
x = x.new_ones(5,3,dtype=torch.double)
print(x)
x = torch.rand_like(x,dtype=torch.float)
print(x)

#获取Size
print(x.size())

##Step2: 数学运算
print("-------张量的加法-------")
#Operation1：加法——按位相加，下面三种加法方式结果相同
y = torch.rand_like(x)
z1 = x + y
z2 = torch.add(x,y)
z3 = torch.empty(x.size())
torch.add(x,y,out=z3)
print(y)
print(z1)
print(z2)
print(z3)
#Operation2: 自加——自己即作加数也做和
y.add_(x)
print(f'y = {y}')

##Step3: 索引和reshape
print("-------张量的索引和reshape-------")
#Method1： 索引——类似numpy索引
x = torch.rand(4,4)
print(x)
print(x[:,2])
print(x[3,:])

#Method2: Multi
x = torch.ones(4,4) * 2
y = torch.ones(4,4) * 2
print(x)
print(y)
z = x * y
print(f"点乘：z =\n {z}")
k = x @ y
print(f"矩阵乘法：k =\n {k}")

#Method3： reshape
y = x.view(16)
print(y)
print(x.view(2,-1))
print(x.view(-1,2))

##Step4: tensor转numpy array
print("-------张量和numpy相互转换-------")
x = torch.rand(5,3)
y = x.numpy()
print(y)
x.add_(2)
print(x)
print(y)

x = np.random.rand(16).reshape(4,4)
y = torch.from_numpy(x)
np.add(x,2.,out=x)
print(x)
print(y)
print("注意这里y也跟着变了，说明tensor和numpy相互转换的时候只是指针赋值而非数值拷贝")

##Step5: CUDA张量
print("-------CUDA中操作张量-------")
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.rand(10,10)
    y = torch.ones_like(x,device=device)
    print(f"直接用GPU创建张量: \n{y}")
    x = x.to(device)
    print(f"也可以用.to去讲张量移动到cuda中:\n{x}")
    z = x + y
    print(z)
    print(z.to("cpu",torch.double))