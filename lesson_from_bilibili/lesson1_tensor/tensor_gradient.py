import torch
#Author:    KDQ
#Date:      2023-7-8
#Reference: https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter01_getting-started/1_3_2_autograd_tutorial/
#Purpose:   学习pytorch中的张量梯度的概念以及如何开启和关闭自动求导
#Summary：  


###Step1: tensor变量默认是不打开自动求导的，需要手动打开，具体有如下两种方式
print("---------开启张量自动求导--------")
#Method1: 声明tensor变量的时候设置
x = torch.rand(5,3,requires_grad=True)
print(x)

#Method2: 手动设置tensor变量打开自动求导
x = torch.rand(5,3)
x.requires_grad_(True)
print(x)

###Step2: tensor有一个.grad_fn属性，该属性对应function其表示和存储了梯度完整的计算历史，如果没有开启自动求导，该属性
###默认是None，否则可以查看其地址
print("---------Tensor的梯度属性.grad_fn---------")
x = torch.ones(4,4) * 2
x.requires_grad_(True)
print("x没有梯度信息，因为x没有进行数学运算")
print(x.grad_fn)
print("y有梯度信息，因为它是来自于开启自动求导的张量数学运算而来，仔细体会")
y = x * 5
print(f"y = \n{y}")
print(y.grad_fn)
print("关闭x自动求导")
x.requires_grad_(False)
z = x * x
print(f"z.grad_fn =  {z.grad_fn}")


###Step3: 使用backward获取张量导数
print("----------用backward获取张量的梯度-----------")
x = torch.ones(4,4) * 2
x.requires_grad_(True)
y = x * 5
# z = 1/16 * (x * x * 5) 
# z'/x' = 10/16 * x|2 = 20 / 16
z = (y * x).mean()
print(f"z = {z}")
print(f"z grad_fn = {z.grad_fn}")
z.backward()
print(f"求z相对x的梯度：\n {x.grad}")

print("如果计算y的时候没有使能x的自动求导，那么z不会对y进行求导")
x = torch.ones(4,4) * 2
y = x * 5
x.requires_grad_(True)
# z = 1/16 * (10 * x) 
# z'/x' = 10/16 = 0.625
z = (y * x).mean()
print(f"z = {z}")
print(f"z grad_fn = {z.grad_fn}")
z.backward()
print(f"求z相对x的梯度：\n {x.grad}")



print("-------gradient------")
x = torch.randn(3, requires_grad=True)

y = x * 2
count = 0
print(x)
while y.data.norm() < 1000:
    count = count + 1
    y = y * 2
print(f"count = {count}")
print(y)
print("向量对向量求导，这个需要参考如下链接：{}".format("https://blog.csdn.net/comli_cn/article/details/104664494"))
gradients = torch.tensor([1., 1.0, 1.0], dtype=torch.float)
y.backward(gradients)
print(x.grad)

###Step4: 如果tensor变量已经使能了自动求导，但是又不想进行自动求导，可以使用“torch.no_grad()”

x = torch.rand(5,3,requires_grad = True)
y = x ** 2
print(y.requires_grad)

with  torch.no_grad():
    k = x ** 2
    print(k.requires_grad)
