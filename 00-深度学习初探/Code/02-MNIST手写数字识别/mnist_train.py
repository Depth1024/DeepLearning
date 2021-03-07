import torch
from torch import nn    # nn用于完成神经网络相关的工作
from torch.nn import functional as F  # 常用函数库
from torch import optim  # 优化工具包

import torchvision  # 视觉工具包
from matplotlib import  pyplot as plt  # 绘图工具包

from utils import plot_image,plot_curve,one_hot

batch_size = 512   #batch_size的概念：控制每次并行处理的图片数量

# Step1:Load Data  加载mnist数据集
# 加载训练集train
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',  # 指定mnist数据集的下载路径
                               train=True,  # 指定下载的数据集是60k的train部分 or 10k的test部分
                               download=True,   # True：若当前文件中没有mnist文件，则自动从网上拉取
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(), # 将Numpy格式的文件转换为Tensor
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 正则化处理，不必要，但可以适当提高性能
                                   # 因为像素点的灰度值x in [0,1]，始终在0右侧
                                   # 故对x进行正则化处理，令(x - 0.1307)/0.3081，保证数据分布在0附近均匀分布，便于神经网络进行优化
                               ])),
    batch_size=batch_size,  # 并行处理数量
    shuffle=True   # 加载时进行“随机打散”
)
# 加载测试集test
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',
                               train=False,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size,
    shuffle=False
)

# # 利用迭代器，从train_loader中取sample
# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), x.max()) # 512张图片，1个通道，28行，28列; x的min和max在0附近晃动（正则化处理的结果）
# # 把加载的图片显示出来（具体在utils中实现，此处略）
# plot_image(x, y, 'image sample')


# Step2：Build Model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x

# Step3:Train
net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []

for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10==0:
            print(epoch, batch_idx, loss.item())
# 可视化梯度下降过程
plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]

# Step4:Test
total_correct = 0
for x,y in test_loader:
    x  = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')




