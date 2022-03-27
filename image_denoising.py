import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import scipy.io as scio
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim


def read_image(data_path):  # 读取二进制文件并返回归一化后的图片
    with open(data_path, 'rb') as f:
        data1 = np.fromfile(f, dtype=np.uint8)
        image = np.reshape(data1, (-1, 3, 96, 96))
        images = np.transpose(image, (0, 3, 2, 1))
    return images/255.0


def gaussian_noise(images, sigma):  # 给图片添加高斯噪声
    sigma2 = sigma**2 / (255**2)  # 噪声方差，sigma越大噪声越多
    images_noisy = np.zeros_like(images)
    for ii in range(images.shape[0]):
        image = images[ii]
        noise_im = random_noise(image, mode='gaussian', var=sigma2, clip=True) # 使用random_noise方法给图像添加噪声
        images_noisy[ii] = noise_im
    return images_noisy  # 返回添加噪声后的图像


data_path = r'G:\data\STL10\stl10_binary\train_X.bin'  # 数据集路径，r表示防止转义
images = read_image(data_path)


# print('image.shape', images.shape)  #(5000,96,96,3)，是我们训练集的size
'''5000说明图像有5000张，96，96，3表示我们的图像是96*96*3的，96*96表示图像的宽和高，3表示深度'''
images_noise = gaussian_noise(images, 30)
# images_noise是添加了噪声的图片，images表示原图


# 显示原图
plt.figure(figsize=(6,6))
for ii in np.arange(36):
    plt.subplot(6, 6, ii+1)
    plt.imshow(images[ii, ...])
    plt.axis('off')
plt.show()


# 显示带噪声的图片
plt.figure(figsize=(6,6))
for ii in np.arange(36):
    plt.subplot(6, 6, ii+1)
    plt.imshow(images_noise[ii, ...])
    plt.axis('off')
plt.show()

# 我们要将带噪声的图像输入网络来训练，输出不带噪声的，输入是data_X,输出是data_Y
data_X = np.transpose(images_noise, (0, 3, 2, 1))  # 输入带噪声
data_Y = np.transpose(images, (0, 3, 2, 1))  # 输出不带噪声
X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.2, random_state=123)
# train_test_split对数据集进行划分，分为训练集和验证集，test_size=0.2表示测试集的数量站数据集的20%


# 数据转为tensor的格式
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 将训练集的输入和输出整合到一起
train_data = Data.TensorDataset(X_train, y_train)

# 将验证集的输入和输出整合到一起
val_data = Data.TensorDataset(X_val, y_val)

# 输出数据类型的格式和shape，当时调试用的，删了也无所谓
print(type(X_train))
print(type(y_train))

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_val.shape:', X_val.shape)
print('y_val.shape:', y_val.shape)


# 定义一个数据加载器用来训练网络，数据加载器中的数据是train_data
# batch_size=32 表示一次喂给网络32张图片
# shuffle=True 表示我们先打乱数据再喂给网络
# num_workers=4 表示使用4个线程
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 定义一个验证集加载器，参数与上同
val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=32,
    shuffle=True,
    num_workers=4
)


# 定义自编码去噪网络
class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()

        # Encoder，编码器网络结构
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
        )

        # Decoder，解码器网络结构
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder


# 实例化我们的网络
DAEmodel = DenoiseAutoEncoder()
print(DAEmodel)  # 输出网络结构看看

# 定义优化器，采用Adam算法，想详细了解的话可以看下这篇论文：https://arxiv.org/pdf/1412.6980.pdf，初始学习率为0.0003
optimizer = optim.Adam(DAEmodel.parameters(), lr=0.0003)
loss_func = nn.MSELoss()  # 定义损失函数为均方损失函数，loss(x,y)=(x-y)^2


train_loss = []
val_loss = []

for epoch in range(10):  # 训练10轮
    train_loss_epoch = 0
    val_loss_epoch = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        DAEmodel.train()  # 网络为训练模式
        _, output = DAEmodel(b_x)  # _加密，解密输出output
        loss = loss_func(output, b_y)  # b_y 为不加噪声原图像，b_x为加了噪声的图片

        optimizer.zero_grad()  # 1，清空参数梯度
        loss.backward()  # 2，求导
        optimizer.step()  # 3，向求导（导数）的方向更新

        train_loss_epoch += loss.item() * b_x.size(0)
    train_loss.append(train_loss_epoch / len(train_data))  # 训练集每一轮的loss保存在train_loss中

    for step, (b_x, b_y) in enumerate(val_loader):
        DAEmodel.eval()  # 网络开启验证模式，关闭所有梯度
        _, output = DAEmodel(b_x)  # 加密，解密，_表示我们忽略了这个返回值，因为后面没有用到
        loss = loss_func(output, b_y)
        val_loss_epoch += loss.item() * b_x.size(0)
    val_loss.append(val_loss_epoch / len(val_data))  # 验证集每一轮的loss保存在val_loss中

    print('*************************************************', 'finished:', epoch+1)
    print('train_loss:', train_loss)
    print('val_loss:', val_loss)

# 保存模型，设置_use_new_zipfile_serialization=False，可以让模型在不同操作系统上读取
torch.save(DAEmodel, 'DAEmodel.pth', _use_new_zipfile_serialization=False)

# 将训练集和验证集的loss保存为mat格式
scio.savemat('train_loss.mat', {'train_loss': train_loss})
scio.savemat('val_loss.mat', {'val_loss': val_loss})


# 带噪图
im = X_val[1, ...].unsqueeze(0)
imnose = np.transpose(im.data.numpy(), (0, 3, 2, 1))[0, ...]

# 去噪
DAEmodel.eval()
_, output = DAEmodel(im)
imde = np.transpose(output.data.numpy(), (0, 3, 2, 1))[0, ...]

# 原图
im = y_val[1, ...].unsqueeze(0)
imor = np.transpose(im.data.numpy(), (0, 3, 2, 1))[0, ...]

# 计算PNSR
'''
峰值信噪比：PNSR（peak signal noise ration）
来评价两个图片的相似程度，越大表明两个图片越相似
'''
print('加噪后的PNSR：', peak_signal_noise_ratio(imor, imnose), 'dB')
print('去噪后的：', peak_signal_noise_ratio(imor, imde), 'dB')

# 画出原图，加噪声后的，去噪声后的对比图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(imor)
plt.axis('off')
plt.title('Origin image')
plt.subplot(1, 3, 2)
plt.imshow(imnose)
plt.axis('off')
plt.title('noise image sigma=30')
plt.subplot(1, 3, 3)
plt.imshow(imde)
plt.axis('off')
plt.title('Denoise image')
plt.savefig('compare.jpg')  # 保存图片名为compare.jpg
plt.show()


# 计算验证集图片PNSR的平均提升
psnr_val = []
DAEmodel.eval()
for ii in range(X_val.shape[0]):
    im = X_val[ii, ...].unsqueeze(0)  # X_val 带噪
    imnose = np.transpose(im.data.numpy(), (0, 3, 2, 1))[0, ...]

    # 去噪
    _, output = DAEmodel(im)
    imde = np.transpose(output.data.numpy(), (0, 3, 2, 1))[0, ...]

    # 输出
    im = y_val[ii, ...]
    imor = im.unsqueeze(0)
    imor = np.transpose(imor.data.numpy(), (0, 3, 2, 1))
    imor = imor[0, ...]
    psnr_val.append(peak_signal_noise_ratio(imor, imde) - peak_signal_noise_ratio(imor, imnose))

print('PSNR平均提升:', np.mean(psnr_val))

