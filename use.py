import torch
from skimage.util import random_noise
import numpy as np
import tkinter as tk
from tkinter import filedialog
import easygui as a
from PIL import Image
import matplotlib.pyplot as plt


class DenoiseAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()

        # Encoder，编码器网络结构
        self.Encoder =torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64, 64, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(2, 2),
        torch.nn.BatchNorm2d(64),
        torch.nn.Conv2d(64, 128, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 128, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(128),
        torch.nn.Conv2d(128, 256, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(2, 2),
        torch.nn.BatchNorm2d(256),
        )

        # Decoder，解码器网络结构
        self.Decoder =torch.nn.Sequential(
        torch.nn.ConvTranspose2d(256, 128, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(128),
        torch.nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(128),
        torch.nn.ConvTranspose2d(128, 64, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(64),
        torch.nn.ConvTranspose2d(64, 32, 3, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(32),
        torch.nn.ConvTranspose2d(32, 32, 3, 1, 1),
        torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
        torch.nn.ReLU(),

        torch.nn.BatchNorm2d(16),
        torch.nn.ConvTranspose2d(16, 3, 3, 1, 1),
        torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder


DAEmodel = torch.load('DAEmodel.pth')  # 加载模型


def function_selection():
    choice = a.ccbox(msg='选择功能', title='功能选项', choices=('图片加噪', '图片去噪'))
    return choice


while True:
    choice = function_selection()  # 先选择加噪还是去噪
    if choice == None:
        break

    root = tk.Tk()
    root.withdraw()

    imgpath = filedialog.askopenfilename()  # 获得图片路径
    img = Image.open(imgpath)
    img = np.array(img)

    if imgpath.endswith('.jpg') or imgpath.endswith('.png'):  # 如果选的是图片继续执行

        if choice:  # 加噪处理
            sigma = a.integerbox(msg='输入噪声方差sigama(0-255)', default=30, lowerbound=0, upperbound=255)
            sigma = sigma**2 / (255**2)
            noise_im = random_noise(img, mode='gaussian', var=sigma, clip=True)  # 添加噪声
            plt.imshow(noise_im)
            plt.axis('off')
            plt.savefig('noise.jpg')  # 保存加噪后的图像,
            a.msgbox(msg='加噪图像已保存到当前路径', title='save')

        elif not choice:  # 去噪处理
            X = np.transpose(img, (2, 1, 0)) / 255  # 先归一化图像
            tensor_img = torch.tensor(X, dtype=torch.float32)  # 图像数据转为tensor
            _, output = DAEmodel(tensor_img.unsqueeze(0))  # 输入网络，得到输出
            output = np.transpose(output.data.numpy(), (0, 3, 2, 1))[0, ...]
            plt.imshow(output)
            plt.axis('off')
            plt.savefig('de-noise.jpg')
            a.msgbox(msg='去噪图像已保存到当前路径', title='save')

        elif choice == None:
            break

    else:
        a.msgbox('文件不是图片！！！')

