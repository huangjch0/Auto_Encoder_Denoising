# Auto_Encoder_Denoising
自编码去噪网络，刚开始用的随机梯度下降，效果并不是特别好，然后换成Adam，效果竟然比SGD好。
我觉得吧！大部分情况下SGD会优于Adam，但也存在例外，具体还要实验看看。
传统图片处理也有不少去噪的方法，用深度学习有点杀鸡用牛刀的感觉。有点鸡肋，就当是做个练习。

training_set:https://cs.stanford.edu/~acoates/stl10/

image_denoising:
训练脚本，会生成模型文件，加噪去噪后的对比图片，输出PNSR的平均提升

use.py：
做了个简单的ui交互界面，亲测实用。运行要加载模型文件'DAEmodel.pth'
