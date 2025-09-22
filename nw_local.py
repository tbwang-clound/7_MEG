# 导入 PyTorch 库，用于深度学习模型的构建和训练
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter

# 定义基本块类，继承自 nn.Module
class BasicBlock(nn.Module):
    # 扩展因子，用于残差连接
    expansion = 1

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        """
        初始化 BasicBlock 类。

        参数:
        channels (int): 输入和输出的通道数。
        kernel_size (int): 卷积核的大小，默认为 1。
        stride (int): 卷积的步长，默认为 1。
        padding (int): 卷积的填充大小，默认为 0。
        dilation (int): 卷积的扩张率，默认为 1。
        bias (bool): 是否使用偏置，默认为 False。
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride, padding, dilation, bias=bias)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm1d(channels)
        # 第二个卷积层，填充和扩张率增加
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride, padding + 3, dilation + 3, bias=bias)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm1d(channels)

        # 捷径连接，默认为空序列
        self.shortcut = nn.Sequential()
        # 如果步长不为 1 或者通道数不匹配，需要调整捷径连接
        if stride != 1 or channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * channels)
            )

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 第一个卷积层和批量归一化，然后使用 ReLU 激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积层和批量归一化
        out = self.bn2(self.conv2(out))
        # 加上捷径连接的输出
        out += self.shortcut(x)
        # 使用 ReLU 激活函数
        out = F.relu(out)
        return out

# 定义 SE 层类，继承自 nn.Module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化 SELayer 类。

        参数:
        channel (int): 输入通道数。
        reduction (int): 缩减因子，默认为 16。
        """
        super(SELayer, self).__init__()
        # 自适应平均池化层，输出大小为 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层序列
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 获取批量大小、通道数和序列长度
        b, c, _ = x.size()
        # 自适应平均池化并调整形状
        y = self.avg_pool(x).view(b, c)
        # 通过全连接层并调整形状
        y = self.fc(y).view(b, c, 1)
        # 将输入张量与输出相乘
        return x * y.expand_as(x)

# 定义卷积、批量归一化和 ReLU 层类，继承自 nn.Module
class Conv1dBnRelu_jy(nn.Module):
    # 输入和输出的张量形状注释
    # in:torch.Size([64, 80, 501]); out:torch.Size([64, 512, 501])
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        """
        初始化 Conv1dBnRelu_jy 类。

        参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核的大小，默认为 1。
        stride (int): 卷积的步长，默认为 1。
        padding (int): 卷积的填充大小，默认为 0。
        dilation (int): 卷积的扩张率，默认为 1。
        bias (bool): 是否使用偏置，默认为 False。
        """
        super().__init__()
        # 卷积层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        # 批量归一化层
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 卷积、批量归一化，然后使用 ReLU 激活函数
        return F.relu(self.bn(self.conv(x)))

# 定义 SE 基本块函数
def SE_BasicBlock_jy(channels, kernel_size, stride, padding, dilation):
    """
    创建一个包含卷积、基本块、卷积和 SE 层的序列。

    参数:
    channels (int): 输入和输出的通道数。
    kernel_size (int): 卷积核的大小。
    stride (int): 卷积的步长。
    padding (int): 卷积的填充大小。
    dilation (int): 卷积的扩张率。

    返回:
    nn.Sequential: 包含多个层的序列。
    """
    return nn.Sequential(
        # 第一个卷积、批量归一化和 ReLU 层
        Conv1dBnRelu_jy(channels, channels, kernel_size=1, stride=1, padding=0),
        # 基本块
        BasicBlock(channels, kernel_size, stride, padding, dilation),
        # 第二个卷积、批量归一化和 ReLU 层
        Conv1dBnRelu_jy(channels, channels, kernel_size=1, stride=1, padding=0),
        # SE 层
        SELayer(channels)
    )

# 定义注意力统计池化层类，继承自 nn.Module
class AttentiveStatsPool_ori(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        """
        初始化 AttentiveStatsPool_ori 类。

        参数:
        in_dim (int): 输入维度。
        bottleneck_dim (int): 瓶颈维度。
        """
        super().__init__()
        # 第一个卷积层，相当于论文中的 W 和 b
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        # 第二个卷积层，相当于论文中的 V 和 k
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 使用 tanh 激活函数
        alpha = torch.tanh(self.linear1(x))
        # 使用 softmax 函数计算注意力权重
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        # 计算加权平均值
        mean = torch.sum(alpha * x, dim=2)
        # 计算残差
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        # 计算标准差，确保数值稳定性
        std = torch.sqrt(residuals.clamp(min=1e-9))
        # 将平均值和标准差拼接在一起
        return torch.cat([mean, std], dim=1)

# 定义局部头部类，继承自 nn.Module
class local_head(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192):
        """
        初始化 local_head 类。

        参数:
        input_size (int): 输入大小，默认为 80。
        channels (int): 通道数，默认为 512。
        embd_dim (int): 嵌入维度，默认为 192。
        """
        super().__init__()
        # 第一个卷积、批量归一化和 ReLU 层
        self.layer1 = Conv1dBnRelu_jy(input_size, channels, kernel_size=5, padding=2, dilation=1)
        # 第一个 SE 基本块
        self.layer2 = SE_BasicBlock_jy(channels, kernel_size=3, stride=1, padding=2, dilation=2)
        # 第二个 SE 基本块
        self.layer3 = SE_BasicBlock_jy(channels, kernel_size=3, stride=1, padding=3, dilation=3)

        # 拼接后的通道数
        cat_channels = channels * 2
        # 输出通道数
        out_channels = cat_channels * 2
        # 卷积层
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        # 注意力统计池化层
        self.pooling = AttentiveStatsPool_ori(cat_channels, 128)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 第一个线性层
        self.linear_r = nn.Linear(out_channels, embd_dim)
        # 第二个批量归一化层
        self.bn2_r = nn.BatchNorm1d(embd_dim)

        # 第二个线性层
        self.linear_z = nn.Linear(out_channels, embd_dim)
        # 第三个批量归一化层
        self.bn2_z = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        tuple: 包含两个输出张量的元组。
        """
        # 通过第一个卷积、批量归一化和 ReLU 层
        out1 = self.layer1(x)
        # 通过第一个 SE 基本块
        out2 = self.layer2(out1)
        # 通过第二个 SE 基本块
        out3 = self.layer3(out2)

        # 拼接第二个和第三个输出
        out = torch.cat([out2, out3], dim=1)
        # 卷积并使用 ReLU 激活函数
        out = F.relu(self.conv(out))
        # 注意力统计池化和批量归一化
        out = self.bn1(self.pooling(out))

        # 通过第一个线性层和批量归一化层
        out_r = self.bn2_r(self.linear_r(out))
        # 通过第二个线性层和批量归一化层
        out_z = self.bn2_z(self.linear_z(out))
        return out_r, out_z

# 定义局部网络类，继承自 nn.Module
class local_network(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192):
        """
        初始化 local_network 类。

        参数:
        input_size (int): 输入大小，默认为 80。
        channels (int): 通道数，默认为 512。
        embd_dim (int): 嵌入维度，默认为 192。
        """
        super().__init__()
        # 局部头部
        self.local_head = local_head(input_size, channels, embd_dim)
        # 第一个全连接层
        self.fc_r = nn.Linear(embd_dim, 1)
        # 第二个全连接层
        self.fc_z = nn.Linear(embd_dim, 1)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        tuple: 包含两个输出张量的元组。
        """
        # 通过局部头部
        out_r, out_z = self.local_head(x)
        # 通过第一个全连接层
        out_r = self.fc_r(out_r)
        # 通过第二个全连接层
        out_z = self.fc_z(out_z)
        # 移除最后一个维度
        out_r = out_r.squeeze(-1)
        out_z = out_z.squeeze(-1)
        # 注释掉的代码，用于拼接两个输出
        # out = torch.cat([out_r, out_z], dim=1)
        return out_r, out_z

if __name__ == '__main__':
    # 初始化模型，指定输入大小、通道数和嵌入维度
    model = local_network(input_size=201, channels=512, embd_dim=192)
    # 生成随机输入数据，形状为 (64, 201, 512)
    input_tensor = torch.randn(64, 201, 512)
    # 前向传播
    output_r, output_z = model(input_tensor)
    # 打印输出结果的形状
    print("Output shape:", output_r.shape)

