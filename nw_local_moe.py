# 导入 PyTorch 库，用于深度学习相关操作
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的神经网络函数模块
import torch.nn.functional as F
# 导入 PyTorch 的初始化模块
from torch.nn import init

# 导入 PyTorch 卷积层的基类
from torch.nn.modules.conv import _ConvNd
# 导入 PyTorch 用于处理二维参数的工具
from torch.nn.modules.utils import _pair

# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 PyTorch 的自动求导变量模块（旧版本，现多使用张量直接处理）
from torch.autograd import Variable
# 导入 PyTorch 的参数模块
from torch.nn import Parameter

# 定义基本块类，继承自 PyTorch 的 nn.Module
class BasicBlock(nn.Module):
    # 定义扩张因子，用于残差连接时通道数的调整
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
        # 调用父类的构造函数
        super(BasicBlock, self).__init__()
        # 定义第一个一维卷积层
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride, padding, dilation, bias=bias)
        # 定义第一个一维批量归一化层
        self.bn1 = nn.BatchNorm1d(channels)
        # 定义第二个一维卷积层，填充和扩张率有所增加
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride, padding + 3, dilation + 3, bias=bias)
        # 定义第二个一维批量归一化层
        self.bn2 = nn.BatchNorm1d(channels)

        # 定义捷径连接，初始为空序列
        self.shortcut = nn.Sequential()
        # 如果步长不为 1 或者通道数不匹配，需要调整捷径连接
        if stride != 1 or channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                # 1x1 卷积层用于调整通道数和特征图大小
                nn.Conv1d(channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                # 批量归一化层
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
        # 通过第一个卷积层和批量归一化层，然后使用 ReLU 激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 通过第二个卷积层和批量归一化层
        out = self.bn2(self.conv2(out))
        # 将输出与捷径连接的输出相加
        out += self.shortcut(x)
        # 再次使用 ReLU 激活函数
        out = F.relu(out)
        return out

# 定义 SE 层类，继承自 PyTorch 的 nn.Module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化 SELayer 类。

        参数:
        channel (int): 输入通道数。
        reduction (int): 缩减因子，默认为 16。
        """
        # 调用父类的构造函数
        super(SELayer, self).__init__()
        # 定义自适应平均池化层，输出大小为 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 定义全连接层序列
        self.fc = nn.Sequential(
            # 第一个全连接层，输入特征数为 channel，输出特征数为 channel // reduction
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU 激活函数
            nn.ReLU(inplace=True),
            # 第二个全连接层，输入特征数为 channel // reduction，输出特征数为 channel
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid 激活函数
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
        # 获取输入张量的批量大小、通道数和序列长度
        b, c, _ = x.size()
        # 通过自适应平均池化层并调整形状
        y = self.avg_pool(x).view(b, c)
        # 通过全连接层序列并调整形状
        y = self.fc(y).view(b, c, 1)
        # 将输入张量与输出相乘
        return x * y.expand_as(x)

# 定义包含卷积、批量归一化和 ReLU 激活函数的模块类
class Conv1dBnRelu_jy(nn.Module):
    # 注释说明输入和输出的张量形状
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
        # 调用父类的构造函数
        super().__init__()
        # 定义一维卷积层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        # 定义一维批量归一化层
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 通过卷积层和批量归一化层，然后使用 ReLU 激活函数
        return F.relu(self.bn(self.conv(x)))

# 定义 SE 基本块函数，返回一个包含多个层的序列
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

# 定义注意力统计池化层类
class AttentiveStatsPool_ori(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        """
        初始化 AttentiveStatsPool_ori 类。

        参数:
        in_dim (int): 输入维度。
        bottleneck_dim (int): 瓶颈维度。
        """
        # 调用父类的构造函数
        super().__init__()
        # 定义第一个一维卷积层，相当于论文中的 W 和 b
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        # 定义第二个一维卷积层，相当于论文中的 V 和 k
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 通过第一个卷积层并使用 tanh 激活函数
        alpha = torch.tanh(self.linear1(x))
        # 通过第二个卷积层并使用 softmax 函数计算注意力权重
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        # 计算加权平均值
        mean = torch.sum(alpha * x, dim=2)
        # 计算残差
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        # 计算标准差，确保数值稳定性
        std = torch.sqrt(residuals.clamp(min=1e-9))
        # 将平均值和标准差拼接在一起
        return torch.cat([mean, std], dim=1)

# 定义局部头部类
class local_head(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192):
        """
        初始化 local_head 类。

        参数:
        input_size (int): 输入大小，默认为 80。
        channels (int): 通道数，默认为 512。
        embd_dim (int): 嵌入维度，默认为 192。
        """
        # 调用父类的构造函数
        super().__init__()
        # 定义第一个卷积、批量归一化和 ReLU 层
        self.layer1 = Conv1dBnRelu_jy(input_size, channels, kernel_size=5, padding=2, dilation=1)
        # 定义第一个 SE 基本块
        self.layer2 = SE_BasicBlock_jy(channels, kernel_size=3, stride=1, padding=2, dilation=2)
        # 定义第二个 SE 基本块
        self.layer3 = SE_BasicBlock_jy(channels, kernel_size=3, stride=1, padding=3, dilation=3)

        # 拼接后的通道数
        cat_channels = channels * 2
        # 输出通道数
        out_channels = cat_channels * 2
        # 定义卷积层
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        # 定义注意力统计池化层
        self.pooling = AttentiveStatsPool_ori(cat_channels, 128)
        # 定义第一个批量归一化层
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 定义第一个线性层
        self.linear_r = nn.Linear(out_channels, embd_dim)
        # 定义第二个批量归一化层
        self.bn2_r = nn.BatchNorm1d(embd_dim)

        # 定义第二个线性层
        self.linear_z = nn.Linear(out_channels, embd_dim)
        # 定义第三个批量归一化层
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

        # 将第二个和第三个输出拼接在一起
        out = torch.cat([out2, out3], dim=1)
        # 通过卷积层并使用 ReLU 激活函数
        out = F.relu(self.conv(out))
        # 通过注意力统计池化层和批量归一化层
        out = self.bn1(self.pooling(out))

        # 通过第一个线性层和批量归一化层
        out_r = self.bn2_r(self.linear_r(out))
        # 通过第二个线性层和批量归一化层
        out_z = self.bn2_z(self.linear_z(out))
        return out_r, out_z

# 定义局部网络类
class local_network(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192, num_experts=5, k=3):
        """
        初始化 local_network 类。

        参数:
        input_size (int): 输入大小，默认为 80。
        channels (int): 通道数，默认为 512。
        embd_dim (int): 嵌入维度，默认为 192。
        num_experts (int): 专家网络的数量，默认为 5。
        k (int): 选择的前 k 个专家网络的数量，默认为 3。
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化局部头部
        self.local_head = local_head(input_size, channels, embd_dim)
        # 专家网络的数量
        self.num_experts = num_experts
        # 选择的前 k 个专家网络的数量
        self.k = k
        # 专家网络
        # 定义一个模块列表，包含 num_experts 个全连接层，输入特征数为 embd_dim，输出特征数为 1
        self.experts_r = nn.ModuleList([nn.Linear(embd_dim, 1) for _ in range(num_experts)])
        # 定义一个模块列表，包含 num_experts 个全连接层，输入特征数为 embd_dim，输出特征数为 1
        self.experts_z = nn.ModuleList([nn.Linear(embd_dim, 1) for _ in range(num_experts)])
        # 门控网络
        # 定义一个全连接层，输入特征数为 embd_dim，输出特征数为 num_experts
        self.gate_r = nn.Linear(embd_dim, num_experts)
        # 定义一个全连接层，输入特征数为 embd_dim，输出特征数为 num_experts
        self.gate_z = nn.Linear(embd_dim, num_experts)

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
        # 计算门控权重
        # 通过门控网络并使用 softmax 函数计算权重
        gate_weights_r = F.softmax(self.gate_r(out_r), dim=-1)
        # 通过门控网络并使用 softmax 函数计算权重
        gate_weights_z = F.softmax(self.gate_z(out_z), dim=-1)
        # Top-K 选择
        # 选择门控权重中前 k 大的索引
        topk_indices_r = torch.topk(gate_weights_r, self.k, dim=-1).indices
        # 选择门控权重中前 k 大的索引
        topk_indices_z = torch.topk(gate_weights_z, self.k, dim=-1).indices
        # 专家网络输出
        # 计算每个专家网络的输出
        expert_outputs_r = [expert(out_r) for expert in self.experts_r]
        # 计算每个专家网络的输出
        expert_outputs_z = [expert(out_z) for expert in self.experts_z]
        # 组合专家输出
        # 初始化最终输出为 0
        final_output_r = 0
        # 初始化最终输出为 0
        final_output_z = 0
        # 遍历前 k 个专家网络的索引
        for i in range(self.k):
            # 扩展索引以匹配专家输出的形状
            index_r = topk_indices_r[:, i].unsqueeze(-1)
            # 扩展索引以匹配专家输出的形状
            index_z = topk_indices_z[:, i].unsqueeze(-1)
            # 使用 gather 函数获取对应门控权重
            weight_r = gate_weights_r.gather(1, index_r)
            # 使用 gather 函数获取对应门控权重
            weight_z = gate_weights_z.gather(1, index_z)
            # 使用 gather 函数获取对应专家网络的输出并去除维度
            output_r = torch.gather(torch.stack(expert_outputs_r, dim=1), 1, index_r.unsqueeze(-1)).squeeze(1)
            # 使用 gather 函数获取对应专家网络的输出并去除维度
            output_z = torch.gather(torch.stack(expert_outputs_z, dim=1), 1, index_z.unsqueeze(-1)).squeeze(1)
            # 加权相加
            final_output_r += weight_r * output_r
            # 加权相加
            final_output_z += weight_z * output_z
        # 去除最后一个维度
        final_output_r = final_output_r.squeeze(-1)
        # 去除最后一个维度
        final_output_z = final_output_z.squeeze(-1)
        return final_output_r, final_output_z

if __name__ == '__main__':
    # 初始化模型，指定输入大小、通道数和嵌入维度
    model = local_network(input_size=201, channels=512, embd_dim=192)
    # 生成随机输入数据，模拟输入
    input_tensor = torch.randn(64, 201, 512)
    # 进行前向传播
    output_r, output_z = model(input_tensor)
    # 打印输出结果的形状
    print("Output shape:", output_r.shape)

