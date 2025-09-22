# 导入 PyTorch 库，用于深度学习模型的构建和训练
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的神经网络函数模块
import torch.nn.functional as F

# 定义分类头部类，继承自 nn.Module
class class_head(nn.Module):
    def __init__(self, input_size, channels, embd_dim=192):
        """
        初始化分类头部类。

        参数:
        input_size (int): 输入数据的特征维度。
        channels (int): 卷积层的通道数。
        embd_dim (int): 嵌入向量的维度，默认为 192。
        """
        # 调用父类的构造函数
        super().__init__()
        # 直接在整体模型中定义SimplifiedBackbone的卷积层和BN层
        # 定义一维卷积层，输入通道数为 input_size，输出通道数为 channels，卷积核大小为 5，填充为 2，膨胀率为 1，不使用偏置
        self.backbone_conv = nn.Conv1d(input_size, channels, kernel_size=5, padding=2, dilation=1, bias=False)
        # 定义一维批量归一化层，输入通道数为 channels
        self.backbone_bn = nn.BatchNorm1d(channels)

        # 直接在整体模型中定义Expert的卷积层、池化层和全连接层
        # 定义一维卷积层，输入通道数为 channels，输出通道数为 channels * 2，卷积核大小为 3，填充为 1，不使用偏置
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        # 定义一维批量归一化层，输入通道数为 channels * 2
        self.bn1 = nn.BatchNorm1d(channels * 2)
        # 定义一维卷积层，输入通道数为 channels * 2，输出通道数为 channels * 4，卷积核大小为 3，填充为 1，不使用偏置
        self.conv2 = nn.Conv1d(channels * 2, channels * 4, kernel_size=3, padding=1, bias=False)
        # 定义一维批量归一化层，输入通道数为 channels * 4
        self.bn2 = nn.BatchNorm1d(channels * 4)
        # 定义自适应平均池化层，输出大小为 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 定义全连接层，输入特征数为 channels * 4，输出特征数为 embd_dim
        self.e_fc1 = nn.Linear(channels * 4, embd_dim)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 骨干网络前向传播
        # 通过卷积层
        x = self.backbone_conv(x)
        # 通过批量归一化层
        x = self.backbone_bn(x)
        # 使用 ReLU 激活函数
        x = F.relu(x)

        # 专家层前向传播
        # 通过第一个卷积层
        x = self.conv1(x)
        # 通过第一个批量归一化层
        x = self.bn1(x)
        # 使用 ReLU 激活函数
        x = F.relu(x)
        # 通过第二个卷积层
        x = self.conv2(x)
        # 通过第二个批量归一化层
        x = self.bn2(x)
        # 使用 ReLU 激活函数
        x = F.relu(x)
        # 通过自适应平均池化层并去除最后一个维度
        x = self.pool(x).squeeze(-1)
        # 注释掉的代码，可能是之前的全连接层
        # x = self.fc(x)
        # 通过全连接层
        x = self.e_fc1(x)
        # 使用 ReLU 激活函数
        x = F.relu(x)
        return x

# 定义分类网络类，继承自 nn.Module
class class_network(nn.Module):
    def __init__(self, input_size, channels, embd_dim, num_classes, num_experts=5, k=3):
        """
        初始化分类网络类。

        参数:
        input_size (int): 输入数据的特征维度。
        channels (int): 卷积层的通道数。
        embd_dim (int): 嵌入向量的维度。
        num_classes (int): 分类的类别数。
        num_experts (int): 专家网络的数量，默认为 5。
        k (int): 选择的前 k 个专家网络的数量，默认为 3。
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化分类头部
        self.class_head = class_head(input_size, channels, embd_dim)
        # 专家网络的数量
        self.num_experts = num_experts
        # 选择的前 k 个专家网络的数量
        self.k = k
        # 专家网络
        # 定义一个模块列表，包含 num_experts 个全连接层，输入特征数为 embd_dim，输出特征数为 num_classes
        self.experts = nn.ModuleList([nn.Linear(embd_dim, num_classes) for _ in range(num_experts)])
        # 门控网络
        # 定义一个全连接层，输入特征数为 embd_dim，输出特征数为 num_experts
        self.gate = nn.Linear(embd_dim, num_experts)

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 通过分类头部
        out = self.class_head(x)
        # 计算门控权重
        # 通过门控网络并使用 softmax 函数计算权重
        gate_weights = F.softmax(self.gate(out), dim=-1)
        # Top-K 选择
        # 选择门控权重中前 k 大的索引
        topk_indices = torch.topk(gate_weights, self.k, dim=-1).indices
        # 专家网络输出
        # 计算每个专家网络的输出
        expert_outputs = [expert(out) for expert in self.experts]
        # 将专家网络的输出堆叠在一起
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        # 组合专家输出
        # 初始化最终输出为 0
        final_output = 0
        # 遍历前 k 个专家网络的索引
        for i in range(self.k):
            # 扩展索引以匹配 stacked_outputs 的形状
            index = topk_indices[:, i].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, stacked_outputs.shape[-1])
            # 使用索引切片替代 gather
            # 生成批量索引
            batch_indices = torch.arange(gate_weights.size(0))
            # 获取对应索引的门控权重
            weight = gate_weights[batch_indices, topk_indices[:, i]]
            # 使用 gather 函数获取对应专家网络的输出并去除维度
            output = torch.gather(stacked_outputs, 1, index).squeeze(1)
            # 加权相加
            final_output += weight.unsqueeze(-1) * output
        return final_output

if __name__ == "__main__":
    # 定义模型参数
    # 输入数据的特征维度
    input_size = 200
    # 卷积层的通道数
    channels = 512
    # 嵌入向量的维度
    embd_dim = 192
    # 分类的类别数
    num_classes = 5

    # 创建模型实例
    # 初始化分类网络
    model = class_network(input_size, channels, embd_dim, num_classes)

    # 生成随机输入数据，模拟输入
    # 批量大小
    batch_size = 16
    # 序列长度
    sequence_length = 100
    # 生成随机输入张量
    input_tensor = torch.randn(batch_size, input_size, sequence_length)

    # 进行前向传播
    # 通过模型进行前向传播
    output = model(input_tensor)

    # 打印输出形状，验证模型是否正常工作
    print(f"Output shape: {output.shape}")

    # 模拟计算损失
    # 生成随机目标标签
    target = torch.randint(0, 5, (batch_size,))
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 计算损失
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")