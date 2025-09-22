import torch
import torch.nn as nn
import torch.nn.functional as F

class class_head(nn.Module):
    def __init__(self, input_size, channels, embd_dim=192):
        super().__init__()
        # 直接在整体模型中定义SimplifiedBackbone的卷积层和BN层
        self.backbone_conv = nn.Conv1d(input_size, channels, kernel_size=5, padding=2, dilation=1, bias=False)
        self.backbone_bn = nn.BatchNorm1d(channels)

        # 直接在整体模型中定义Expert的卷积层、池化层和全连接层
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels * 2)
        self.conv2 = nn.Conv1d(channels * 2, channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels * 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.e_fc1 = nn.Linear(channels * 4, embd_dim)

    def forward(self, x):
        # 骨干网络前向传播
        x = self.backbone_conv(x)
        x = self.backbone_bn(x)
        x = F.relu(x)

        # 专家层前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)
        # x = self.fc(x)
        x = self.e_fc1(x)
        x = F.relu(x)
        return x
    
class class_network(nn.Module):
    """
    端到端分类网络
    Args:
        input_size: 输入特征维度 (默认201个频点)
        channels: 初始通道数 (默认512)
        embd_dim: 特征嵌入维度 (默认192)
        num_classes: 分类类别数
    Forward:
        输入: (batch, freq_bins, time_steps)
        输出: (batch, num_classes)
    """
    def __init__(self, input_size, channels, embd_dim, num_classes):
        super().__init__()
        self.class_head = class_head(input_size, channels, embd_dim)
        self.fc2 = nn.Linear(embd_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.class_head(x))

if __name__ == "__main__":
    # 定义模型参数
    input_size = 200
    channels = 512
    num_classes = 5

    # 创建模型实例
    model = class_network(input_size, channels, num_classes)

    # 生成随机输入数据，模拟输入
    batch_size = 16
    sequence_length = 100
    input_tensor = torch.randn(batch_size, input_size, sequence_length)

    # 进行前向传播
    output = model(input_tensor)

    # 打印输出形状，验证模型是否正常工作
    print(f"Output shape: {output.shape}")

    # 模拟计算损失
    target = torch.randint(0, 5, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")