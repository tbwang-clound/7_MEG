import torch
import torch.nn as nn
# from nw_class import class_network
# from nw_local import local_network
from nw_class_moe import class_network
from nw_local_moe import local_network

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, input_size, channels, embd_dim, num_classes):
        super(MultiTaskLossWrapper, self).__init__()

        # # 加入共享的骨干网络，输入和输出维度一致，结构是CONV+BN+RELU
        # self.backbone_conv = nn.Sequential(
        #     nn.Conv1d(input_size, input_size, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm1d(input_size),
        #     nn.ReLU(inplace=True)
        # )
    
        # self.jymodel = model
        self.classn = class_network(input_size, channels, embd_dim, num_classes)
        self.localn = local_network(input_size, channels, embd_dim)

        # 这里的 log_vars2 可以理解为 c_tau 的对数
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.lossRec = torch.nn.CrossEntropyLoss()


    def forward(self, x, welx, avgx, label, Rr, Sz):
        """
        前向传播函数

        参数:
        x (Tensor): 输入特征
        label (Tensor): 分类标签
        Rr (Tensor): 定位距离的目标值
        Sz (Tensor): 定位深度的目标值

        返回:
        Tensor: 多任务损失
        """
        # x = self.backbone_conv(x)
        outtaskrcgC = self.classn(x)
        lossRec = 0.5 * self.lossRec(outtaskrcgC, label)

        outtaskLocR, outtaskLocD = self.localn(x)
        lossLocR = 0.5 * torch.sum((Rr - outtaskLocR) ** 2) 
        lossLocD = 0.5 * torch.sum((Sz - outtaskLocD) ** 2) 

        # 计算 c_tau
        c_tau = torch.exp(self.log_vars)
        c_tau_squared = c_tau ** 2

        # 计算每个任务的加权损失
        weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
        weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
        weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]

        # 计算正则化项
        reg_loss = torch.log(1 + c_tau_squared).sum()
        # reg_loss = torch.log(1 + c_tau_squared[1:]).sum()

        # 计算多任务损失
        mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
        # mtl_loss = weighted_lossLocR + weighted_lossLocD + reg_loss
        # mtl_loss = weighted_lossRec

        return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared

if __name__ == '__main__':
    inputs = torch.randn(5, 201, 512)
    welchx = torch.randn(5, 201)
    avgx = torch.randn(5, 201)
    model = MultiTaskLossWrapper(input_size=201, channels=512, embd_dim=192, num_classes=5)
    label = torch.randint(0, 5, (5,))
    Rr = torch.rand(5)
    Sz = torch.rand(5)
    l, c, r, d, p = model(inputs, welchx, avgx, label, Rr, Sz)
    print(l, c, r, d, p)
