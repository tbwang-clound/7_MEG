import os
import json
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman，字号为 24
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

# 定义训练参数文件的路径
datapath = r'E:\2.0_UWTR\models\2025-03-21-15-25-25epoch150_bs64_lr0.001\train_params.json'

# 打开并加载训练参数文件中的数据
with open(datapath, 'r') as f:
    data = json.load(f)

# 获取保存模型的路径
save_model_path = os.path.dirname(datapath)

# 从加载的数据中提取训练相关的列表
loss_train_list = data['loss_train_list']
pres_rc_train_list = data['pres_rc_train_list']
pres_lr_train_list = data['pres_lr_train_list']
pres_lz_train_list = data['pres_lz_train_list']
acc_train_list = data['acc_train_list']
acc_str_train_list = data['acc_str_train_list']
acc_val_list = data['acc_val_list']
ABSE_Rs_train_list = data['ABSE_Rs_train_list']
ABSE_Ds_train_list = data['ABSE_Ds_train_list']
ABSE_Rs_val_list = data['ABSE_Rs_val_list']
ABSE_Ds_val_list = data['ABSE_Ds_val_list']

# 创建一个新的图形，设置图形大小和分辨率
plt.figure(figsize=(6, 5), dpi=300)
# 绘制训练损失曲线
plt.plot(range(len(loss_train_list)), loss_train_list, linewidth=2)
# 设置 x 轴标签为 epoch
plt.xlabel('epoch')
# 设置 y 轴标签为 loss
plt.ylabel('loss')
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 取消显示训练损失图的标题
# plt.title('(a) Train loss')
# 将训练损失图保存为 PDF 格式
plt.savefig(os.path.join(save_model_path,'train_loss.pdf'), dpi=300 )

# 开始绘制多任务权重曲线
plt.figure(figsize=(6, 5), dpi=300)
# 绘制 c 权重曲线
plt.plot(range(len(pres_rc_train_list)), pres_rc_train_list, label='c-weight', linewidth=2)
# 绘制 r 权重曲线
plt.plot(range(len(pres_lr_train_list)), pres_lr_train_list, label='r-weight', linewidth=2)
# 绘制 d 权重曲线
plt.plot(range(len(pres_lz_train_list)), pres_lz_train_list, label='d-weight', linewidth=2)
# 显示图例
plt.legend()
# 设置 x 轴标签为 epoch
plt.xlabel('epoch')
# 设置 y 轴标签为 weight
plt.ylabel('weight')
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 取消显示多任务权重图的标题
# plt.title('(b) Task weight')
# 将多任务权重图保存为 PDF 格式
plt.savefig(os.path.join(save_model_path,'train_weight.pdf'), dpi=300 )

# 开始绘制准确率曲线
plt.figure(figsize=(6, 5), dpi=300)
# 绘制训练准确率曲线
plt.plot(range(len(acc_str_train_list)), acc_str_train_list, label='train-acc', linewidth=2)
# 绘制测试准确率曲线
plt.plot(range(len(acc_val_list)), acc_val_list, label='test-acc', linewidth=2)
# 显示图例
plt.legend()
# 设置 x 轴标签为 epoch
plt.xlabel('epoch')
# 设置 y 轴标签为 acc (%)
plt.ylabel('acc (%)')
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 取消显示准确率图的标题
# plt.title('(c) Train and test accuracy')
# 将准确率图保存为 PDF 格式
plt.savefig(os.path.join(save_model_path,'acc.pdf'), dpi=300 )

# 开始绘制 ABSE 曲线
plt.figure(figsize=(6, 5), dpi=300)
# 绘制训练距离绝对误差曲线
plt.plot(range(len(ABSE_Rs_train_list)), ABSE_Rs_train_list, label='r-train', color='#1f77b4', linestyle='--', linewidth=2)
# 绘制训练深度绝对误差曲线
plt.plot(range(len(ABSE_Ds_train_list)), ABSE_Ds_train_list, label='d-train', color='#ff7f0e', linestyle='--', linewidth=2)
# 绘制测试距离绝对误差曲线
plt.plot(range(len(ABSE_Rs_val_list)), ABSE_Rs_val_list, label='r-test', color='#1f77b4', linestyle='-', linewidth=2)
# 绘制测试深度绝对误差曲线
plt.plot(range(len(ABSE_Ds_val_list)), ABSE_Ds_val_list, label='d-test', color='#ff7f0e', linestyle='-', linewidth=2)
# 显示图例
plt.legend()
# 设置 x 轴标签为 epoch
plt.xlabel('epoch')
# 设置 y 轴标签为 normalized MAE
plt.ylabel('normalized MAE')
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 取消显示 ABSE 图的标题
# plt.title('(d) Train and test ABSE')
# 取消保存 ABSE 图为 PNG 格式
# plt.savefig(os.path.join(save_model_path,'train_and_test_ABSE.png'), dpi=300 )
# 将 ABSE 图保存为 PDF 格式
plt.savefig(os.path.join(save_model_path,'train_and_test_ABSE.pdf'), dpi=300 )
# 关闭所有打开的图形
plt.close('all')