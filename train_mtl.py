import argparse
import os
import json
# 导入 datetime 模块中的 datetime 类，用于获取当前时间
from datetime import datetime
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from nw_mtl import MultiTaskLossWrapper
from md_moe_rl import Dataset_audio
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为 Times New Roman，字体大小为 24
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 24})

# 读取 JSON 文件
def read_json_file(file_path):
    """
    读取指定路径的 JSON 文件并返回解析后的数据。
    
    Args:
        file_path (str): JSON 文件的路径。
    
    Returns:
        dict: 解析后的 JSON 数据。
    
    Raises:
        FileNotFoundError: 文件未找到时抛出此异常。
        IOError: 文件读取失败时抛出此异常。
        ValueError: JSON 解析失败时抛出此异常。
        Exception: 其他未知错误时抛出此异常。
    """
    try:
        # 尝试打开文件并读取 JSON 数据
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 解析 JSON 数据
        return data
    except FileNotFoundError:
        # 文件不存在
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except IOError as e:
        # 文件读取错误
        raise IOError(f"文件读取失败: {e}")
    except json.JSONDecodeError as e:
        # JSON 解析错误
        raise ValueError(f"JSON 解析失败: {e}")
    except Exception as e:
        # 其他未知错误
        raise Exception(f"未知错误: {e}")

def plot_confusion_matrix_jy(cm, save_path, class_labels, title='Confusion Matrix', show=True):
    """
    绘制混淆矩阵并保存为图片。
    
    Args:
        cm (np.ndarray): 混淆矩阵。
        save_path (str): 混淆矩阵图片的保存路径。
        class_labels (list): 类别标签列表。
        title (str): 混淆矩阵图的标题，默认为 'Confusion Matrix'。
        show (bool): 是否显示混淆矩阵图，默认为 True。
    """
    plt.figure(figsize=(12, 8), dpi=300)
    size = len(class_labels)
    proportion = []
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(size, size) 
    pshow = np.array(pshow).reshape(size, size)
    #print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)

    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, fontsize=24)
    plt.yticks(tick_marks, class_labels, fontsize=24)
    
    thresh = cm.max() / 2.
    iters = np.reshape([[[i, j] for j in range(size)] for i in range(size)], (cm.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=18, color='white', weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=18, color='white')
        else:
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=18)   # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=18)
    
    plt.ylabel('True label', fontsize=24)
    plt.xlabel('Predict label', fontsize=24)
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    if show:
        # 显示图片
        plt.show()

    plt.close('all')

def get_args():
    """
    解析命令行参数并返回参数对象。
    
    Returns:
        argparse.Namespace: 包含所有解析后参数的对象。
    """
    parser = argparse.ArgumentParser(description='PyTorch classifiaction Training')
    # Datasets
    parser.add_argument('-classes', '--num_classes', default='5', type=int)
    parser.add_argument('-workers', '--num_workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--save_model", type=str, default='/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/models/')
    parser.add_argument("--train_list_path", type=str, default='/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/data/train_list.txt')
    parser.add_argument("--test_list_path", type=str, default='/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/data/test_list.txt')
    parser.add_argument("--label_list_path", type=str, default='/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/data/label_list.txt')
    parser.add_argument('--num_epoch', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test_batch', default=64, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 22],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')   
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set') 
    parser.add_argument('--arch', '-a', metavar='ARCH', default='torchvision_models_se_resnet50')
    # Device options
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    args.log_file = '/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/output/train' + '_' + time_stamp + '.log'

    with open(args.log_file, 'w') as fp:
        fp.write(str(args) + '\n\n')
    return args

# 初始化最佳准确率为 0
best_acc = 0

# 评估模型
@torch.no_grad()
def test(model, test_loader, device, cfmatrix_logdir, class_labels, epoch, Rrmax, Szmax):
    """
    模型评估函数 | Model evaluation function
    
    Args:
        model: 待评估的PyTorch模型 | PyTorch model to evaluate
        test_loader: 测试数据加载器 | Test data loader
        device: 计算设备 (cuda/cpu) | Computing device
        cfmatrix_logdir: 混淆矩阵保存路径 | Confusion matrix save path
        class_labels: 类别标签列表 | List of class labels
        epoch: 当前训练轮次 | Current epoch number
        Rrmax: 距离归一化系数 | Distance normalization factor
        Szmax: 深度归一化系数 | Depth normalization factor
    
    Returns:
        acc_val: 测试准确率 | Test accuracy
        cm: 混淆矩阵 | Confusion matrix
        loss_val: 测试损失 | Test loss
        acc_str: 各类别准确率统计 | Per-class accuracy statistics
        sklearn_accuracy: sklearn计算准确率 | sklearn calculated accuracy
        sklearn_precision: sklearn计算精确率 | sklearn calculated precision
        sklearn_recall: sklearn计算召回率 | sklearn calculated recall
        sklearn_f1: sklearn计算F1分数 | sklearn calculated F1 score
        ABSE_Rs_test: 距离平均绝对误差 | Mean absolute error of distance
        ABSE_Ds_test: 深度平均绝对误差 | Mean absolute error of depth
    """
    model.eval()
    accuracies, preds, labels, ABSE_Rs, ABSE_Ds = [], [], [], [], []
    loss_val_sum = []
    correct_perclass = list(0. for i in range(args.num_classes))
    total_perclass = list(0. for i in range(args.num_classes))
    with torch.no_grad():
        for batch_id, (inputs, inputs_w, inputs_a, label, Rr, Sz) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs_w = inputs_w.to(device)
            inputs_a = inputs_a.to(device)
            label = label.to(device).long()
            Rr = Rr.to(device, dtype=torch.float64)
            Sz = Sz.to(device, dtype=torch.float64)
            Rr = Rr / Rrmax
            Sz = Sz / Szmax
            los_val, output, outtaskLocR, outtaskLocD, prec = model(inputs, inputs_w, inputs_a, label, Rr, Sz)
            # los_val = loss_count(output, outtaskLocR, outtaskLocD, label, Rr, Sz)

            prediction = torch.argmax(output, 1)
            res = prediction == label

            for label_idx in range(len(label)):
                label_single = label[label_idx]
                correct_perclass[label_single] += res[label_idx].item()
                total_perclass[label_single] += 1

            output = torch.nn.functional.softmax(output, dim=1)
            output = output.data.cpu().numpy()
            pred = np.argmax(output, axis=1)
            
            preds.extend(pred.tolist())
            label = label.data.cpu().numpy()
            labels.extend(label.tolist())
            acc = np.mean((pred == label).astype(int))
            accuracies.append(acc.item())
            loss_val_sum.append(los_val)

            # 计算距离定位误差
            outtaskLocR = outtaskLocR.data.cpu().numpy()
            Rr = Rr.data.cpu().numpy()
            ABSE_R = np.mean(np.abs(outtaskLocR - Rr))
            ABSE_Rs.append(ABSE_R)

            # 计算深度定位误差
            outtaskLocD = outtaskLocD.data.cpu().numpy()
            Sz = Sz.data.cpu().numpy()
            ABSE_D = np.mean(np.abs(outtaskLocD - Sz))
            ABSE_Ds.append(ABSE_D)

        sklearn_accuracy = accuracy_score(labels, preds)
        sklearn_precision = precision_score(labels, preds, average='weighted')
        sklearn_recall = recall_score(labels, preds, average='weighted')
        sklearn_f1 = f1_score(labels, preds, average='weighted')

        conf_matrix = confusion_matrix(labels, preds)
        plot_confusion_matrix_jy(cm=conf_matrix, save_path=os.path.join(cfmatrix_logdir, f'混淆矩阵_{epoch}.png'), class_labels=class_labels, show=False)
        print("[Test_sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))

        if epoch == args.num_epoch - 1 or args.evaluate:
            print(classification_report(labels, preds))
            print(conf_matrix)

        acc_str = sum(correct_perclass) / sum(total_perclass)

    acc_val = float(sum(accuracies) / len(accuracies))
    loss_val = float(sum(loss_val_sum) / len(loss_val_sum))
    ABSE_R_val = float(sum(ABSE_Rs) / len(ABSE_Rs))
    ABSE_D_val = float(sum(ABSE_Ds) / len(ABSE_Ds))
    ABSE_Rs_test = float(sum(ABSE_Rs) / len(ABSE_Rs))
    ABSE_Ds_test = float(sum(ABSE_Ds) / len(ABSE_Ds))
    cm = confusion_matrix(labels, preds)

    print('=' * 70)
    print(f'[{datetime.now()}] Test {epoch}, loss: {loss_val:.4f}, accuracy: {acc_val:.4f}, ABSE_R: {ABSE_R_val:.4f}, ABSE_D: {ABSE_D_val:.4f}')
    print('=' * 70)

    return acc_val, cm, loss_val, acc_str, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1, ABSE_Rs_test, ABSE_Ds_test

def train(model, train_loader, device, optimizer, scheduler, epoch, Rrmax, Szmax):
    """
    模型训练函数 | Model training function
    
    Args:
        model: 待训练的PyTorch模型 | PyTorch model to train
        train_loader: 训练数据加载器 | Training data loader
        device: 计算设备 (cuda/cpu) | Computing device
        optimizer: 优化器实例 | Optimizer instance
        scheduler: 学习率调度器 | Learning rate scheduler
        epoch: 当前训练轮次 | Current epoch number
        Rrmax: 距离归一化系数 | Distance normalization factor
        Szmax: 深度归一化系数 | Depth normalization factor
    
    Returns:
        acc_train: 训练准确率 | Training accuracy
        loss_train: 训练损失 | Training loss
        acc_str_train: 各类别准确率统计 | Per-class accuracy statistics
        pres_rc_train: 分类任务权重 | Classification task weight
        pres_lr_train: 距离回归任务权重 | Distance regression task weight
        pres_lz_train: 深度回归任务权重 | Depth regression task weight
        ABSE_Rs_train: 距离平均绝对误差 | Mean absolute error of distance
        ABSE_Ds_train: 深度平均绝对误差 | Mean absolute error of depth
    """
    model.train()
    loss_sum = []
    accuracies = []
    ABSE_Rs = []
    ABSE_Ds = []

    pres_rc_list = []
    pres_lr_list = []
    pres_lz_list = []

    correct_perclass = list(0. for i in range(args.num_classes))
    total_perclass = list(0. for i in range(args.num_classes))
    for batch_id, (inputs, inputs_w, inputs_a, label, Rr, Sz) in enumerate(train_loader):
        # 打印inputs的形状
        # print(f"inputs shape: {inputs.shape}")
        inputs = inputs.to(device)
        inputs_w = inputs_w.to(device)
        inputs_a = inputs_a.to(device)
        label = label.to(device).long()
        Rr = Rr.to(device, dtype=torch.float64)
        Sz = Sz.to(device, dtype=torch.float64)
        Rr = Rr / Rrmax
        Sz = Sz / Szmax
        # print("Rr, Sz: ", Rr, Sz)
        los, output, outtaskLocR, outtaskLocD, prec = model(inputs, inputs_w, inputs_a, label, Rr, Sz)
        # prec从device转移到cpu，并从tensor转为numpy数组
        prec = prec.cpu()
        prec_rc = prec[0]
        prec_lr = prec[1]
        prec_lz = prec[2]
        pres_rc_list.append(prec_rc)
        pres_lr_list.append(prec_lr)
        pres_lz_list.append(prec_lz)

        # 计算损失值
        # los = loss_count(output, outtaskLocR, outtaskLocD, label, Rr, Sz)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()

        prediction = torch.argmax(output, 1)
        res = prediction == label

        for label_idx in range(len(label)):
            label_single = label[label_idx]
            correct_perclass[label_single] += res[label_idx].item()
            total_perclass[label_single] += 1

        # 计算准确率
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc_train = np.mean((output == label).astype(int))
        accuracies.append(acc_train)
        loss_sum.append(los)

        # 计算距离定位误差
        outtaskLocR = outtaskLocR.data.cpu().numpy()
        Rr = Rr.data.cpu().numpy()
        ABSE_R = np.mean(np.abs(outtaskLocR - Rr))
        ABSE_Rs.append(ABSE_R)

        # 计算深度定位误差
        outtaskLocD = outtaskLocD.data.cpu().numpy()
        Sz = Sz.data.cpu().numpy()
        ABSE_D = np.mean(np.abs(outtaskLocD - Sz))
        ABSE_Ds.append(ABSE_D)

    acc_str = sum(correct_perclass) / sum(total_perclass)

    print(f'[{datetime.now()}] Train epoch [{epoch}/{args.num_epoch}], '
          # f'lr: {scheduler.get_last_lr()[0]:.8f}, loss: {sum(loss_sum) / len(loss_sum):.8f}, '
          f'lr: {args.lr:.8f}, loss: {sum(loss_sum) / len(loss_sum):.3f}, '
          f'pres_rc: {sum(pres_rc_list) / len(pres_rc_list):.3f}, '
          f'pres_lr: {sum(pres_lr_list) / len(pres_lr_list):.3f}, '
          f'pres_lz: {sum(pres_lz_list) / len(pres_lz_list):.3f}, '
          f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
          f'ABSE_R: {sum(ABSE_Rs) / len(ABSE_Rs):.5f}, '
          f'ABSE_D: {sum(ABSE_Ds) / len(ABSE_Ds):.5f}')
    pres_rc_train = float(sum(pres_rc_list) / len(pres_rc_list))
    pres_lr_train = float(sum(pres_lr_list) / len(pres_lr_list))
    pres_lz_train = float(sum(pres_lz_list) / len(pres_lz_list))
    ABSE_Rs_train = float(sum(ABSE_Rs) / len(ABSE_Rs))
    ABSE_Ds_train = float(sum(ABSE_Ds) / len(ABSE_Ds))
    acc_train = float(sum(accuracies) / len(accuracies))
    loss_train = float(sum(loss_sum) / len(loss_sum))
    
    return acc_train, loss_train, acc_str, pres_rc_train, pres_lr_train, pres_lz_train, ABSE_Rs_train, ABSE_Ds_train

def run(args):
    """
    运行训练和评估流程。
    
    Args:
        args (argparse.Namespace): 包含所有解析后参数的对象。
    """
    global best_acc

    train_dataset = Dataset_audio(args.train_list_path, sr=16000, chunk_duration=5)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_dataset = Dataset_audio(args.test_list_path, sr=16000, chunk_duration=5)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    folder_path = os.path.dirname(args.train_list_path)
    json_path = os.path.join(folder_path, 'config.json')
    datard = read_json_file(json_path)
    Rrmax = datard['Rrmax']
    Szmax = datard['Szmax']

    print("train nums:{}".format(len(train_dataset)))
    print("test  nums:{}".format(len(test_dataset)))

    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        class_labels = [l.replace('\n', '') for l in lines]  # ['fold1', 'fold10', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9']

    device = torch.device("cuda")

    model = MultiTaskLossWrapper(input_size=200, channels=512, embd_dim=192, num_classes=args.num_classes)
    model.to(device)

    # 获取优化方法
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=5e-4)
    # print('optimizer:', optimizer)
    # print('weight_decay:', args.weight_decay)
    # 获取学习率衰减函数
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    
    confusion_matrix_logdir = os.path.join('log', cur_time + 'epoch{}_bs{}_lr{}'.format(args.num_epoch, args.batch_size, args.lr))

    # 恢复训练
    if args.resume:
        print('resum', args.resume)
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model.pth'), weights_only=True))
        state = torch.load(os.path.join(args.resume, 'model.state'), weights_only=True)
        last_epoch = state['last_epoch']
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'), weights_only=True)
        optimizer.load_state_dict(optimizer_state)
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    if args.evaluate:
        print('\nEvaluation only')
        acc_val, cm, loss_val, acc_str_val, skl_acc_val, skl_precision_val, skl_recall_val, skl_f1_val = test(model, test_loader, device, confusion_matrix_logdir, class_labels, epoch)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (loss_val, acc_val))
        return

    acc_train_list = []
    loss_train_list = []
    acc_str_train_list = []
    pres_rc_train_list = []
    pres_lr_train_list = []
    pres_lz_train_list = []
    ABSE_Rs_train_list = []
    ABSE_Ds_train_list = []

    # 新增：记录测试指标的列表
    acc_val_list = []
    ABSE_Rs_val_list = []
    ABSE_Ds_val_list = []

    # 开始训练
    for epoch in range(args.num_epoch):
        # 训练模型
        acc_train, loss_train, acc_str_train, pres_rc_train, pres_lr_train, pres_lz_train, ABSE_Rs_train, ABSE_Ds_train = train(model, train_loader, device, optimizer, scheduler, epoch, Rrmax, Szmax)
        acc_str_train_list.append(acc_str_train * 100)
        acc_train_list.append(acc_train * 100)
        loss_train_list.append(loss_train)
        pres_rc_train_list.append(pres_rc_train)
        pres_lr_train_list.append(pres_lr_train)
        pres_lz_train_list.append(pres_lz_train)
        ABSE_Rs_train_list.append(ABSE_Rs_train)
        ABSE_Ds_train_list.append(ABSE_Ds_train)

        # 评估模型
        acc_val, cm, loss_val, acc_str_val, skl_acc_val, skl_precision_val, skl_recall_val, skl_f1_val, ABSE_Rs_val, ABSE_Ds_val = test(model, test_loader, device, confusion_matrix_logdir, class_labels, epoch, Rrmax, Szmax)
        # 新增：记录测试指标
        acc_val_list.append(acc_val * 100)
        ABSE_Rs_val_list.append(ABSE_Rs_val)
        ABSE_Ds_val_list.append(ABSE_Ds_val)

        scheduler.step()
        
        # 保存模型
        save_model_path = os.path.join(args.save_model, cur_time + 'epoch{}_bs{}_lr{}'.format(args.num_epoch, args.batch_size, args.lr))
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_model_path, 'model.pth'))
        torch.save({'last_epoch': torch.tensor(epoch)}, os.path.join(save_model_path, 'model.state'))
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer.pth'))

        is_best = acc_str_val > best_acc
        best_acc = max(acc_val, best_acc)

        if is_best:
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(save_model_path, 'model_best.pth'))
            torch.save({'last_epoch': torch.tensor(epoch)}, os.path.join(save_model_path, 'model_best.state'))
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_best.pth'))
        print('===>best_epoch is {},best_acc: {:.4f}.'.format(best_epoch, best_acc))

    # 对训练过程的参数画图
    # 画loss曲线
    plt.figure(figsize=(12, 10), dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(range(len(loss_train_list)), loss_train_list, linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('(a) Train loss')
    
    save_vis_root_path='/home/wangtengbo/A800-7-nfs/A800-13-data/temp_demo/UWTRL-MEG-main/visualization'
    #存储图示
    plt.savefig(os.path.join(save_vis_root_path, 'Train_loss.png'),dpi=300 )
    # 画多任务权重曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(len(pres_rc_train_list)), pres_rc_train_list, label='c-weight', linewidth=2)
    plt.plot(range(len(pres_lr_train_list)), pres_lr_train_list, label='r-weight', linewidth=2)
    plt.plot(range(len(pres_lz_train_list)), pres_lz_train_list, label='d-weight', linewidth=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('weight')
    plt.title('(b) Task weight')
    plt.savefig(os.path.join(save_vis_root_path, 'Task_weight.png'),dpi=300 )
    # 画acc曲线
    plt.subplot(2, 2, 3)
    # plt.plot(range(len(acc_train_list)), acc_train_list, label='train-acc')
    plt.plot(range(len(acc_str_train_list)), acc_str_train_list, label='train-acc', linewidth=2)
    # 新增：画测试正确率曲线
    plt.plot(range(len(acc_val_list)), acc_val_list, label='test-acc', linewidth=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc (%)')
    plt.title('(c) Train and test accuracy')
    plt.savefig(os.path.join(save_vis_root_path, 'Train_test_accuracy.png'),dpi=300 )

    # 画ABSE曲线
    plt.subplot(2, 2, 4)
    plt.plot(range(len(ABSE_Rs_train_list)), ABSE_Rs_train_list, label='r-train', color='#1f77b4', linestyle='--', linewidth=2)
    plt.plot(range(len(ABSE_Ds_train_list)), ABSE_Ds_train_list, label='d-train', color='#ff7f0e', linestyle='--', linewidth=2)
    # 新增：画测试距离绝对误差和深度绝对误差曲线
    plt.plot(range(len(ABSE_Rs_val_list)), ABSE_Rs_val_list, label='r-test', color='#1f77b4', linestyle='-', linewidth=2)
    plt.plot(range(len(ABSE_Ds_val_list)), ABSE_Ds_val_list, label='d-test', color='#ff7f0e', linestyle='-', linewidth=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('normalized ABSE')
    plt.title('(d) Train and test ABSE')
    plt.tight_layout()
    plt.savefig(os.path.join(save_model_path, 'train_and_test_ABSE.png'), dpi=300 )
    plt.savefig(os.path.join(save_vis_root_path, 'Train_test_ABSE.png'),dpi=300 )

    plt.close('all')

    # 把画图参数全存到json中
    train_params = {
        'acc_train_list': acc_train_list,
        'loss_train_list': loss_train_list, 
        'acc_str_train_list': acc_str_train_list,
        'pres_rc_train_list': pres_rc_train_list,
        'pres_lr_train_list': pres_lr_train_list,
        'pres_lz_train_list': pres_lz_train_list,
        'ABSE_Rs_train_list': ABSE_Rs_train_list,
        'ABSE_Ds_train_list': ABSE_Ds_train_list,
        'acc_val_list': acc_val_list,
        'ABSE_Rs_val_list': ABSE_Rs_val_list,
        'ABSE_Ds_val_list': ABSE_Ds_val_list
    }
    with open(os.path.join(save_model_path, 'train_params.json'), 'w') as f:
        json.dump(train_params, f)

if __name__ == '__main__':
    args = get_args()
    # print_arguments(args)
    run(args)