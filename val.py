import os
import math
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
from my_dataset import *
from utils import read_split_data, train_one_epoch, evaluate
from models.sig_module import *
from models import *
import matplotlib.pyplot as plt
from collections import OrderedDict


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir,exist_ok=True)
    size = [1024, 512]
    data_transform = {
        "val": transforms.Compose([transforms.Resize(size),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                   ])}
    dir = args.data_path
    val_df = pd.read_csv(args.csv)
    # 实例化验证数据集
    val_data_set = GE_dataset(pathImageDirectory=dir,
                              df=val_df,
                              transform=data_transform["val"])

    batch_size = args.batch_size
    nw = args.num_worker  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # 如果存在预训练权重则载入
    model = DenseNet121(args.num_classes)
    device_ids = [0, 1]
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
    # 导入预训练权重
    # if args.checkpoint is not None:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.checkpoint)
    #     pretrained_dict = {'den121.' + k: v for k, v in pretrained_dict.items() if 'den121.' + k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(OrderedDict(model_dict))
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    loss_function = torch.nn.BCELoss()
    # pos_weight = torch.ones([1]) * 3  # finetune
    # loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight.cuda())  # finetune
    results = pd.DataFrame(
        columns=["loss/train", "loss/val", "metric/Acc", "metric/Precision", "metric/Recall", "metric/AUC","thresholds"])
    # validate
    sum_loss, acc, precision, recall, fpr, tpr, auc, idx, thresholds = evaluate(model=model,
                                                                                loss=loss_function,
                                                                                data_loader=val_loader,
                                                                                device=device)
    val_loss = sum_loss / len(val_loader)
    plt.figure(clear=True)
    plt.plot(fpr, tpr, linewidth=2, label="ROC")
    plt.title('AUC:{}'.format(round(auc, 3)))
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc=4)  # 图例的位置
    plt.plot(fpr[idx], tpr[idx], marker="^", c='r')
    plt.annotate("({:.2f},{:.2f})".format(fpr[idx], tpr[idx]), xy=[fpr[idx], tpr[idx]], xytext=(20, -10),
                 textcoords='offset points')
    plt.annotate("Thresholds:{:.2f}".format(thresholds[idx]), xy=[fpr[idx], tpr[idx]], xytext=(20, -20),
                 textcoords='offset points')
    plt.savefig(args.save_dir + 'AUC.png')

    result = pd.DataFrame(
        {"loss/val": val_loss, "metric/Acc": acc, "metric/Precision": precision,
         "metric/Recall": recall,
         "metric/AUC": auc,
         "thresholds": thresholds[idx],
         }, index=[1])
    results = results.append(result, ignore_index=True)
    results.to_csv(args.save_dir + 'results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--csv', type=str, default='./val')
    parser.add_argument('--save_dir', type=str, default='GE')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--data_path', type=str, default="/data")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)


