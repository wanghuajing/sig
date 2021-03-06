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
from models import myloss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=./runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(args.save_dir)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    size = [1024, 512]
    # GE1 = [0.218, 0.218, 0.218]
    # GE2 = [0.164, 0.164, 0.164]
    # HLG1 = [0.276, 0.276, 0.276]
    # HLG2 = [0.176, 0.176, 0.176]

    dir = args.data_path
    train_df = pd.read_csv(dir + 'GE/{}_train.csv'.format(args.type))
    val_df = pd.read_csv(dir + 'GE/{}_val.csv'.format(args.type))
    # 实例化训练数据集
    train_data_set = GE_dataset(pathImageDirectory=dir,
                                df=train_df,
                                data='train')

    # 实例化验证数据集
    val_data_set = GE_dataset(pathImageDirectory=dir,
                              df=val_df,
                              data='val')

    batch_size = args.batch_size
    nw = args.num_worker  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # 选择模型
    model = enhance_net_nopool()
    # model = DenseNet121(args.num_classes)
    # model = ResNet50(args.num_classes)
    # model = DenseNet121_z(args.num_classes)
    # model = sig_add()
    # model = Sigmoid(3, 2, 12)
    # model = single_sig()
    # model = test()
    device_ids = [0, 1]
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    # 导入部分预训练权重
    # if args.checkpoint is not None:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.checkpoint)
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(OrderedDict(model_dict))
    # 导入全部预训练权重
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    # 是否冻结权重
    # for name, para in model.named_parameters():
    #     if 'den121' in name:
    #         para.requires_grad = False
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, mode='max')
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    loss_function = torch.nn.BCELoss()


    # pos_weight = torch.ones([1]) * 3  # finetune
    # loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight.cuda())  # finetune
    best_auc = 0
    results = pd.DataFrame(
        columns=["loss/train", "loss/val", "metric/Acc", "metric/Precision", "metric/Recall", "metric/AUC",
                 "thresholds"])
    for epoch in range(args.epochs):
        # train
        sum_loss = train_one_epoch(model=model,
                                   loss=loss_function,
                                   optimizer=optimizer,
                                   data_loader=train_loader,
                                   device=device,
                                   epoch=epoch)

        train_loss = sum_loss / len(train_loader)
        # validate
        sum_loss, acc, precision, recall, fpr, tpr, auc, idx, thresholds = evaluate(model=model,
                                                                                    loss=loss_function,
                                                                                    data_loader=val_loader,
                                                                                    device=device)
        scheduler.step(auc)
        val_loss = sum_loss / len(val_loader)
        if best_auc < auc:
            torch.save(model.state_dict(), args.save_dir + 'best.pth')
            best_auc = auc
            plt.figure(clear=True)
            plt.plot(fpr, tpr, linewidth=2, label="ROC")
            plt.xlabel("false positive rate")
            plt.ylabel("true positive rate")
            plt.ylim(0, 1.05)
            plt.xlim(0, 1.05)
            plt.legend(loc=4)  # 图例的位置
            plt.title('Epoch:{}--AUC:{}'.format(epoch, round(auc, 2)))
            plt.plot(fpr[idx], tpr[idx], marker="^", c='r')
            plt.annotate("({:.2f},{:.2f})".format(fpr[idx], tpr[idx]), xy=[fpr[idx], tpr[idx]], xytext=(20, -10),
                         textcoords='offset points')
            plt.annotate("Thresholds:{:.2f}".format(thresholds[idx]), xy=[fpr[idx], tpr[idx]], xytext=(20, -20),
                         textcoords='offset points')
            plt.savefig(args.save_dir + 'AUC.png')
        torch.save(model.state_dict(), args.save_dir + 'e_{}.pth'.format(epoch))
        print("[epoch {}] AUC:{}".format(epoch, auc))
        tags = ["loss/train", "loss/val", "metric/Acc", "metric/Precision", "metric/Recall", "metric/AUC",
                "metric/thresholds",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], acc, epoch)
        tb_writer.add_scalar(tags[3], precision, epoch)
        tb_writer.add_scalar(tags[4], recall, epoch)
        tb_writer.add_scalar(tags[5], auc, epoch)
        tb_writer.add_scalar(tags[6], thresholds[idx], epoch)
        tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)
        result = pd.DataFrame(
            {"loss/train": train_loss, "loss/val": val_loss, "metric/Acc": acc, "metric/Precision": precision,
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
    parser.add_argument('--type', type=str, default='./runs')
    parser.add_argument('--save_dir', type=str, default='GE')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--num_worker', type=int, default=8)

    # 数据集所在根目录
    parser.add_argument('--data_path', type=str, default="/data")

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--weights', type=str, default='densenet121.pth',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
