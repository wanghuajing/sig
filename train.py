import os
import math
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import *
from my_dataset import *
from utils import read_split_data, train_one_epoch, evaluate
from models.sig_module import *
from models import *


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.Resize([1280, 1280]),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([transforms.Resize([1280, 1280]),
                                   transforms.ToTensor(),
                                   ])}
    dir = args.data_path
    train_df = pd.read_csv(dir + 'GE_train.csv')
    val_df = pd.read_csv(dir + 'GE_val.csv')
    # 实例化训练数据集
    train_data_set = GE_dataset(pathImageDirectory=dir,
                                df=train_df,
                                transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = GE_dataset(pathImageDirectory=dir,
                              df=val_df,
                              transform=data_transform["val"])

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

    # 如果存在预训练权重则载入
    model = res50(3, 2)
    model.to(device)

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    loss=loss_function,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        sum_loss, acc, precision, recall = evaluate(model=model,
                                                    loss=loss_function,
                                                    data_loader=val_loader,
                                                    device=device)
        val_loss = sum_loss / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, acc))
        tags = ["loss/train", "loss/val", "metric/Acc", "metric/Precision", "metric/Recall", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], acc, epoch)
        tb_writer.add_scalar(tags[3], precision, epoch)
        tb_writer.add_scalar(tags[4], recall, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
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
