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
from tqdm import tqdm

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
    # model = DenseNet121(alossrgs.num_classes)
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
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # loss_function = torch.nn.BCELoss()

    # zero_dence loss
    L_color = myloss.L_color()
    L_spa = myloss.L_spa()
    L_exp = myloss.L_exp(16, 0.6)
    L_TV = myloss.L_TV()

    for epoch in range(args.epochs):
        train_loader = tqdm(train_loader)
        for iteration, data in enumerate(train_loader):
            img_lowlight, _ = data
            img_lowlight = img_lowlight.to(device)

            enhanced_image_1, enhanced_image, A = model(img_lowlight)

            Loss_TV = 200 * L_TV(A)

            loss_spa =  torch.mean(L_spa(enhanced_image, img_lowlight))

            # loss_col = 5 * torch.mean(L_color(enhanced_image))

            loss_exp = 10*torch.mean(L_exp(enhanced_image))

            # best_loss
            loss = Loss_TV + loss_spa + loss_exp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if ((iteration + 1) % 20) == 0:
            #     print('loss:{}, | TV:{} | spa:{} | exp:{}'.format(loss.item(), Loss_TV.item(), loss_spa.item(),
            #                                                       loss_exp.item()))
        torch.save(model.state_dict(), args.save_dir + 'e_{}.pth'.format(epoch))


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
