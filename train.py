#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn as nn
import random
#from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataset.AFLW_dataset import AFLWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from models.mobileV3 import mobilenetv3_small, mobilenetv3_large
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.cuda.empty_cache()
def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, pfld_backbone, auxiliarynet, criterion, optimizer,
          epoch, mute = False):
    losses = AverageMeter()
    weighted_loss, loss = None, None
    with tqdm(total = len(train_loader), disable = mute) as pbar:
        for _, img, landmark_gt, euler_angle_gt in train_loader:
            #print(img.shape, landmark_gt.shape, euler_angle_gt.shape)
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            features, landmarks = pfld_backbone(img)
            # landmarks = pfld_backbone(img)
            
            auxiliarynet = auxiliarynet.to(device)
            
            angle = auxiliarynet(features)
            # angle = 0
            weighted_loss, loss = criterion(landmark_gt,
                                            euler_angle_gt, angle, landmarks)
            optimizer.zero_grad()
            weighted_loss.backward()
            # loss.backward()
            optimizer.step()
            #print(pfld_backbone.conv1.weight)
            losses.update(loss.item())
            pbar.update(1)
    return weighted_loss, loss


def validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet, criterion):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for _, img, landmark_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = pfld_backbone(img)
            # landmark = pfld_backbone(img)
            loss = torch.mean(torch.sqrt(torch.sum((landmark_gt - landmark)**2, axis=1)))/384
            losses.append(loss.cpu().numpy())

    return np.mean(losses)


def main(args):
    fixed_seed(args.seed)
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    pfld_backbone = mobilenetv3_large(136).to(device)
    # pfld_backbone = PFLDInference().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{
        'params': pfld_backbone.parameters()
    }, {
        'params': auxiliarynet.parameters()
    }],
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5 , patience=args.lr_patience, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    #wlfwdataset = WLFWDatasets(args.dataroot, transform)
    aflwdataset = AFLWDatasets(args.dataroot, transform)
    dataloader = DataLoader(aflwdataset,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    '''wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=args.val_batchsize,
                                     shuffle=False,
                                     num_workers=args.workers)'''
    aflw_val_dataset = AFLWDatasets(args.val_dataroot, transform)
    val_dataloader = DataLoader(aflw_val_dataset,
                                     batch_size=args.val_batchsize,
                                     shuffle=False,
                                     num_workers=args.workers)

    # step 4: run
    #writer = SummaryWriter(args.tensorboard)
    best_loss = 10
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone,
                                                auxiliarynet, criterion,
                                                optimizer, epoch)
        val_loss = validate(val_dataloader, pfld_backbone, auxiliarynet,
                            criterion)
        logging.info('Epoch - %d,\t Weighted Train Loss: %4f,\t Train Loss: %.4f,\t Validation Loss: %.4f' % (epoch, weighted_train_loss, train_loss, val_loss))
        if val_loss < best_loss:
            logging.info('get best model!')
            best_loss = val_loss
            filename = os.path.join(str(args.best_model), "best" + '.pth.tar')
            save_checkpoint(
            {
                'pfld_backbone': pfld_backbone.state_dict(),
            }, filename)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint(
            {
                'pfld_backbone': pfld_backbone.state_dict(),
            }, filename)
        
        scheduler.step(val_loss)
        # print('Epoch - %d,\t Weighted Train Loss: %4f,\t Train Loss: %.4f,\t Validation Loss: %.4f' % (epoch, weighted_train_loss, train_loss, val_loss))
       


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.001, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=5, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--best_model',
                        default='./checkpoint/best_model/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='./data/AFLW_train_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='./data/AFLW_validation_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--test_dataroot',
                        default='./data/AFLW_test_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--train_batchsize', default=60, type=int)
    parser.add_argument('--val_batchsize', default=60, type=int)
    parser.add_argument('--seed', default=7414, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
