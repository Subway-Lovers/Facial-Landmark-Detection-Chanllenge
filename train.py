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
# from models.mobilenetV3 import mobilenetv3_large, mobilenetv3_small
from models.mobilenetV3 import mobilenetv3
#from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataset.AFLW_dataset import AFLWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

# change the cuda number if needed
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#torch.cuda.empty_cache()

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, model, auxiliarynet, criterion, optimizer,
          epoch, mute = False):
    losses = AverageMeter()

    # # -- mobilenetV3 ---
    # mobilenet = mobilenetv3(outnum = 136, mode = 'small')
    # print(mobilenet)
    # # ------------------

    weighted_loss, loss = None, None
    with tqdm(total = len(train_loader), disable = mute) as pbar:
        for _, img, landmark_gt, euler_angle_gt in train_loader:
            #print(img.shape, landmark_gt.shape, euler_angle_gt.shape)
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            
            model = model.to(device)
            #features, landmarks = pfld_backbone(img)
            # _, landmarks = model(img) # pfld_backbone
            landmarks = model(img) # our own model
            
            auxiliarynet = auxiliarynet.to(device)
            #angle = auxiliarynet(features)
            angle = 0
            weighted_loss, loss = criterion(landmark_gt,
                                            euler_angle_gt, angle, landmarks)
            optimizer.zero_grad()
            loss.backward()
            #weighted_loss.backward()
            optimizer.step()
            #print(pfld_backbone.conv1.weight)
            losses.update(loss.item())
            pbar.update(1)
    return weighted_loss, loss


def validate(wlfw_val_dataloader, model, auxiliarynet, criterion):
    model.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for _, img, landmark_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            model = model.to(device)
            auxiliarynet = auxiliarynet.to(device)
            # _, landmark = model(img) # pfld_backbone
            landmark = model(img) # our own model
            loss = torch.mean(torch.sqrt(torch.sum((landmark_gt - landmark)**2, axis=1)))/384
            losses.append(loss.cpu().numpy())

    return np.mean(losses)


def main(args):
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
    model = mobilenetv3(outnum = 136, mode = 'small').to(device)
    # model = PFLDInference().to(device)
    print(model)

    auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{
        'params': model.parameters()
    }, {
        'params': auxiliarynet.parameters()
    }],
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor = 0.5, patience=args.lr_patience, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        model.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    # wlfwdataset = WLFWDatasets(args.dataroot, transform)
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
    
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, model,
                                                auxiliarynet, criterion,
                                                optimizer, epoch)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
            }, filename)

        val_loss = validate(val_dataloader, model, auxiliarynet,
                            criterion)

        scheduler.step(val_loss)
        print('Epoch - %d,\t Weighted Train Loss: %4f,\t Train Loss: %.4f,\t Validation Loss: %.2f' % (epoch, weighted_train_loss, train_loss, val_loss))
        
        '''writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {
            'val loss': val_loss,
            'train loss': train_loss
        }, epoch)
    writer.close()'''



def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
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
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=128, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
