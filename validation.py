# ------------------------------------------------------------------------------
# Copyright (c) Zhichao Zhao
# Licensed under the MIT License.
# Created by Zhichao zhao(zhaozhichao4515@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets
from dataset.AFLW_dataset import AFLWDatasets

from models.pfld import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros((N, 68))
    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        assert (L == 68)
        rmse[i]  = np.mean(np.sqrt(np.sum(np.power((pts_pred - pts_gt), 2), 1)))/384
    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def validate(val_dataset, pfld_backbone):
    pfld_backbone.eval()
    count = 0
    nme_list = []
    cost_time = []
    val_dataloader = DataLoader(val_dataset,
                                     batch_size=128,
                                     shuffle=False,
                                     num_workers=0)
    #print(len(val_dataloader))
    with torch.no_grad():
        for img_name, img, landmark_gt, _ in val_dataloader:
            print(img.shape)
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)

            start_time = time.time()
            _, landmarks = pfld_backbone(img)
            cost_time.append(time.time() - start_time)
            
            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                          2)  # landmark
            img_name = np.array(img_name)
            #print(img_name.shape, landmarks.shape)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                              2).cpu().numpy()  # landmark_gt
            nme_temp = compute_nme(landmarks, landmark_gt)
            for item in nme_temp:

                nme_list.append(item)

            ''' if args.show_image:
                show_img = np.array(
                    np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] * [112, 112]

                cv2.imwrite("show_img.jpg", show_img)
                img_clone = cv2.imread("show_img.jpg")

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255, 0, 0), -1)
                cv2.imshow("show_img.jpg", img_clone)
                cv2.waitKey(0)'''

            
        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
            failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])

    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = AFLWDatasets(args.test_dataset, transform)
    validate(val_dataset, pfld_backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Validating')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint_epoch_8.pth.tar",
                        type=str)
    parser.add_argument('--test_dataset',
                        default='./data/AFLW_validation_data/list.txt',
                        type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
