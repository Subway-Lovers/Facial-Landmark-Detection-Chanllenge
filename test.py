import argparse
import time
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.AFLW_dataset import AFLW_test_Datasets

from models.pfld import PFLDInference
from models.mobileV3 import mobilenetv3_small
cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)'''

def test(test_dataloader, pfld_backbone, outdir):
    pfld_backbone.eval()
    with torch.no_grad():
        if os.path.exists(os.path.join(outdir, 'solution.txt')):
            os.remove(os.path.join(outdir, 'solution.txt'))
        with open(os.path.join(outdir, 'solution.txt'),'w') as f:
            for img_name, img in test_dataloader:
                img = img.to(device)
                pfld_backbone = pfld_backbone.to(device)
                _, landmark = pfld_backbone(img)
                landmark = landmark.cpu().numpy()
                assert (landmark.shape[1] == 136)
                for i in range(len(img_name)):
                    landmark_str = ' '.join(list(map(str,landmark[i].reshape(-1).tolist())))
                    f.writelines('{} {}\n'.format(img_name[i], landmark_str))
            

def main(args):
    
    checkpoint = torch.load(args.model_path, map_location=device)
    # pfld_backbone = PFLDInference().to(device)
    pfld_backbone = mobilenetv3_small(136)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = AFLW_test_Datasets(args.test_dataroot, transforms = transform)
    
    test_dataloader = DataLoader(test_dataset,
                                     batch_size=25,
                                     shuffle=False,
                                     num_workers=0)
    test(test_dataloader, pfld_backbone, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint_epoch_11.pth.tar",
                        type=str)
    parser.add_argument('--test_dataroot',
                        default='./data/AFLW/data/aflw_test',
                        type=str)
    parser.add_argument('--out_dir',
                        default='./output',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    '''with open('./output/ans.txt','r') as a:
        lines = a.readlines()
        for line in lines:
            print(os.path.exists(os.path.join(args.test_dataroot, line.split()[0])))'''
    main(args)