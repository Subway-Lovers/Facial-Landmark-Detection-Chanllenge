import numpy as np
import cv2
import sys
import os
import random
sys.path.append('..')

from torch.utils import data

def random_flip(img, annotation, p = 0.5):
    if random.random() < p:
        return img, annotation

    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    new_annotation = np.array(new_annotation)

    return img, new_annotation


class AFLWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.path = self.line[0]
        self.img_name = self.path.split('/')[-1].split('_')[1]+'.jpg'
        self.img = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:137], dtype=np.float32)
        #self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[137:140], dtype=np.float32)
        if self.transforms:
            # Apply transform to augment the image
            # self.img, self.landmark = random_noise(self.img, self.landmark)
            # self.img, self.landmark = random_brightness(self.img, self.landmark)
            # self.img, self.landmark = random_contrast(self.img, self.landmark)
            # self.img, self.landmark = random_saturation(self.img, self.landmark)
            # self.img, self.landmark = random_hue(self.img, self.landmark)
            self.img, self.landmark = random_flip(self.img, self.landmark)
            self.img = self.transforms(self.img)
        return (self.img_name, self.img, self.landmark, self.euler_angle)

    def __len__(self):
        return len(self.lines)

class AFLW_test_Datasets(data.Dataset):
    def __init__(self, file_root, transforms=None):
        self.transforms = transforms
        self.root = file_root
        self.path = None
        self.lines = os.listdir(file_root)
    def __getitem__(self, index):
        self.img_name = self.lines[index].strip()
        self.path = os.path.join(self.root, self.img_name)
        self.img = cv2.imread(self.path)
        if self.transforms:
            self.img = cv2.resize(self.img, (384, 384))
            self.img = self.transforms(self.img)
            
        return (self.img_name, self.img)

    def __len__(self):
        return len(self.lines)