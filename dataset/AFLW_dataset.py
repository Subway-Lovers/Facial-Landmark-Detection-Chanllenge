import numpy as np
import cv2
import sys
import os
import random
sys.path.append('..')

from torch.utils import data


def flip(img, annotation):
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

    return img, new_annotation


def channel_shuffle(img, annotation):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    coef = np.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = cv2.resize(img, (h, w),
                   preserve_range=True,
                   anti_aliasing=True,
                   mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


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