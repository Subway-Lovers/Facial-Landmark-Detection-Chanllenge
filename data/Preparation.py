#-*- coding: utf-8 -*-
import pickle5 as pickle
import os
import numpy as np
import cv2
import shutil
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from pfld.utils import calculate_pitch_yaw_roll
debug = False

'''def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_'''

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

def random_hue(img, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, annotation

def augmentation(img, annotation):
    img, annotation = random_noise(img, annotation)
    img, annotation = random_brightness(img, annotation)
    img, annotation = random_contrast(img, annotation)
    img, annotation = random_saturation(img, annotation)
    img, annotation = random_hue(img, annotation)
    img, annotation = random_flip(img, annotation)
    img, annotation = channel_shuffle(img, annotation)
    annotation = np.array(annotation)
    return img, annotation


'''def scale(img, annotation):
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

    return image, new_annotation'''

class ImageDate():
    def __init__(self, img, landmark, imgDir, is_train, image_size=384):
        self.image_size = image_size
        assert(len(landmark) == 68)
        self.list = landmark
        self.landmark = np.asarray((landmark), dtype=np.float32).reshape(-1, 2)
        self.is_train = is_train
        self.train_path = os.path.join(imgDir, 'synthetics_train', img)
        
        self.test_path = os.path.join(imgDir, 'aflw_val', img)
        #print(self.test_path, os.path.exists(self.test_path))
        self.img = None
        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def load_data(self, repeat, augment):
        # AFLW has good rotation distribution (-90 to 90 degree),
        # So we don't have to do rotational augmentation here, use other transforms to replace it 
        if self.is_train:
            self.img = cv2.imread(self.train_path)
        else:
            self.img = cv2.imread(self.test_path)
        
        self.imgs.append(self.img)
        self.landmarks.append(self.landmark)

        if self.is_train and augment == True:
            while len(self.imgs) < repeat:
                img_T, landmark_T = augmentation(self.img, self.landmark)
                self.imgs.append(img_T)
                self.landmarks.append(landmark_T)

    def save_data(self, path, prefix):
        labels = []
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (68, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {}\n'.format(save_path, landmark_str, euler_angles_str)

            labels.append(label)
        return labels

def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir,'rb') as f:
        annot = pickle.load(f)
        imgs, landmarks = annot

        '''#for debug
        imgs = imgs[:100]
        landmarks = landmarks[:100]'''
        
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        for i in range(len(imgs)):
            Img = ImageDate(imgs[i], landmarks[i], imgDir, is_train)
            if is_train:
                img_name = Img.train_path
            else:
                img_name = Img.test_path
            # Determine augment or not and the augmentation factor
            Img.load_data(repeat = 5, augment = True)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i)+'_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i+1, len(imgs)))

    with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = os.path.join(root_dir, 'AFLW/data')
    landmarkDirs = ['AFLW/data/aflw_val/annot.pkl',
                    'AFLW/data/synthetics_train/annot.pkl']

    outDirs = ['AFLW_validation_data', 'AFLW_train_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'aflw_val' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)
    print('end')

