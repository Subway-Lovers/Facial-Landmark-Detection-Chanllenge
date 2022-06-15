#-*- coding: utf-8 -*-
import pickle5 as pickle
import os
import numpy as np
import cv2
import shutil
import sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from pfld.utils import calculate_pitch_yaw_roll
debug = False

def rotate(angle, center, landmark):
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
    return M, landmark_

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

    def load_data(self, repeat):
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        if self.is_train:
            self.img = cv2.imread(self.train_path)
        else:
            self.img = cv2.imread(self.test_path)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2

        #print(xy, boxsize)
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = self.img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        '''imgT = self.img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)'''
        self.imgs.append(self.img)
        self.landmarks.append(self.landmark)

        '''if self.is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(self.img, M, (int(self.img.shape[1]*1.1), int(self.img.shape[0]*1.1)))

                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                self.imgs.append(imgT)
                self.landmarks.append(landmark)'''

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
            Img.load_data(10)
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

