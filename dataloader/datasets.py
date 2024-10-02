import os
import os.path

import random

import PIL.Image
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from dataloader.data_io import get_transform, get_transform_toy
from dataloader.readpfm import readPFM as pfm_imread
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image

def cv2_disparity_loader(path):
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return disp.astype(np.float32) / 32

def disparity_loader(path):
    if '.png' in path:
        return Image.open(path)
    else:
        return pfm_imread(path)

# SceneFlow_disparity
def disparity_loader_SceneFlow(path):
    return pfm_imread(path)

# eth3d_disparity
def disparity_loader_eth3d(path):
    return pfm_imread(path)

# KITTI_disparity
def disparity_loader_KITTI(path):
    return Image.open(path)

def dataloader_eth3d(filepath):
  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath, img) for img in img_list]
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]

  return left_train, right_train, disp_train_L


def dataloader_middlebury(filepath):
  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath, img) for img in img_list]
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]

  return left_train, right_train, disp_train_L

def dataloader_middleburyadditional(filepath):
  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath, img) for img in img_list]
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0.pfm' % (filepath,img) for img in img_list]

  return left_train, right_train, disp_train_L



def dataloader_KITTI(filepath):
    left_fold = 'colored_0/'
    right_fold = 'colored_1/'
    disp_noc = 'pseudo_kit12/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    train = image[:]
    val = image[:]

    left_train = [filepath + left_fold + img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    disp_train = [filepath + disp_noc + img for img in train]

    left_val = [filepath + left_fold + img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    disp_val = [filepath + disp_noc + img for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val


def dataloader_KITTI2015(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'pseudo_kit15/'
    disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    train = image[:]
    val = image[:]

    left_train = [filepath + left_fold + img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    disp_train_L = [filepath + disp_L + img for img in train]
    disp_train_R = [filepath + disp_R + img for img in train]

    left_val = [filepath + left_fold + img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    disp_val_L = [filepath + disp_L + img for img in val]
    disp_val_R = [filepath + disp_R + img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    # return left_train, right_train, disp_train_L


def dataloader_SceneFlow(filepath):
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    monkaa_path = filepath + 'Monkaa/' + 'frames_finalpass/' 
    monkaa_disp = filepath + 'Monkaa/' + 'disparity/'

    monkaa_dir = os.listdir(monkaa_path)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
            if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                all_left_img.append(monkaa_path + '/' + dd + '/left/' + im)
                all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')

        for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
            if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                all_right_img.append(monkaa_path + '/' + dd + '/right/' + im)

    flying_path = filepath +  'FlyingThings3D/' + 'frames_finalpass/' 
    flying_disp = filepath +  'FlyingThings3D/' + 'disparity/' 
    flying_dir = flying_path + '/TRAIN/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    flying_dir = flying_path + '/TEST/'

    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    driving_dir = filepath + 'Driving/' + 'frames_finalpass/' 
    driving_disp = filepath + 'Driving/' + 'disparity/' 

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                    all_left_disp.append(
                        driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


def dataloader_toy_example(filepath):
    image = [img for img in os.listdir(filepath)]
    train = image[:]
    val = image[:]

    img_train = [filepath + img for img in train]
    img_val = [filepath + img for img in val]

    return img_train, img_val,



class myImageFloder_SceneFlow(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader,
                 dploader=disparity_loader_SceneFlow):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 320, 640

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

        else:
            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

class myImageFloder_KITTI(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader_KITTI):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)

        disp_L = self.disp_L[index]
        dataL = self.dploader(disp_L)
        if self.training:
            w, h = left_img.size
            th, tw = 320, 832

            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            right_img.flags.writeable = True
            if np.random.binomial(1, 0.25):
                sx = int(np.random.uniform(35, 100))
                sy = int(np.random.uniform(25, 75))
                cx = int(np.random.uniform(sx, right_img.shape[0] - sx))
                cy = int(np.random.uniform(sy, right_img.shape[1] - sy))
                right_img[cx - sx:cx + sx, cy - sy:cy + sy] = np.mean(np.mean(right_img, 0), 0)[np.newaxis, np.newaxis]

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

        else:
            w, h = left_img.size
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


class myImageFloder_eth3d(data.Dataset):

    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity

        self.training = training
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)

        disp_L = self.disp_L[index]
        flag = False
        if '.png' in disp_L:
            dataL = self.dploader(disp_L)
            flag = True
        else:
            dataL, _ = self.dploader(disp_L)


        if self.training:
            w, h = left_img.size
            th, tw = 256, 512
            #th, tw = 320, 704

            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            right_img.flags.writeable = True

            if np.random.binomial(1, 0.2):
              sx = int(np.random.uniform(35, 100))
              sy = int(np.random.uniform(25, 75))
              cx = int(np.random.uniform(sx, right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy, right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx, cy-sy:cy+sy] = np.mean(np.mean(right_img, 0), 0)[np.newaxis,np.newaxis]

            if flag == True:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            else:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

        else:
            w, h = left_img.size
            # normalize
            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            # top_pad = 768 - h
            # right_pad = 1024 - w
            # assert top_pad >= 0 and right_pad >= 0
            #
            # # pad images
            # left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
            #                    constant_values=0)
            # # pad disparity gt
            # assert len(dataL.shape) == 2
            # dataL = np.lib.pad(dataL, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # print("top_pad",top_pad)
            # print("right_pad",right_pad)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


class myImageFloder_middlebury(data.Dataset):

    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.training = training
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)

        disp_L = self.disp_L[index]
        flag = False
        if '.png' in disp_L:
            dataL = self.dploader(disp_L)
            flag = True
        else:
            dataL, scaleL = self.dploader(disp_L)

        dataL[dataL==np.inf] = 0

        if self.training:
            w, h = left_img.size
            th, tw = 320, 640
            #th, tw = 320, 704

            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            right_img.flags.writeable = True
            if np.random.binomial(1, 0.2):
              sx = int(np.random.uniform(35, 100))
              sy = int(np.random.uniform(25, 75))
              cx = int(np.random.uniform(sx, right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy, right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx, cy-sy:cy+sy] = np.mean(np.mean(right_img, 0), 0)[np.newaxis,np.newaxis]

            if flag == True:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            else:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        
        else:
            dataL = np.ascontiguousarray(dataL, dtype=np.float32)

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)



class myImageFloder_middlebury_additional(data.Dataset):

    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity

        self.training = training
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]

        left_img = cv2.imread(left, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right, cv2.IMREAD_COLOR)

        resize_scale = 0.5
        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA,
        )

        left_img = Image.fromarray(left_img.astype('uint8')).convert('RGB')
        right_img = Image.fromarray(right_img.astype('uint8')).convert('RGB')

        disp_L = self.disp_L[index]
        flag = False

        if '.png' in disp_L:
            dataL = self.dploader(disp_L)
            flag = True
        else:
            dataL, scaleL = self.dploader(disp_L)

        dataL = (
            cv2.resize(
            dataL,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA,
            )
            * resize_scale
        )

        dataL[dataL==np.inf] = 0

        if self.training:
            w, h = left_img.size
            # th, tw = 256, 512
            th, tw = 320, 640

            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = np.array(right_img)
            left_img = np.array(left_img)

            right_img.flags.writeable = True
            if np.random.binomial(1, 0.2):
              sx = int(np.random.uniform(35, 100))
              sy = int(np.random.uniform(25, 75))
              cx = int(np.random.uniform(sx, right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy, right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx, cy-sy:cy+sy] = np.mean(np.mean(right_img, 0), 0)[np.newaxis,np.newaxis]

            if flag == True:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            else:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

        else:
            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            w, h = left_img.size

            # normalize
            processed = get_transform(augment=False)
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            top_pad = 1024 - h
            right_pad = 1536 - w
            assert top_pad >= 0 and right_pad >= 0

            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
            # pad disparity gt
            assert len(dataL.shape) == 2
            dataL = np.lib.pad(dataL, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # print("top_pad",top_pad)
            # print("right_pad",right_pad)
            return left_img, right_img, dataL


    def __len__(self):
        return len(self.left)



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs], self._pad
        # return [F.pad(x, self._pad, mode='constant', value=0) for x in inputs], self._pad
