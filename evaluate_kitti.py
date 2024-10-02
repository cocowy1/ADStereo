from __future__ import print_function, division
from torch.autograd import Variable
import argparse
from models.adstereo import *

import os
import torch
import torch.nn as nn
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
from imageio import imread, imsave
import time

# -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--data_path', default='/data1/ywang/dataset/kitti_2015/testing/', help='datapath')
parser.add_argument('--test_list', type=str, default='./filenames/kitti2015_test.list',
                    help="training list")

parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--save_path', default='./disp_0/', help='save path')
parser.add_argument('--kit_type', type=str, default='kit15', help='choose differnet kitti types')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print_freq', type=int, default=1, help='the frequency of printing losses (iterations)')
parser.add_argument('--use_structure', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--refine', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')

args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = ADStereo(args.maxdisp, args.mixed_precision, False, False)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'], strict=True)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def my_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, 0: w] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, 0: crop_width]

    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def inference_kitti(index, leftname, rightname, savename):
    temp_data = load_data(leftname, rightname)
    imgL, imgR, height, width = my_transform(temp_data, crop_height=384, crop_width=1280)
    img_left = Variable(imgL, requires_grad=False)
    img_right = Variable(imgR, requires_grad=False)

    model.eval()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    img_left = img_left.cuda()
    img_right = img_right.cuda()

    with torch.no_grad():
        disp_pr, _ = model(img_left, img_right)
    disp_pr = disp_pr.squeeze().detach().cpu().numpy()
    
    torch.cuda.synchronize()
    print('%d-th sample with inference time = %.3f ms' % (index, (time.time() - start_time)*1000))

    if height <= 384 and width <= 1280:
        disp_pr = disp_pr[384 - height: 384, 0: width]
    else:
        disp_pr = disp_pr[:, :]

    imsave(savename, (disp_pr * 256).astype('uint16'))


def main():
    file_path = args.data_path
    file_list = args.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()

    start_full_time = time.time()
    for index in range(len(filelist)):
        current_file = filelist[index]
        
        if args.kit_type == 'kit15':
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
        elif args.kit_type == 'kit12':
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]

        savename = args.save_path + current_file[0: len(current_file) - 1]
        inference_kitti(index, leftname, rightname, savename)

    print('full time training time = %.4f seconds' % ((time.time() - start_full_time)))


if __name__ == '__main__':
    main()
