import sys
sys.path.append('core')
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from models.adstereo import ADStereo
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Efficient Stereo Matching with Adaptive Downsampling and Disparity Alignment (ADStereo)')
parser.add_argument('--loadmodel', default='/data1/ywang/my_projects/adstereo-main/fined/KITTI/checkpoint_310.tar', help='load model')
parser.add_argument('--save_path', default='./demo/', help='save path')

parser.add_argument('--save_numpy', default=True, action='store_false', help='save output as numpy arrays')
parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data1/ywang/dataset/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/*.png")
parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data1/ywang/dataset/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/*.png")
parser.add_argument('--maxdisp', type=int, default=160, help="max disp of geometry encoding volume")
parser.add_argument('--use_structure', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--refine', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')


args = parser.parse_args()
model = ADStereo(args.maxdisp, args.mixed_precision, False, False)

model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'], strict=True)

model = model.module
model.eval()

left_images = sorted(glob.glob(args.left_imgs, recursive=True))
right_images = sorted(glob.glob(args.right_imgs, recursive=True))
print(f"Found {len(left_images)} images.")


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
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


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


if __name__ == '__main__':

    fps_list = np.array([])
    videoWrite = cv2.VideoWriter('./ADStereo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1242, 750))
    if args.save_numpy:
        frames = []  # Save the output frames to numpy arrays 
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):

        temp_data = load_data(imfile1, imfile2)
        imgL, imgR, height, width = my_transform(temp_data, crop_height=384, crop_width=1280)
        img_left = Variable(imgL, requires_grad=False)
        img_right = Variable(imgR, requires_grad=False)

        model.eval()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        img_left = img_left.cuda()
        img_right = img_right.cuda()

        with torch.no_grad():
            disp_pr, _ = model(img_left, img_right)
            disp_pr = disp_pr.squeeze()
            if height <= 384 and width <= 1280:
                disp_pr = disp_pr[384 - height: 384, 0: width]
            else:
                disp_pr = disp_pr[:, :]
        end.record()
        
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        fps = 1000/runtime
        fps_list = np.append(fps_list, fps)
        if len(fps_list) > 5:
            fps_list = fps_list[-5:]
        avg_fps = np.mean(fps_list)
        print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

        disp_np = (2*disp_pr).data.cpu().numpy().squeeze().astype(np.uint8)
        if args.save_numpy:
            frames.append(disp_np)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)
        image_np = np.array(Image.open(imfile1)).astype(np.uint8)       
        out_img = np.concatenate((image_np, disp_np), 0)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, image_np.shape[0]+30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.imshow('img', out_img)
        cv2.waitKey(1)
        videoWrite.write(out_img)
    videoWrite.release()
    if args.save_numpy:
        np.savez('ADStereo.npz', *frames)