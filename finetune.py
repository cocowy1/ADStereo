from __future__ import print_function
import sys
sys.path.append("dataloader")
from torch.autograd import Variable
from models.adstereo import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import dataloader.datasets as DA

from models.loss import model_loss
import torch.backends.cudnn as cudnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
        
parser = argparse.ArgumentParser(description='GwcNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath_kitti2015', default='/data1/ywang/dataset/kitti_2015/training/',
                    help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_kitti', default='/data1/ywang/dataset/kitti_2012/training/',
                     help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_eth3d', default='/data1/ywang/dataset/eth3d/two_view_training/',
                     help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_middlebury', default='/data1/ywang/dataset/middlebury_half/trainingH/',
                     help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_middlebury_additional', default='/data1/ywang/dataset/middlebury_half/additionalF/',
                     help='datapath for sceneflow monkaa dataset')

####
parser.add_argument('--finetune_type', type=str, default="kit", choices=["kit", "eth3d", "mid"], help="choose the different data type for finetuning")
parser.add_argument('--epochs', type=int, default=800, help='number of epochs to train')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--gpus', type=int, nargs='+', default=[0])
parser.add_argument('--savemodel', default='./fined/KITTI/', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--print_freq', type=int, default=1, help='the frequency of printing losses (iterations)')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')

parser.add_argument('--use_structure', default=False, action='store_false', help='use mixed precision')
parser.add_argument('--refine', default=False, action='store_false', help='use mixed precision')
parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if args.finetune_type == 'kit':
    all_left_img_0, all_right_img_0, all_left_disp_0,  \
    test_left_img_0, test_right_img_0, test_left_disp_0 = DA.dataloader_KITTI2015(args.datapath_kitti2015)
    # #
    all_left_img_1, all_right_img_1, all_left_disp_1,  \
    test_left_img_1, test_right_img_1, test_left_disp_1 = DA.dataloader_KITTI(args.datapath_kitti)

    all_left_img = all_left_img_0 + all_left_img_1 
    all_right_img = all_right_img_0 + all_right_img_1
    all_left_disp = all_left_disp_0 + all_left_disp_1 

    train_dataset = DA.myImageFloder_KITTI(all_left_img, all_right_img, all_left_disp, training=True),
    test_dataset = DA.myImageFloder_KITTI(test_left_img_1 + test_left_img_0, test_right_img_1 + test_right_img_0, 
                           test_left_disp_1 + test_left_disp_0, training=False)

elif args.finetune_type == 'eth3d':
    all_left_img, all_right_img, all_left_disp = DA.dataloader_eth3d(args.datapath_eth3d)

    train_dataset = DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, training=True),
    test_dataset = DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, training=False)

elif args.finetune_type == 'mid':

    all_left_img_0, all_right_img_0, all_left_disp_0 = DA.dataloader_middlebury(args.datapath_middlebury)
    all_left_img_1, all_right_img_1, all_left_disp_1 = DA.dataloader_middlebury(args.datapath_middlebury_additional)

    all_left_img = all_left_img_0 + all_left_img_1 
    all_right_img = all_right_img_0 + all_right_img_1
    all_left_disp = all_left_disp_0 + all_left_disp_1 

    train_dataset = DA.myImageFloder_middlebury(all_left_img, all_right_img, all_left_disp, training=True),
    test_dataset = DA.myImageFloder_middlebury_additional(all_left_img_0, all_right_img_0, all_left_disp_0, training=False)
    
    
TrainImgLoader = torch.utils.data.DataLoader(
   train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)


scaler = torch.amp.GradScaler('cuda', args.mixed_precision)
model = ADStereo(args.maxdisp, args.mixed_precision, args.use_structure, args.refine)
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'], strict=True)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


def learning_rate_adjust(optimizer, epoch):
    if epoch < 300:
        lr = 0.001
    elif epoch < 600:
        lr = 0.0001
    # elif epoch <= 600:
    #     lr = 0.00005
    # elif epoch < 600:
    #     lr = 0.0001
    else:
        lr = 0.00001
    print('learning rate = %.5f'%(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def train(imgL, imgR, disp_true):
    model.train()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    disp_true = disp_true.unsqueeze(1)
    mask = ((disp_true < 160) & (disp_true > 0)).byte().bool()
    mask.detach_()
    # ----
    optimizer.zero_grad()
    outputs = model(imgL, imgR)
    loss = model_loss(outputs["disp"], disp_true, mask)
    epe = torch.mean(torch.abs(outputs["disp"][-1][mask] - disp_true[mask]))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.99)

    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), epe.item()


def mytest(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    disp_true = disp_true.unsqueeze(1)
    mask = (disp_true > 0) & (disp_true < args.maxdisp)

    if imgL.shape[2] % 64 != 0:
        times = imgL.shape[2] // 64
        top_pad = (times + 1) * 64 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 64 != 0:
        times = imgL.shape[3] // 64
        right_pad = (times + 1) * 64 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        final_disp, _ = model(imgL, imgR)

    b, _, h, w = final_disp.shape
    if top_pad != 0:
        pred_disp = final_disp[:, :, top_pad:, :w-right_pad]
    else:
        pred_disp = final_disp
    pred_disp = pred_disp.data.cpu()

    if len(disp_true[mask]) == 0:
        loss = 0.
        epe = 0.
        px1 = 0.
    else:
        loss = F.smooth_l1_loss(pred_disp[mask], disp_true[mask], reduction='mean')        
        epe = (pred_disp - disp_true).abs()
        epe = epe.view(-1)[mask.view(-1)]
        px1 = (epe >1).float().mean().item()
        px3 = (epe >3).float().mean().item()
        epe = epe.mean().item()
    return loss, epe, px1, px3


print("Traindataset is %d"%len(TrainImgLoader))
print("Testdataset is %d"%len(TestImgLoader))


def main():
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0.0
        total_train_epe = 0.0
        learning_rate_adjust(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        
            loss, epe = train(imgL_crop, imgR_crop, disp_crop_L)
            total_train_loss += loss
            total_train_epe += epe
        
        if (epoch + 1) % 1 == 0:
            avg_epe = total_train_epe / len(TrainImgLoader)
            avg_loss = total_train_loss / len(TrainImgLoader)
            print('Train Epoch----%5d of %d, train_loss---%.3f, train_EPE---%.3f' %
                  (epoch, len(TrainImgLoader), avg_loss, avg_epe))
        
        # SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        if epoch > 300:
            torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

        # ------------- TEST ------------------------------------------------------------
        if epoch > 300:
            total_test_loss = 0.0
            total_test_epe = 0.0
            total_test_1px = 0.0
            total_test_3px = 0.0
            start_time = time.time()
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, test_epe, test_px1, test_px3 = mytest(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_test_epe += test_epe
                total_test_1px += test_px1
                total_test_3px += test_px3
            if (epoch + 1) % 1 == 0:
                avg_epe = total_test_epe / len(TestImgLoader)
                avg_loss = total_test_loss / len(TestImgLoader)
                avg_px1 = total_test_1px / len(TestImgLoader)
                avg_px3 = total_test_3px / len(TestImgLoader)
                print('Test epoch----%5d pf %d, total loss---%.3f, total EPE---%.3f,\
                      total_px1---%.3f,  total_px3---%.3f' %
                    (epoch, len(TestImgLoader), avg_loss, avg_epe, avg_px1*100, avg_px3*100))


if __name__ == '__main__':
    main()

