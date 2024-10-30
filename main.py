from __future__ import print_function, division
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.adstereo import *

import dataloader.datasets as DA
from models.metrics import METRICS
from models.loss import model_loss
cudnn.benchmark = True

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
        
parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--datapath', default='/data1/ywang/dataset/SceneFlow/', help='datapath')
parser.add_argument('--print_freq', type=int, default=200, help='the freuency of printing losses (iterations)')
parser.add_argument('--lrepochs', type=str, default="14,20,26,30:2", help='the epochs to decay lr: the downscale rate')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--savemodel', default=None, help='save model')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--use_structure', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--refine', default=False, action='store_true', help='use mixed precision')
parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')

# parse arguments, set seeds
args = parser.parse_args()

# dataset, dataloader
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    # parse downscale rate (after :)
    downscale_rate = float(splits[1])
    print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# model, optimizer
model = ADStereo(args.maxdisp, args.mixed_precision, args.use_structure, args.refine)
model = nn.DataParallel(model)
model.cuda()

scaler = torch.amp.GradScaler('cuda', args.mixed_precision)

# load parameters
if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = DA.dataloader_SceneFlow(
    args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True),
    batch_size=32, shuffle=True, num_workers=1, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1, shuffle=False, num_workers=1, drop_last=False)


def train(imgL, imgR, disp_true, epoch):
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
    
    if 'structure_discrepancy' in outputs:
        loss += outputs["structure_discrepancy"] 
        
    epe = torch.mean(torch.abs(outputs["disp"][-1][mask] - disp_true[mask]))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), epe.item()


def mytest(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

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
        output3, _ = model(imgL, imgR)
    if top_pad != 0:
        pred_disp = output3[:, :, top_pad:, :]
    else:
        pred_disp = output3
    # pred_disp = pred_disp.data.cpu()


    if len(disp_true[mask]) == 0:
        loss = 0
        metrics = {
            'epe': 0,
            'edge_epe': 0,
            '1px': 0,
            '3px': 0,
            'D1': 0,
        }
        distance = 0.
        print('it meet a 0 number')

    else:
        loss = F.smooth_l1_loss(pred_disp[mask], disp_true[mask], reduction='mean')
        # epe = torch.mean(torch.abs(pred_disp[mask]-disp_true[mask]))  # end-point-error
        # epe = F.l1_loss(pred_disp[mask], disp_true[mask], reduction='mean')
        epe = (pred_disp - disp_true).abs()
        epe = epe.view(-1)[mask.view(-1)]
        edge_epe = METRICS()(disp_true, pred_disp)
        metrics = {
            'epe': epe.mean().item(),
            'edge_epe': edge_epe,
            '1px': (epe > 1).float().mean().item(),
            '3px': (epe > 3).float().mean().item(),
            'D1': (epe > 5).float().mean().item(),
        }
    return loss, metrics



def main():
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        print('This is %d-th epoch, gwc_2x_adaptive.py' % (epoch))
        total_train_loss = 0.0
        total_train_epe = 0.0
        adjust_learning_rate(optimizer, epoch, args.lr, args.lrepochs)

        # # ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        
            loss, epe = train(imgL_crop, imgR_crop, disp_crop_L, epoch)
            total_train_loss += loss
            total_train_epe += epe
        
            if batch_idx % args.print_freq == 0:
                print('### batch_idx %5d of total %5d, loss---%.3f, EPE---%.3f ###' %
                      (batch_idx + 1,
                       len(TrainImgLoader),
                       float(total_train_loss / (batch_idx + 1)),
                       float(total_train_epe / (batch_idx + 1))))
        
        print('epoch %d total train loss = %.3f, total train epe = %.4f' % (epoch,
                                                                            total_train_loss / len(TrainImgLoader),
                                                                            total_train_epe / len(TrainImgLoader)))

        # SAVE
        if epoch > 10:
            savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
            }, savefilename)


        # ------------- TEST ------------------------------------------------------------
        if epoch > -1:
            total_test_loss = 0.0
            total_test_epe = 0.0
            total_test_edge_epe = 0.0
            total_test_1px = 0.0
            total_test_3px = 0.0
            total_test_D1 = 0.0
            total_distance = 0.0

            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, metrics = mytest(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_test_epe += metrics['epe']
                total_test_edge_epe += metrics['edge_epe']
                total_test_1px += metrics['1px']
                total_test_3px += metrics['3px']
                total_test_D1 += metrics['D1']

            print(
                'epoch %d, total test loss = %.3f, total_test_epe = %.4f, total_test_edge_epe = %.4f, total_distance=%.4f, total_test_1px=%.4f, total_test_3px=%.4f, total_test_D1=%.4f' % (
                epoch,
                total_test_loss / len(TestImgLoader), total_test_epe / len(TestImgLoader),
                total_test_edge_epe / len(TestImgLoader), total_distance / len(TestImgLoader),
                100 * total_test_1px / len(TestImgLoader),  100 * total_test_3px / len(TestImgLoader),
                 100 * total_test_D1 / len(TestImgLoader)))

if __name__ == '__main__':
    main()
