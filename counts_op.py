from __future__ import print_function
import sys
sys.path.append("dataloader")
import argparse
from models.adstereo import *
from thop import profile
import torch.nn.parallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with torch.cuda.device(0):
    parser = argparse.ArgumentParser(description='ADStereo')
    parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
    parser.add_argument('--use_structure', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--refine', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')

    args = parser.parse_args()

    model = ADStereo(args.maxdisp, args.mixed_precision, args.use_structure, args.refine)
    model.cuda() 
    model.eval()
    left = torch.randn(1, 3, 384, 1280).cuda()
    right = torch.randn(1, 3, 384, 1280).cuda()
    macs, params = profile(model, inputs=(left, right), verbose=True)
    print('{:<30}  {:<10}'.format('Computational complexity: ', macs/1000000000))
    print('{:<30}  {:<10}'.format('Number of parameters: ', params/1000000))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
