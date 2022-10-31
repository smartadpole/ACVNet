# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2
from file import Walk, MkdirSimple
from tqdm.contrib import tzip
from torchvision import transforms
from PIL import Image

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def GetArgs():
    parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
    parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
    parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
    parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',help='load the weights from a specific checkpoint')
    parser.add_argument('--output', type=str)

    # parse arguments, set seeds
    args = parser.parse_args()

    return args

def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = Walk(path, ['jpg', 'png', 'jpeg'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
    elif 'indemind' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len


def LoadModel(args):
    # model, optimizer
    model = __models__[args.model](args.maxdisp, False, False)
    model = nn.DataParallel(model)
    model.cuda()

    # load parameters
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

    return model

def  WriteDepth(depth, limg,  path, name):
    output_concat =  os.path.join(path, "concat", name)
    output_gray =  os.path.join(path, "gray", name)
    output_color =  os.path.join(path, "color", name)
    MkdirSimple(output_concat)
    MkdirSimple(output_gray)
    MkdirSimple(output_color)

    predict_np = depth.squeeze().cpu().numpy()

    disp = depth

    predict_np = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np, cv2.COLORMAP_HOT)
    limg_cv = cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img = np.vstack([limg_cv, color_img])

    cv2.imwrite(output_concat, concat_img)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_gray, predict_np)

def test(args):

    model = LoadModel(args)

    left_files, right_files, root_len = GetImages(args.datapath)

    if len(left_files) == 0:
        left_files, right_files, root_len = GetImages(args.datapath, 'indemind')

    for left_image_file, right_image_file in tzip(left_files, right_files):
        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            continue

        output_name = left_image_file[root_len+1:]

        limg = Image.open(left_image_file).convert('RGB')
        rimg = Image.open(right_image_file).convert('RGB')

        # why crop
        w, h = limg.size
        # limg = limg.crop((w - 1232, h - 368, w, h))
        # rimg = rimg.crop((w - 1232, h - 368, w, h))

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            disp_ests = model(limg_tensor, rimg_tensor, False)

        WriteDepth(disp_ests[0], limg, args.output, output_name)

if __name__ == '__main__':
    args = GetArgs()
    test(args)
