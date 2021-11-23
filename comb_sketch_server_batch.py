#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import glob
from tqdm import tqdm
import requests
import json

import io
import json
from flask import Flask, jsonify, request
from PIL import Image
import csv
from model import U2NET
from torch.autograd import Variable
from data_loader import SalObjDataset

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

from data_loader import ToTensor
from data_loader import ToTensorLab

app = Flask(__name__)

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def inference(net,input):

    # normalize the input
    tmpImg = np.zeros((input.shape[0],input.shape[1],3))
    input = input/np.max(input)

    tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
    tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

    # convert BGR to RGB
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis,:,:,:]
    tmpImg = torch.from_numpy(tmpImg)

    # convert numpy array to torch tensor
    tmpImg = tmpImg.type(torch.FloatTensor)
    tmpImg = tmpImg.to(device)
#     if torch.cuda.is_available():
#         tmpImg = Variable(tmpImg.cuda())
#     else:
    tmpImg = Variable(tmpImg)

    # inference
    d1,d2,d3,d4,d5,d6,d7= net(tmpImg)

    # normalization
    pred = 1.0 - d1[:,0,:,:]
    pred = normPRED(pred)

    # convert torch tensor to numpy array
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    del d1,d2,d3,d4,d5,d6,d7

    return pred


def sketch_sk(img):
    with torch.no_grad():
        
        h, w = img.shape[:2]
        im_portrait = inference(net,img)
        dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (w, h))

    return dst


model_sk = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/custom_b2/custom_b2_bce_itr_15249_train_0.601421_tar_0.032526.pth'
net = U2NET(3,1)
torch_dict_sk = torch.load(model_sk, map_location='cpu')

with torch.no_grad():
    net.eval()

net.load_state_dict(torch_dict_sk)

device = 'cuda:1'
net.to(device)

batch_size = 1
print('-------------------------')


# @app.route('/sketch', methods=['POST'])
def sketch():
#     if request.method == 'POST':

#         r = request
#         data_json = r.data
#         data_dict = json.loads(data_json)
        
#         paths = data_dict['img_path'] #list
#     paths = glob.glob('/mnt/vitasoft/kobaco_batch/49_50_163_17/**/*')
#         root_path = ['49_50_163_17',
#      '101_101_210_17',
#      '49_50_174_192',
#      '49_50_161_173',
#      '14_49_44_68']
    root_path = ['49_50_163_17']
    for _root in root_path:
        paths = glob.glob(f'/mnt/vitasoft/kobaco_batch/{_root}_lama/**/*.png', recursive=True)
        print(f'/mnt/vitasoft/kobaco_batch/{_root}_lama/')
    #         if 'output_path' in data_dict:
    #             save_path = data_dict['output_path'] #list
    #         else:
    #             return(jsonify("output_path is None"))
        save_path = []
    #     if not os.path.isdir("/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/temp_sketch/"):
    #         os.mkdir("/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/temp_sketch/")
        for path in paths:
    #         output_path = path.replace("49_50_163_17", "49_50_163_17_sketch")

            output_path = path.replace("lama", "sketch")
            save_path.append(output_path)

        dataset = SalObjDataset(img_name_list = paths,
                        lbl_name_list = [],
                        transform=transforms.Compose([ToTensorLab(flag=0)]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        dst_paths = []
        write = None

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                try:
                    inputs = data['image']
                    inputs = inputs.type(torch.FloatTensor)
                    inputs_v = Variable(inputs.to(device), requires_grad=False)

                    d1,d2,d3,d4,d5,d6,d7= net(inputs_v)

                    # normalization
                    pred = 1.0 - d1[:,0,:,:]
                    pred = normPRED(pred)
                    # convert torch tensor to numpy array
                    pred = pred.squeeze()
                    pred = pred.cpu().data.numpy()
                    dst = (pred*255).astype(np.uint8)

                    if batch_size == 1:
                        thresh = dst.copy()
                        ret, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY) 
                        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2BGRA)
                        thresh[:,:,3][thresh[:,:,0]==255]=0
                        dst_path = save_path[i]
        #                     if 'type' in data_dict:
        #                         data_type = data_dict['type']
        #                         dst_path = dst_path.replace(".png", "_{}.png".format(data_type))
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        write = cv2.imwrite(dst_path, thresh)
                        dst_paths.append(dst_path)
                    else:
                        for im_idx, im in enumerate(dst):
                            thresh = im.copy()
                            ret, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY) 
                            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2BGRA)
                            thresh[:,:,3][thresh[:,:,0]==255]=0
                            idx = i*batch_size + im_idx
                            dst_path = save_path[idx]
        #                         if 'type' in data_dict:
        #                             data_type = data_dict['type']
        #                             dst_path = dst_path.replace(".png", "_{}.png".format(data_type))
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            write = cv2.imwrite(dst_path, thresh)
                            dst_paths.append(dst_path)
                except Exception as e:
                    print(e)
                    continue


#     return jsonify(result_dict)

import time
if __name__ == '__main__':
#     time.sleep(3600*11)
#     app.run(host='0.0.0.0', port=8982, debug=False)

    sketch()
#     app.run(host='0.0.0.0', port=8983, debug=False)