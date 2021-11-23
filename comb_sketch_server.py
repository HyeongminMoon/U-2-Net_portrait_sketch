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
import pandas as pd
import requests

import json
import base64

import io
import json
from flask import Flask, jsonify, request
from PIL import Image
import csv
from model import U2NET
from torch.autograd import Variable

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

# parser = argparse.ArgumentParser(description="image and portrait composite")
# parser.add_argument('-m',
#                     default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/custom_aug4/custom_aug4_bce_itr_15477_train_0.527955_tar_0.038174.pth', 
#                     help='model path')
# parser.add_argument('-f')

# args = parser.parse_args()

model_sk = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/custom_b2/custom_b2_bce_itr_15249_train_0.601421_tar_0.032526.pth'
net = U2NET(3,1)
torch_dict_sk = torch.load(model_sk, map_location='cpu')

with torch.no_grad():
    net.eval()

net.load_state_dict(torch_dict_sk)

device = 'cuda:1'
net.to(device)
print('-------------------------')





@app.route('/sketch', methods=['POST'])
def sketch():
    if request.method == 'POST':

        r = request
        data_json = r.data
        data_dict = json.loads(data_json)
        
        path = data_dict['img_path']
        if 'output_path' in data_dict:
            save_path = data_dict['output_path']
        else:
            save_path = None
            
        img = cv2.imread(path)
        h, w = img.shape[:2]
        
        img = sketch_sk(img)
        dst = cv2.resize(img, dsize = (w, h))
        
        thresh = dst.copy()
        ret, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY) 
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2BGRA)
        thresh[:,:,3][thresh[:,:,0]==255]=0
    
        if 'output_path' in data_dict:
            dst_path = save_path
        else:
            dst_path = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/serverdata/' + path.split('/')[-1][:-4] + '.png'
        
        if 'type' in data_dict:
            data_type = data_dict['type']
            dst_path.replace(".png", "_{}.png".format(data_type))
        
        write = cv2.imwrite(dst_path, thresh)

        if not write:
            print(write, dst_path)
        result_dict = {}
        result_dict['output_path'] = dst_path

        return jsonify(result_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8899, debug=False)
#     app.run(host='0.0.0.0', port=8983, debug=False)