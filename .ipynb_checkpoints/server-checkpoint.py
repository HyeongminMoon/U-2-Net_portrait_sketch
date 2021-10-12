#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from glob import glob
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


# In[2]:


parser = argparse.ArgumentParser(description="image and portrait composite")
# parser.add_argument('-p',default='./test_data/test_portrait_images/your_portrait_im', help='input image folder path')
parser.add_argument('-o',default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/test_data/test_portrait_images/api_results', help='output path')
parser.add_argument('-m1',default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/train_APDrawingGAN/u2net_bce_itr_50000_train_0.806755_tar_0.059212.pth', help='model path')
parser.add_argument('-m2',default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/train_custom_nobg/custom_nobg_bce_itr_60000_train_0.369949_tar_0.032807.pth', help='model path')
parser.add_argument('-m3',default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/seg_detection/u2net_human_seg.pth', help='model path')
parser.add_argument('-f')

args = parser.parse_args()

model_dir_thin = args.m1
model_dir_thick = args.m2
model_dir_nobg = args.m3

# if torch.cuda.is_available():
#     torch.cuda.set_device(2)
# print(torch.cuda.current_device())   

net = U2NET(3,1)

torch_dict_thin = torch.load(model_dir_thin)
torch_dict_thick = torch.load(model_dir_thick)
torch_dict_nobg = torch.load(model_dir_nobg)

# if torch.cuda.is_available():
#     net.cuda()

device = 'cuda:1'


net.to(device)

with torch.no_grad():
    net.eval()
    
print('-------------------------')

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

target_url = "http://localhost:8912/predict"

def detect_single_face(img):
#     img = cv2.imread('/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/train_data/im2/804486H01_1_21-TJ-04-027_00534.jpg')
    # img = cv2.resize(img, (640,480))
    height,width = img.shape[0:2]
    ori_img = img.copy()


    _, img_encoded = cv2.imencode('.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 50])
    img = cv2.imdecode(img_encoded, 1)
    # send http request with image and receive response
    jpg_as_text = base64.b64encode(img_encoded).decode()
    dict = {}
    dict['image'] = jpg_as_text
    dict['shape'] = ori_img.shape

    response = requests.post(target_url, data=json.dumps(dict))

    # print('network inference 시간{0:0.2f}'.format(time.time()-strt ))
    # draw_time = time.time()
    lists = json.loads(response.text)

    if len(lists) == 0:
#         print("no face")
        return img
    
    conf = lists[0]['dots'][0]
    coor = lists[0]['coor']
    face_size = int(coor[3]-coor[1])
    
    resize_factor = int(0.7 * height / face_size)
    if conf < 0.8:
        return img
#     print(resize_factor)
    if resize_factor > 3:
#         print('high')
#         im_face = cv2.resize(img, (1600, 900), interpolation = cv2.INTER_AREA)
        im_face = cv2.resize(img, (2560, 1440), interpolation = cv2.INTER_AREA)
    elif resize_factor > 2:
#         print('midium')
#         im_face = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
        im_face = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
    else:
#         print('low')
#         im_face = cv2.resize(img, (width*resize_factor,height*resize_factor), interpolation = cv2.INTER_AREA)
#         im_face = cv2.resize(img, (960 ,540), interpolation = cv2.INTER_AREA)
        im_face = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)

    return im_face

@app.route('/sketch_thin', methods=['POST'])
def sketch_thin():
    global Model_Flag
    
    if request.method == 'POST':
        
        net.load_state_dict(torch_dict_thin)

        r = request
        data_json = r.data
        data_dict = json.loads(data_json)

        file_path = data_dict['path']
#         print(file_path)

        if not os.path.exists(file_path):
            print("input 경로에 이미지가 없습니다.")
            return jsonify([])
        
        im_list = []
        if file_path.endswith('jpg') or file_path.endswith('png') or file_path.endswith('jpeg'):
#             print(file_path)
            im_list.append(file_path)
        else:
            print("이미지가 아닙니다.")
            return jsonify([])
        
        out_dir = args.o
        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)
            
        with torch.no_grad():
#             for i in range(0,len(im_list)):
#             print("--------------------------")
#             print("inferencing ", 0, "/", len(im_list), im_list[0])

            # load each image
            img = cv2.imread(im_list[0])
#                 height,width = img.shape[0:2]
            im_face = detect_single_face(img)
            if im_face is None:
                return jsonify([])
#             im_face = crop_face(img, face)

            im_portrait = inference(net,im_face)
#             im_portrait = inference(net,img)

            dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (1280, 720))
#             dst = (im_portrait*255).astype(np.uint8)
#             blr = cv2.GaussianBlur(dst, (0, 0), 2)
#             dst2 = np.clip(2.0*dst - blr, 0, 255).astype(np.uint8)
        
            # save the output
            out_path = out_dir+"/"+im_list[0].split('/')[-1][0:-4]+'.png'
#             print(out_path)
            if os.path.exists(out_path):
                os.remove(out_path)
            
            cv2.imwrite(out_path, dst)
            
        result_dict = {}
        result_dict['output_path'] = out_path
        return jsonify(result_dict)

@app.route('/sketch_thick', methods=['POST'])
def sketch_thick():
    if request.method == 'POST':
        
        net.load_state_dict(torch_dict_thick)
        
        r = request
        data_json = r.data
        data_dict = json.loads(data_json)

        file_path = data_dict['path']
        print(file_path)

        if not os.path.exists(file_path):
            print("input 경로에 이미지가 없습니다.")
            return jsonify([])
        
        im_list = []
        if file_path.endswith('jpg') or file_path.endswith('png') or file_path.endswith('jpeg'):
#             print(file_path)
            im_list.append(file_path)
        else:
            print("이미지가 아닙니다.")
            return jsonify([])
        
        out_dir = args.o
        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)
            
        with torch.no_grad():
#             for i in range(0,len(im_list)):
#             print("--------------------------")
#             print("inferencing ", 0, "/", len(im_list), im_list[0])

            # load each image
            img = cv2.imread(im_list[0])
#                 height,width = img.shape[0:2]
            im_face = detect_single_face(img)

            im_portrait = inference(net,im_face)
    
#             im_portrait = inference(net,img)

            dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (1280, 720))
#             dst = (im_portrait*255).astype(np.uint8)
#             blr = cv2.GaussianBlur(dst, (0, 0), 2)
#             dst2 = np.clip(2.0*dst - blr, 0, 255).astype(np.uint8)
        
            # save the output
            out_path = out_dir+"/"+im_list[0].split('/')[-1][0:-4]+'.png'
#             print(out_path)
            if os.path.exists(out_path):
                os.remove(out_path)
            
            cv2.imwrite(out_path, dst)
            
        result_dict = {}
        result_dict['output_path'] = out_path
        return jsonify(result_dict)
    
@app.route('/remove_bg', methods=['POST'])
def remove_bg():
    global Model_Flag
    
    if request.method == 'POST':
        
        net.load_state_dict(torch_dict_nobg)

        r = request
        data_json = r.data
        data_dict = json.loads(data_json)

        file_path = data_dict['img_path']
#         print(file_path)

        if not os.path.exists(file_path):
            print("input 경로에 이미지가 없습니다.")
            return jsonify([])
        
        im_list = []
        if file_path.endswith('jpg') or file_path.endswith('png') or file_path.endswith('jpeg'):
#             print(file_path)
            im_list.append(file_path)
        else:
            print("이미지가 아닙니다.")
            return jsonify([])
        
        out_dir = args.o
        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)
            
        with torch.no_grad():
#             for i in range(0,len(im_list)):
#             print("--------------------------")
#             print("inferencing ", 0, "/", len(im_list), im_list[0])
#             print(im_list[0])
            img = cv2.imread(im_list[0])
            h, w = img.shape[:2]
        
            r = 360 / float(h)
            dim = (int(w * r), 360)
        
            resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            im_portrait = inference(net, resized_img)

            dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (w, h))

#             print(dst.shape)
#             mask2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            ret, mask2 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             kernel = np.ones((5,5), np.uint8)
#             mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
#             mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            mask2_inv = cv2.bitwise_not(mask2)

            # dst_bg = cv2.bitwise_and(dst_img_copy, dst_img_copy, mask=mask_inv)
#             print(mask2.shape)
#             print(img.shape)
            
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            img_fg = cv2.bitwise_and(img, img, mask=mask2_inv)
            img_bg = cv2.bitwise_and(dst, dst, mask=mask2)

            dst = img_fg + img_bg
            
            # save the output
#             out_path = out_dir+"/"+im_list[0].split('/')[-1][0:-4]+'.png'
            out_path = os.path.join(out_dir,im_list[0].split('/')[-1][:-4]+'.png')
#             print(out_path)
            if os.path.exists(out_path):
                os.remove(out_path)
            
            cv2.imwrite(out_path, dst)
            
        result_dict = {}
        result_dict['output_path'] = out_path
        return jsonify(out_path)


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8895, debug=False)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script s')


# In[ ]:




