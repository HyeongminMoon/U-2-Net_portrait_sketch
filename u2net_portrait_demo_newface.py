import cv2
import torch
from model import U2NET
from torch.autograd import Variable
import numpy as np
from glob import glob
import os

import argparse
import requests
import base64
import cv2
import json

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
        print("no face")
        return None
    
    conf = lists[0]['dots'][0]
    coor = lists[0]['coor']
    face_size = int(coor[3]-coor[1])
    
    resize_factor = int(0.7 * height / face_size)
    if conf < 0.8:
        return None
    print(resize_factor)
    if resize_factor > 3:
        print('high')
        im_face = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
    elif resize_factor > 2:
        print('midium')
        im_face = cv2.resize(img, (1600, 900), interpolation = cv2.INTER_AREA)
    else:
        print('low')
#         im_face = cv2.resize(img, (width*resize_factor,height*resize_factor), interpolation = cv2.INTER_AREA)
        im_face = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)

    return im_face

# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):

    # no face detected, return the whole image and the inference will run on the whole image
    if(face is None):
        return None
    (x, y, w, h) = face

    height,width = img.shape[0:2]

    # crop the face with a bigger bbox
    hmw = h - w
    # hpad = int(h/2)+1
    # wpad = int(w/2)+1

    l,r,t,b = 0,0,0,0
    lpad = int(float(w)*0.4)
    left = x-lpad
    if(left<0):
        l = lpad-x
        left = 0

    rpad = int(float(w)*0.4)
    right = x+w+rpad
    if(right>width):
        r = right-width
        right = width

    tpad = int(float(h)*0.6)
    top = y - tpad
    if(top<0):
        t = tpad-y
        top = 0

    bpad  = int(float(h)*0.2)
    bottom = y+h+bpad
    if(bottom>height):
        b = bottom-height
        bottom = height

    face_size = bottom - top
    
    resize_factor = int(0.7 * height / face_size)
    print(height, face_size, resize_factor)
    
#     im_face = img[top:bottom,left:right]
#     if(len(im_face.shape)==2):
#         im_face = np.repeat(im_face[:,:,np.newaxis],(1,1,3))

#     im_face = np.pad(im_face,((t,b),(l,r),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # pad to achieve image with square shape for avoding face deformation after resizing
#     hf,wf = im_face.shape[0:2]
#     if(hf-2>wf):
#         wfp = int((hf-wf)/2)
#         im_face = np.pad(im_face,((0,0),(wfp,wfp),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))
#     elif(wf-2>hf):
#         hfp = int((wf-hf)/2)
#         im_face = np.pad(im_face,((hfp,hfp),(0,0),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # resize to have 512x512 resolution
    
    if resize_factor > 3:
        print('high')
        im_face = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
    elif resize_factor > 2:
        print('midium')
        im_face = cv2.resize(img, (1600, 900), interpolation = cv2.INTER_AREA)
    else:
        print('low')
#         im_face = cv2.resize(img, (width*resize_factor,height*resize_factor), interpolation = cv2.INTER_AREA)
        im_face = cv2.resize(img, (1280 ,720), interpolation = cv2.INTER_AREA)
        
    

    return im_face

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

    if torch.cuda.is_available():
        tmpImg = Variable(tmpImg.cuda())
    else:
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

def main():

    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument('-p',default='./test_data/test_portrait_images/your_portrait_im', help='input image folder path')
    parser.add_argument('-o',default='./test_data/test_portrait_images/relearning_iter90000_face', help='output path')
    parser.add_argument('-m',default='/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/u2net_high/u2net_high_bce_itr_75000_train_0.858591_tar_0.078512.pth', help='model path')
    
    args = parser.parse_args()
    
    # get the image path list for inference
#     im_list = glob('./test_data/test_portrait_images/your_portrait_im/*')
    print(args.p)
    im_list = []
    for root, dirs, files in os.walk(args.p):
        for file in files:
            file_path = file.lower()
            if file_path.endswith('jpg') or file_path.endswith('png') or file_path.endswith('jpeg'):
                print(file_path)
                im_list.append(os.path.join(root,file))
    
    print("Number of images: ",len(im_list))
#     out_dir = args.o
#     model_dir = args.m
    
    
    model_paths = os.listdir('/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/custom_newface_erode')
    max_itr = 0
    max_itr_model = None
    for path in model_paths:
        if path == ".ipynb_checkpoints": continue
        prefit = path.split('_')[5]
        if int(prefit) > max_itr:
            max_itr = int(prefit)
            max_itr_model = path
#     print(max_itr_model)
    
    model_dir = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/custom_newface_erode/' + max_itr_model
    out_dir = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/test_data/test_portrait_images/custom_newface_erode_iter' + str(max_itr)
    
    
#     model_dir = './saved_models/custom_nobg/custom_nobg_bce_itr_60000_train_0.369949_tar_0.032807.pth'
    
#     out_dir = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/test_data/test_portrait_images/removed_bg_best'
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    # load u2net_portrait model
    if torch.cuda.is_available():
         torch.cuda.set_device(2)
    
    torch.cuda.empty_cache()
    
    print(torch.cuda.current_device())    
    net = U2NET(3,1)
    net.to('cuda:2')
    net.load_state_dict(torch.load(model_dir))
    

#     if torch.cuda.is_available():
#         net.cuda()
        
    with torch.no_grad():
        net.eval()

        # do the inference one-by-one
        for i in range(0,len(im_list)):
            print("--------------------------")
            print("inferencing ", i, "/", len(im_list), im_list[i])

            # load each image
            img = cv2.imread(im_list[i])
            height,width = img.shape[0:2]
            
#             im_face = detect_single_face(img)
#             im_face = crop_face(img, face)
#             if im_face is None:
#                 continue
#             img = cv2.resize(img, (480, 320), interpolation = cv2.INTER_AREA)
    
            im_portrait = inference(net,img)

#             dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (1280, 720))
            dst = (im_portrait*255).astype(np.uint8)
            # save the output
            cv2.imwrite(out_dir+"/"+im_list[i].split('/')[-1][0:-4]+'.png',dst)

if __name__ == '__main__':
    main()
