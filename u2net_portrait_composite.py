import os
from skimage import io, transform
from skimage.filters import gaussian
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

import argparse

import cv2

def detect_single_face(face_cascade,img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if(len(faces)==0):
        print("Warming: no face detection, the portrait u2net will run on the whole image!")
        return None

    # filter to keep the largest face
    wh = 0
    idx = 0
    for i in range(0,len(faces)):
        (x,y,w,h) = faces[i]
        if(wh<w*h):
            idx = i
            wh = w*h

    return faces[idx]

# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):

    # no face detected, return the whole image and the inference will run on the whole image
    if(face is None):
        return img
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


    im_face = img[top:bottom,left:right]
    if(len(im_face.shape)==2):
        im_face = np.repeat(im_face[:,:,np.newaxis],(1,1,3))

    im_face = np.pad(im_face,((t,b),(l,r),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # pad to achieve image with square shape for avoding face deformation after resizing
    hf,wf = im_face.shape[0:2]
    if(hf-2>wf):
        wfp = int((hf-wf)/2)
        im_face = np.pad(im_face,((0,0),(wfp,wfp),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))
    elif(wf-2>hf):
        hfp = int((wf-hf)/2)
        im_face = np.pad(im_face,((hfp,hfp),(0,0),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # resize to have 512x512 resolution
    im_face = cv2.resize(im_face, (512,512), interpolation = cv2.INTER_AREA)

    return im_face


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,sigma=2,alpha=0.5):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    image = io.imread(image_name)
    pd = transform.resize(predict_np,image.shape[0:2],order=2)
    pd = pd/(np.amax(pd)+1e-8)*255
    pd = pd[:,:,np.newaxis]

    print(image.shape)
    print(pd.shape)

    ## fuse the orignal portrait image and the portraits into one composite image
    ## 1. use gaussian filter to blur the orginal image
    sigma=sigma
    image = gaussian(image, sigma=sigma, preserve_range=True)

    ## 2. fuse these orignal image and the portrait with certain weight: alpha
    alpha = alpha
    im_comp = image*alpha+pd*(1-alpha)

    print(im_comp.shape)


    img_name = image_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    io.imsave(d_dir+'/'+imidx+'_sigma_' + str(sigma) + '_alpha_' + str(alpha) + '_composite.png',im_comp)

def main():

    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument('-s',action='store',dest='sigma')
    parser.add_argument('-a',action='store',dest='alpha')
    args = parser.parse_args()
    print(args.sigma)
    print(args.alpha)
    print("--------------------")

    # --------- 1. get image path and name ---------
    model_name='u2net_portrait'#u2netp


    image_dir = './test_data/test_portrait_images/your_portrait_im'
    
    model_paths = os.listdir('/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/u2net')
    max_itr = 0
    max_itr_model = None
    for path in model_paths:
        if path == ".ipynb_checkpoints": continue
        prefit = path.split('_')[3]
        if int(prefit) > max_itr:
            max_itr = int(prefit)
            max_itr_model = path
    print(max_itr_model)
    
    prediction_dir = './test_data/test_portrait_images/u2net_results_iter' + str(max_itr)
#     prediction_dir = './test_data/test_portrait_images/u2net_results_iter35000'
    
    if(not os.path.exists(prediction_dir)):
        os.mkdir(prediction_dir)

    model_dir = './saved_models/u2net/' + max_itr_model
#     model_dir = './saved_models/u2net/u2net_bce_itr_35000_train_0.820537_tar_0.064011.pth'
    img_name_list = glob.glob(image_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    

        
    print("...load U2NET---173.6 MB")
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
    print(torch.cuda.current_device())
    net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir), )

    if torch.cuda.is_available():
        net.cuda()    
        
    with torch.no_grad():
        net.eval()
        
        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = 1.0 - d1[:,0,:,:]
            pred = normPRED(pred)

            # save results to test_results folder
            save_output(img_name_list[i_test],pred,prediction_dir,sigma=float(args.sigma),alpha=float(args.alpha))

            del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
