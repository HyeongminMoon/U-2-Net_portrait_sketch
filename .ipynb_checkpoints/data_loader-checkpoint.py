# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#==========================dataset load==========================
class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'],sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
        lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

        return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'],sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image,(new_h,new_w),mode='constant')
        lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx':imidx,'image':image, 'label':label}

class RandomAffine(object):

    def __init__(self):
        self.affine = transforms.RandomAffine(degrees=30, scale=(0.25, 2), fillcolor=255)
    
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # random horizontal flip
        if random.random() >= 0.5:
            image = image[:, ::-1]
            label = label[:, ::-1]
        
        # random affine
        if random.random() >= 0.5:
            scl = random.random() * 0.9 + 0.2
            deg = random.randint(-45, 45)

            self.affine.fillcolor=(255,255,255)
            self.affine.scale =(scl, scl)
            self.affine.degrees =(deg, deg)

            image = Image.fromarray(image)
            label = label[:,:,-1]
            label = Image.fromarray(label)

            image = self.affine(image)
            self.affine.fillcolor=0
            label = self.affine(label)

            image = np.array(image)
            label = np.array(label)
            label = np.expand_dims(label, axis=2)

class RandomAffineJitter(object):

    def __init__(self):
        self.affine = transforms.RandomAffine(degrees=30, scale=(0.25, 2), fillcolor=255)
        self.jitter = transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5)
        
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # random horizontal flip
        if random.random() >= 0.5:
            image = image[:, ::-1]
            label = label[:, ::-1]
        
        # random affine
        if random.random() >= 0.5:
            scl = random.random() * 0.9 + 0.2
            deg = random.randint(-45, 45)

            self.affine.fillcolor=(255,255,255)
            self.affine.scale =(scl, scl)
            self.affine.degrees =(deg, deg)

            image = Image.fromarray(image)
            label = label[:,:,-1]
            label = Image.fromarray(label)

            image = self.affine(image)
            image = self.jitter(image)

            self.affine.fillcolor=0
            label = self.affine(label)

            image = np.array(image)
            label = np.array(label)
            label = np.expand_dims(label, axis=2)
        
        
        return {'imidx':imidx,'image':image, 'label':label}    
    
class RandomJitter(object):

    def __init__(self, people_mask_list=None):
        self.jitter = transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5)
        self.people_mask_list = people_mask_list
    
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # random horizontal flip
#         if random.random() >= 0.5:
#             image = image[:, ::-1]
#             label = label[:, ::-1]
        
        # random affine
        if random.random() >= 0.5:
#             print('RandomJitter')
#             img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
            mask = cv2.imread(self.people_mask_list[imidx[0]])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#             mask = 255 - mask
            
            img = image.copy()
            img = Image.fromarray(img)

            img = self.jitter(img)

            img = np.array(img)
            
            mask_inv = 255 - mask
            img_fg = cv2.bitwise_and(img, img, mask=mask)
#             img_bg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
            image = img_fg + img_bg
        
        return {'imidx':imidx,'image':image, 'label':label}    

class RandomBlur(object):

    def __init__(self, people_mask_list=None):
        self.jitter = transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5)
        self.people_mask_list = people_mask_list
        
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        dst = image.copy()
        # random blur
        if True:
#             print("blur")
            mask1 = cv2.imread(self.people_mask_list[imidx[0]])
            mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
#             plt.imshow(mask1,cmap='gray')
#             plt.show()
            end_flag = True
            while(end_flag):
                img = image.copy()
                mask = mask1.copy()
#                 img_gray = img_gray1.copy()
                try:
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                    find_max = -1000
                    max_idx = -1
                    for idx, contour in enumerate(contours):
                        np_max = np.sum(contour)
                        if find_max < np_max:
                            find_max = np_max
                            max_idx = idx

                    target_coor = tuple(contours[max_idx][random.randint(0, len(contours[max_idx]))][0])

                    blur_size = random.randint(5,14)
                    
                    repeat = random.randint(1,3)

                    for a in range(repeat):
                        target_coor = tuple(contours[max_idx][random.randint(0, len(contours[max_idx]))][0])
                        x = target_coor[1]
                        y = target_coor[0]
                        blur_h = random.randint(100, 200)
                        blur_h2 = random.randint(100, 200)
                        blur_w = random.randint(100, 200)
                        blur_w2 = random.randint(100, 200)
                        patch = img[x-blur_w2:x+blur_w,y-blur_h2:y+blur_h]
#                         if random.random() > 0.5:
                        patch =cv2.blur(patch, (blur_size,blur_size))
#                         else:
#                             patch = Image.fromarray(patch)
#                             patch = self.jitter(patch)
#                             patch = np.array(patch)
#                             blured = cv2.filter2D(patch, -1, kernel)
                        img[x-blur_w2:x+blur_w,y-blur_h2:y+blur_h] = patch

                    end_flag = False
                except Exception as e:
#                     print(e)
                    continue
                    
        return {'imidx':imidx,'image':img, 'label':label}
    
class HeadJitter(object):

    def __init__(self, people_mask_list=None):
        self.jitter = transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5)
        self.people_mask_list = people_mask_list
        
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        dst = image.copy()
        # random blur
        if True:
#             print("headjitter")
            mask1 = cv2.imread(self.people_mask_list[imidx[0]])
            mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
#             plt.imshow(mask1,cmap='gray')
#             plt.show()
            end_flag = True
            while(end_flag):
                img = image.copy()
                mask = mask1.copy()
#                 img_gray = img_gray1.copy()
                try:
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                    find_max = -1000
                    max_idx = -1
                    for idx, contour in enumerate(contours):
                        np_max = np.sum(contour)
                        if find_max < np_max:
                            find_max = np_max
                            max_idx = idx
                    
                    x_list = sorted(np.unique(contours[max_idx][:,:,0]))
                    y_list = sorted(np.unique(contours[max_idx][:,:,1]))
                    min_x, max_x = x_list[0], x_list[-1]
                    randomint = random.randint(5,11)
                    mid_y = int((y_list[0]*randomint + y_list[-1]) / (1 + randomint))
                    mask3 = mask.copy()
                    mask3[mid_y:,:] = 0
                    mask3[:,max_x:] = 0
                    mask3[:,:min_x] = 0

                    img_fg = cv2.bitwise_and(img, img, mask=mask3)
                    img_fg = Image.fromarray(img_fg)
                    img_fg = self.jitter(img_fg)
                    img_fg = np.array(img_fg)
                    img_bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask3))

                    dst = img_fg + img_bg    
                    
                    end_flag = False
                except Exception as e:
                    continue
                    
        return {'imidx':imidx,'image':dst, 'label':label}       
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        tmpLbl = np.zeros(label.shape)

        image = image/np.max(image)
        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpLbl[:,:,0] = label[:,:,0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label =sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        # change the color space
        if self.flag == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

        elif self.flag == 1: #with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
#             print(np.max(image))
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpLbl[:,:,0] = label[:,:,0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

    

class SalObjDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list,transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
#         self.label_name_list = mask_name_list
        self.transform = transform
#         self.mask = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if(0==len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if(3==len(label_3.shape)):
            label = label_3[:,:,0]
        elif(2==len(label_3.shape)):
            label = label_3
            
#         if(0==len(self.label_name_list)):
#             mask_3 = np.zeros(mask.shape)
#         else:
#             mask_3 = io.imread(self.label_name_list[idx])

#         mask = np.zeros(mask_3.shape[0:2])
#         if(3==len(mask_3.shape)):
#             mask = mask_3[:,:,0]
#         elif(2==len(mask_3.shape)):
#             mask = mask_3

        if(3==len(image.shape) and 2==len(label.shape)):
            label = label[:,:,np.newaxis]
#             mask = mask[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(label.shape)):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]
#             mask = mask[:,:,np.newaxis]
            
#         if(3==len(image.shape) and 2==len(mask.shape)):
#             mask = mask[:,:,np.newaxis]
#         elif(2==len(image.shape) and 2==len(mask.shape)):
#             image = image[:,:,np.newaxis]
#             mask = mask[:,:,np.newaxis]

#         sample = {'imidx':imidx, 'image':image, 'label':label, 'mask':mask}
        sample = {'imidx':imidx, 'image':image, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class InferDataset(Dataset):
    def __init__(self,img_name_list,transform=None):
        self.image_name_list = img_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if(2==len(image.shape)):
            image = image[:,:,np.newaxis]
            
        sample = {'imidx':imidx, 'image':image}

        if self.transform:
            sample = self.transform(sample)

        return sample

import cv2
class FillboxDataset(Dataset):
    def __init__(self,img_name_list,transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
# 		self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = cv2.imread(self.image_name_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

# 		if(0==len(self.label_name_list)):
# 			label_3 = np.zeros(image.shape)
# 		else:
# 			label_3 = io.imread(self.label_name_list[idx])
        
        label_3 = image.copy()
        label_3 = 255-label_3
#         image = 255-image
#         print("shape: ", image.shape)

        label = np.zeros(label_3.shape[0:2])
        if(3==len(label_3.shape)):
            label = label_3[:,:,0]
        elif(2==len(label_3.shape)):
            label = label_3

        if(3==len(image.shape) and 2==len(label.shape)):
            label = label[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(label.shape)):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]


        h, w = image.shape[:2]
        max_box_h = (h//4)*3
        max_box_w = (w//4)*3

#         yellow color
        color=(255, 255, 0)

        per_img_box_cnt = random.randint(1,5)    
        for _ in range(per_img_box_cnt):
            box_h = random.randint(30,max_box_h)
            box_w = random.randint(30,max_box_w)

            box_y = random.randint(0, h-30)
            box_x = random.randint(0, w-30)

            image = cv2.rectangle(image,(box_x, box_y), (box_x + box_w, box_y + box_h), color, thickness=-1)
#             B = pil_draw_rect(B, (box_x, box_y), (box_x + box_w, box_y + box_h))

        
#         cv2.imwrite("./testA.png", image)
#         cv2.imwrite("./testB.png", label)
        
        
        sample = {'imidx':imidx, 'image':image, 'label':label}

        
        
        if self.transform:
            sample = self.transform(sample)

        return sample