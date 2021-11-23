import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import random

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import RandomAffine, RandomJitter, RandomBlur, HeadJitter
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import cv2
from PIL import Image

import torch.multiprocessing as mp

# ------- 1. define loss function --------


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

# ------- 2. set the directory of training dataset --------

model_name = 'custom_multi' #'u2net'

data_dir = os.path.join(os.getcwd(), 'trainset_comb' + os.sep)
# tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
# tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)
tra_image_dir = os.path.join('im_saved' + os.sep)
tra_label_dir = os.path.join('gt_saved' + os.sep)
tra_mask_dir = os.path.join('mask_line_add' + os.sep)

val_data_dir = os.path.join(os.getcwd(), 'bad' + os.sep)
val_image_dir = os.path.join('val_im' + os.sep)
val_label_dir = os.path.join('result_bad_cases' + os.sep)


image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 10000
# epoch_num = 100
batch_size_train = 1

batch_size_val = 1
train_num = 0
val_num = 0

def trainer(rank, world_size):    

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = []
    tra_mask_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
        tra_mask_name_list.append(data_dir + tra_mask_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("train masks: ", len(tra_mask_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    # split_length = int(len(tra_img_name_list)*0.9)
    # print(split_length)

    val_img_name_list = glob.glob(val_data_dir + val_image_dir + '*' + image_ext)
    # print(tra_img_name_list)
    val_lbl_name_list = []
    for img_path in val_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        val_lbl_name_list.append(val_data_dir + val_label_dir + imidx + label_ext)



    val_img_name_list = val_img_name_list
    val_lbl_name_list = val_lbl_name_list
    # print(len(tra_img_name_list))
    print("val images: ", len(val_img_name_list))
    print("val labels: ", len(val_lbl_name_list))

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
    #         RescaleT(320),#320
#             RandomJitter(tra_mask_name_list),
#             RandomBlur(tra_mask_name_list),
#             HeadJitter(tra_mask_name_list),
            RandomAffine(),
            ToTensorLab(flag=0)]))

    salobj_val_dataset = SalObjDataset(
        img_name_list=val_img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
    #         RescaleT(320),#320
    #         RandomCrop(288),#288
            ToTensorLab(flag=0)]))
   
    # ------- 3. define model --------
    # define the net

    pretrained = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(salobj_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(salobj_val_dataset, shuffle=False)
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(salobj_val_dataset, batch_size=batch_size_val, num_workers=4, pin_memory=True, sampler=val_sampler)

    if(model_name=='custom_multi'):
        net = U2NET(3, 1)
    elif(model_name=='custom_u2netp2'):
        net = U2NETP(3,1)

    if pretrained:
        net.load_state_dict(torch.load(
            '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/saved_models/best/custom_aug4/custom_aug4_bce_itr_15477_train_0.527955_tar_0.038174.pth', map_location='cpu')
        )
    net = net.to(rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    bce_loss = nn.BCELoss(size_average=True)
    bce_loss = bce_loss.to(rank)
    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = bce_loss(d0,labels_v)
        loss1 = bce_loss(d1,labels_v)
        loss2 = bce_loss(d2,labels_v)
        loss3 = bce_loss(d3,labels_v)
        loss4 = bce_loss(d4,labels_v)
        loss5 = bce_loss(d5,labels_v)
        loss6 = bce_loss(d6,labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

        return loss0, loss
    
    
    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 10 # save the model every 2000 iterations
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            inputs_v, labels_v = Variable(inputs.to(rank), requires_grad=False), Variable(labels.to(rank),
                                                                                            requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if epoch % 10 == 0:
            print(epoch)
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    inputs, labels = data['image'], data['label']

                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)

                    inputs_v, labels_v = Variable(inputs.to(rank), requires_grad=False), Variable(labels.to(rank),
                                                                                                    requires_grad=False)

                    out_dir = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/results_multi/'
                    if(not os.path.exists(out_dir)):
                        os.mkdir(out_dir)

                    d1,d2,d3,d4,d5,d6,d7= net(inputs_v)

                    # normalization
                    pred = 1.0 - d1[:,0,:,:]
                    pred = normPRED(pred)

                    # convert torch tensor to numpy array
                    pred = pred.squeeze()
                    pred = pred.cpu().data.numpy()
                    dst = (pred*255).astype(np.uint8)
                    # save the output
                    cv2.imwrite(os.path.join(out_dir,model_name+f"_epoch{epoch}_{str(i).zfill(4)}.png" ),dst)

            if(not os.path.exists(model_dir)):
                os.mkdir(model_dir)

            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

#             os.system("nohup sh refine.sh val_nohub.log 2>&1 &")
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def init_process0(rank, world_size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9005'
    os.environ['WORLD_SIZE'] = '5'
    os.environ['RANK'] = '0'
    
    dist.init_process_group(
        backend=backend,
#         init_method='tcp://127.0.0.1:9007',
        rank=rank,
        world_size=world_size)
    fn(rank, world_size)
    
def init_process(rank, world_size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9005'
    os.environ['WORLD_SIZE'] = '5'
#     os.environ['RANK'] = 0
    
    dist.init_process_group(
        backend=backend,
#         init_method='tcp://127.0.0.1:9007',
        rank=rank,
        world_size=world_size)
    fn(rank, world_size)

def run(rank, size):
    pass

import time
if __name__ == '__main__':
#     time.sleep(3600*2)
    
    
    world_size = get_world_size()
    rank = get_rank()
    print("rank", rank,"world size", 3)
    
    processes= []
    mp.set_start_method("spawn")
    
    for rank in range(0,3):
        p = mp.Process(target=init_process0, args=(rank, 3, trainer))
        p.start()
        processes.append(p)
    
       
    for p in processes:
        p.join()
    
    print("process loaded") 
    print(processes)
    
#     trainer(rank,3)
 