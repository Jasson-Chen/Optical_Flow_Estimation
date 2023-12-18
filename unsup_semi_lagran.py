import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from netCDF4 import Dataset
import torch.utils.data as data
import glob
from tqdm import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from datetime import datetime
# from Horn_Schunck_method import *

PRINT_INTERVAL = 500
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

## Model 
def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.ReLU(inplace=True))

def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, 5, stride=1, padding=2, bias=False)

def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.ReLU(inplace=True))

def concatenate(tensor1, tensor2, tensor3):
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    _, _, h3, w3 = tensor3.shape
    h, w = min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :h, :w], tensor2[:, :, :h, :w], tensor3[:, :, :h, :w]), 1)

def generate_grid(B, H, W, device):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = torch.transpose(grid, 1, 2)
    grid = torch.transpose(grid, 2, 3)
    grid = grid.to(device)
    return grid

class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(6, 64, kernel_size=7, stride=1)
        self.conv2 = conv(64, 128, kernel_size=5)
        self.conv3 = conv(128, 256, kernel_size=5)
        self.conv3_1 = conv(256, 256, stride=1)
        self.conv4 = conv(256, 512)
        self.conv4_1 = conv(512, 512, stride=1)
        self.conv5 = conv(512, 512)
        self.conv5_1 = conv(512, 512, stride=1)
        self.conv6 = conv(512, 1024)

        self.predict_flow6 = predict_flow(1024)  # conv6 output
        self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
        self.predict_flow4 = predict_flow(770)  # upconv4 + 2 + conv4_1
        self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
        self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2
        self.predict_flow1 = predict_flow(98) 

        self.upconv5 = upconv(1024, 512)
        self.upconv4 = upconv(1026, 256)
        self.upconv3 = upconv(770, 128)
        self.upconv2 = upconv(386, 64)
        self.upconv1 = upconv(194, 32)

        self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):

        out_conv1 = self.conv1(x) # 64, 128, 128
        out_conv2 = self.conv2(out_conv1) # 128, 64, 64 
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) # 256, 32, 32
        out_conv4 = self.conv4_1(self.conv4(out_conv3)) # 512, 16, 16
        out_conv5 = self.conv5_1(self.conv5(out_conv4)) # 512, 8, 8
        out_conv6 = self.conv6(out_conv5) # 1024, 4, 4

        flow6 = self.predict_flow6(out_conv6) # 2, 4, 4
        up_flow6 = self.upconvflow6(flow6)  # 2, 8, 8
        out_upconv5 = self.upconv5(out_conv6) # 512, 8, 8
        concat5 = concatenate(out_upconv5, out_conv5, up_flow6) # 1026, 8, 8

        flow5 = self.predict_flow5(concat5) # 2, 8, 8
        up_flow5 = self.upconvflow5(flow5) # 2, 16, 16
        out_upconv4 = self.upconv4(concat5) # 256, 16, 16
        concat4 = concatenate(out_upconv4, out_conv4, up_flow5) # 770, 16, 16

        flow4 = self.predict_flow4(concat4) # 2, 16, 16
        up_flow4 = self.upconvflow4(flow4) # 2, 32, 32
        out_upconv3 = self.upconv3(concat4) # 128, 32, 32
        concat3 = concatenate(out_upconv3, out_conv3, up_flow4) # 386, 32, 32

        flow3 = self.predict_flow3(concat3) # 2, 32, 32
        up_flow3 = self.upconvflow3(flow3) # 2, 64, 64
        out_upconv2 = self.upconv2(concat3) # 64, 64, 64
        concat2 = concatenate(out_upconv2, out_conv2, up_flow3) # 194, 64, 64

        flow2 = self.predict_flow2(concat2) # 2, 64, 64
        up_flow2 = self.upconvflow2(flow2) # 2, 128, 128
        out_upconv1 = self.upconv1(concat2) #32, 128, 128 
        concat1 = concatenate(out_upconv1, out_conv1, up_flow2) # 98, 128, 128

        finalflow = self.predict_flow1(concat1) # 2, 128, 128



        if self.training:
            return finalflow, flow2, flow3, flow4, flow5, flow6
        else:
            return finalflow,

class FlowNetS_simple(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(6, 64, kernel_size=7, stride=1)

        self.conv2 = conv(64, 128, kernel_size=5)

        self.conv3 = conv(128, 256, kernel_size=5)
        # self.conv3_1 = conv(256, 256, stride=1)

        self.conv4 = conv(256, 512)
        # self.conv4_1 = conv(512, 512, stride=1)

        # self.conv5 = conv(512, 512)
        # self.conv5_1 = conv(512, 512, stride=1)


        # self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
        self.predict_flow4 = predict_flow(512)  # upconv4 + 2 + conv4_1
        self.predict_flow3 = predict_flow(514)  # upconv3 + 2 + conv3_1
        self.predict_flow2 = predict_flow(258)  # upconv2 + 2 + conv2
        self.predict_flow1 = predict_flow(130)    # upconv1 + 2 + conv1
        # self.predict_flow0 = predict_flow()    #upconv0 + 2 + 

        # self.upconv5 = upconv(1024, 512)
        self.upconv4 = upconv(512, 256)
        self.upconv3 = upconv(514, 128)
        self.upconv2 = upconv(258, 64)
        self.upconv1 = upconv(130, 32)


        # self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):

        out_conv1 = self.conv1(x) #(64, 64, 64)
        out_conv2 = self.conv2(out_conv1) # (128, 32, 32)
        out_conv3 = self.conv3(out_conv2) # (256, 16, 16)
        out_conv4 = self.conv4(out_conv3) # (512, 8, 8)


        # flow6 = self.predict_flow6(out_conv6)
        # up_flow6 = self.upconvflow6(flow6)
        # out_upconv5 = self.upconv5(out_conv6)
        # concat5 = concatenate(out_upconv5, out_conv5, up_flow6)

        # flow5 = self.predict_flow5(concat5)
        # up_flow5 = self.upconvflow5(flow5)
        # out_upconv4 = self.upconv4(concat5)
        # concat4 = concatenate(out_upconv4, out_conv4, up_flow5)

        flow4 = self.predict_flow4(out_conv4) #（2，8，8）
        up_flow4 = self.upconvflow4(flow4)  #（2，16，16）
        out_upconv4 = self.upconv4(out_conv4) # (256, 16, 16)

        concat3 = concatenate(out_upconv4, out_conv3, up_flow4) #(256 + 256 + 2, 16, 16)
        print(concat3.shape)

        flow3 = self.predict_flow3(concat3) # (2, 16, 16)
        up_flow3 = self.upconvflow3(flow3) # (2, 32, 32)
        out_upconv3 = self.upconv3(concat3) # (128, 32, 32)

        concat2 = concatenate(out_upconv3, out_conv2, up_flow3) # (128 + 128 + 2, 32, 32)
        print(concat2.shape)

        flow2 = self.predict_flow2(concat2) # (2, 32, 32)
        up_flow2 = self.upconvflow2(flow2) # (2, 64, 64)
        out_upconv2 = self.upconv2(concat2) # (64, 64, 64)

        concat1 = concatenate(out_upconv2, out_conv1, up_flow2) # (64 + 64 + 2, 64, 64)
        print(concat1.shape)


        flow1 = self.predict_flow1(concat1) # (2, 64, 64)

        

        print(finalflow.shape)



        # if self.training:
        #     return finalflow, flow3, flow4, flow5, flow6
        # else:
        #     return finalflow,

class Unsupervised(nn.Module):
    def __init__(self):
        super(Unsupervised, self).__init__()

        self.predictor = FlowNetS()
    
    # def warp(self, flow, img):
    #     b, _, h, w = flow.shape
    #     img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=True)
    #     flow = torch.transpose(flow, 1, 2)
    #     flow = torch.transpose(flow, 2, 3)
    #     # height, width, _ = img_current.shape
    #     warped_img = torch.zeros_like(img)

    #     for n in range(b):
    #         for y in range(h):
    #             for x in range(w):
    #                 # Backward advection
    #                 u, v = flow[n, y, x]
    #                 x_prev = int(x - u)
    #                 y_prev = int(y - v)

    #                 # Bilinear interpolation
    #                 if 0 <= x_prev < w - 1 and 0 <= y_prev < h - 1:
    #                     alpha = x - int(x)
    #                     beta = y - int(y)

    #                     intensity = (1 - alpha) * (1 - beta) * img[n, :, y_prev, x_prev] + \
    #                                 alpha * (1 - beta) * img[n, :, y_prev, x_prev + 1] + \
    #                                 (1 - alpha) * beta * img[n, :, y_prev + 1, x_prev] + \
    #                                 alpha * beta * img[n, :, y_prev + 1, x_prev + 1]

    #                     # Assign the interpolated intensity to the warped image
    #                     warped_img[n, :, y, x] = intensity
    #     return warped_img

    def stn(self, flow, frame):
        b, _, h, w = flow.shape
        frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=True)
        flow = torch.transpose(flow, 1, 2)
        flow = torch.transpose(flow, 2, 3)

        grid = flow + generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1
        warped_frame = F.grid_sample(frame, grid, align_corners=True)

        return warped_frame
    
    def forward(self, x):

        flow_predictions = self.predictor(x)
        frame2 = x[:, 3:, :, :]
        warped_images = [self.stn(flow, frame2) for flow in flow_predictions]

        return flow_predictions, warped_images

## Dataset
def Load_SST(file):
    dataset = []
    rn1,rm1,rn2,rm2 = (128,128, 384, 640)

    data = Dataset(file)
    T = data.variables['thetao'][:]
    _,_,n,m = T.shape
    T = T.reshape((n,m)).filled(T.min())
    T = np.flipud(T)
    T = T[rn1:rn2,rm1:rm2]
    for i in range(4):
        for j in range(8):
            sub_region = T[i*64:(i+1)*64,j*64:(j+1)*64]
            dataset.append(sub_region)
    return dataset

def arrange_data(files_path):
    files = sorted(glob.glob(files_path+'/*.nc'))
    dataset = []
    dataset_new = []
    for i in range(0, len(files), 2):
        data_1 = Load_SST(files[i])
        data_2 = Load_SST(files[i + 1]) if i + 1 < len(files) else None
        if data_2 != None:
            for sub_data1, sub_data2 in zip(data_1, data_2):
                sub_data1 = np.expand_dims(sub_data1, axis=0)
                sub_data2 = np.expand_dims(sub_data2, axis=0)
                data_1_RGB = np.vstack((sub_data1, sub_data1, sub_data1))
                data_2_RGB = np.vstack((sub_data2, sub_data2, sub_data2))
                data = np.concatenate((data_1_RGB, data_2_RGB), axis=0)
                dataset.append(data)
    print(np.array(dataset).shape)
    mean = np.mean(np.array(dataset))
    std = np.std(np.array(dataset))
    for data in dataset:
        data = (data - mean) / std
        dataset_new.append(data)
    
    return dataset_new

class NATL_Dataset(data.Dataset):

    def __init__(self, data_path):
        data = np.load(data_path)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # sample = {
        #     'input': torch.tensor(self.data[index], dtype=torch.float32)
        # }
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        return sample

## Loss     
def charbonnier(x, alpha=0.25, epsilon=1.e-9):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)
    # return torch.pow(x, 2)

def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.sum(s_loss)/b

def photometric_loss(wraped, frame1):
    h, w = wraped.shape[2:]
    frame1 = F.interpolate(frame1, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(wraped - frame1)
    p_loss = torch.sum(p_loss, dim=1)/3
    return torch.sum(p_loss)/frame1.size(0)

def unsup_loss(pred_flows, wraped_imgs, frame1, weights=(1,)): #(0.005, 0.01, 0.02, 0.08, 0.32) 
    # if len(pred_flows) < 5:
    #     weights = [0.005]*len(pred_flows)
    bce = 0
    smooth = 0
    for i in range(len(weights)):
        bce += weights[i] * photometric_loss(wraped_imgs[i], frame1)
        # print(weights[i], pred_flows[i].shape)
        smooth += weights[i] * smoothness_loss(pred_flows[i])

    loss = bce + 0.5 * smooth
    return loss, bce, smooth

## Training
class AverageMeter(object):

    def __init__(self, keep_all=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_smooth_loss = AverageMeter()
    avg_bce_loss = AverageMeter()

    tic = time.time()
    for i, imgs in enumerate(data):
        imgs = imgs.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred_flows, wraped_imgs = model(imgs)
            loss, bce_loss, smooth_loss = criterion(pred_flows, wraped_imgs, imgs[:, :3, :, :])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time = time.time() - tic
        tic = time.time()
        avg_bce_loss.update(bce_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
                  'bce_loss {bce.val:5.4f} ({bce.avg:5.4f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                smooth=avg_smooth_loss, bce=avg_bce_loss))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg bce_loss {bce.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, bce=avg_bce_loss))

    return avg_smooth_loss.avg, avg_bce_loss.avg, avg_loss.avg


if __name__ == '__main__':

    # dataset = arrange_data('/users/Etu9/21205099/CMEMS_DATA')
    # np.save('NATL_Data.npy', dataset)

    # model = Unsupervised().to('cpu')
    # summary(model, input_size=(6,128,128), device='cpu')

    lr = 1.6e-7 #0.000016
    batch_size = 16
    model = Unsupervised()
    model.to(device)
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    # save_path = os.path.join('Unsupervised', type(model.predictor).__name__)
    save_path = os.path.join('Unsupervised', formatted_datetime)
    loss_fnc = unsup_loss
    optim = torch.optim.Adam(model.parameters(), lr)

    os.makedirs(os.path.join("Checkpoints", save_path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", save_path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs", save_path), flush_secs=20)

    NATL_dataset = NATL_Dataset('./NATL_Data.npy')
    train_size = int(0.8 * len(NATL_dataset))
    val_size = len(NATL_dataset) - train_size

    train_set, val_set = data.random_split(NATL_dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    for e in range(200):
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        smooth_loss, bce_loss, total_loss = epoch(model, train_loader, loss_fnc, optim)

        smooth_loss_val, bce_loss_val, total_loss_val = epoch(model, val_loader, loss_fnc)

        if e == 0:
            best_loss = total_loss_val
        
        if total_loss_val < best_loss:
            best_loss = total_loss
            print('Saving checkpoints')
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'best_loss': total_loss_val,
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join("Checkpoints", save_path, 'training_state.pt'))
        
        tb.add_scalars('loss', {"train": total_loss, "val": total_loss_val}, e)
        tb.add_scalars('smooth_loss', {"train": smooth_loss, "val": smooth_loss_val}, e)
        tb.add_scalars('bce_loss', {"train": bce_loss, "val": bce_loss_val}, e)
        
        e = e + 1