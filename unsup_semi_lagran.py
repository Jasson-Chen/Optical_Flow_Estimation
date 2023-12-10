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

PRINT_INTERVAL = 20
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

class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(6, 64, kernel_size=7)
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

        self.upconv5 = upconv(1024, 512)
        self.upconv4 = upconv(1026, 256)
        self.upconv3 = upconv(770, 128)
        self.upconv2 = upconv(386, 64)

        self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)

        flow6 = self.predict_flow6(out_conv6)
        up_flow6 = self.upconvflow6(flow6)
        out_upconv5 = self.upconv5(out_conv6)
        concat5 = concatenate(out_upconv5, out_conv5, up_flow6)

        flow5 = self.predict_flow5(concat5)
        up_flow5 = self.upconvflow5(flow5)
        out_upconv4 = self.upconv4(concat5)
        concat4 = concatenate(out_upconv4, out_conv4, up_flow5)

        flow4 = self.predict_flow4(concat4)
        up_flow4 = self.upconvflow4(flow4)
        out_upconv3 = self.upconv3(concat4)
        concat3 = concatenate(out_upconv3, out_conv3, up_flow4)

        flow3 = self.predict_flow3(concat3)
        up_flow3 = self.upconvflow3(flow3)
        out_upconv2 = self.upconv2(concat3)
        concat2 = concatenate(out_upconv2, out_conv2, up_flow3)

        finalflow = self.predict_flow2(concat2)

        if self.training:
            return finalflow, flow3, flow4, flow5, flow6
        else:
            return finalflow,

class Unsupervised(nn.Module):
    def __init__(self):
        super(Unsupervised, self).__init__()

        self.predictor = FlowNetS()
    
    def warp(self, flow, img):
        b, _, h, w = flow.shape
        img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=True)
        flow = torch.transpose(flow, 1, 2)
        flow = torch.transpose(flow, 2, 3)
        # height, width, _ = img_current.shape
        warped_img = torch.zeros_like(img)

        for n in range(b):
            for y in range(h):
                for x in range(w):
                    # Backward advection
                    u, v = flow[n, y, x]
                    x_prev = int(x - u)
                    y_prev = int(y - v)

                    # Bilinear interpolation
                    if 0 <= x_prev < w - 1 and 0 <= y_prev < h - 1:
                        alpha = x - int(x)
                        beta = y - int(y)

                        intensity = (1 - alpha) * (1 - beta) * img[n, :, y_prev, x_prev] + \
                                    alpha * (1 - beta) * img[n, :, y_prev, x_prev + 1] + \
                                    (1 - alpha) * beta * img[n, :, y_prev + 1, x_prev] + \
                                    alpha * beta * img[n, :, y_prev + 1, x_prev + 1]

                        # Assign the interpolated intensity to the warped image
                        warped_img[n, :, y, x] = intensity
        return warped_img

    def forward(self, x):

        flow_predictions = self.predictor(x)
        frame2 = x[:, 3:, :, :]
        warped_images = [self.warp(flow, frame2) for flow in flow_predictions]

        return flow_predictions, warped_images

## Dataset
def Load_SST(file):
    rn1,rm1,rn2,rm2 = (99,269, 99+240, 269+240)

    data = Dataset(file)
    T = data.variables['thetao'][:]
    _,_,n,m = T.shape
    T = T.reshape((n,m)).filled(T.min())
    T = np.flipud(T)
    T = T[rn1:rn2,rm1:rm2]
    return T

def arrange_data(files_path):
    files = sorted(glob.glob(files_path+'/*.nc'))
    dataset = []
    for i in range(0, len(files), 2):
        data_1 = Load_SST(files[i])
        data_2 = Load_SST(files[i + 1]) if i + 1 < len(files) else None
        if data_2.all() != None:
            data_1 = np.expand_dims(data_1, axis=0)
            data_2 = np.expand_dims(data_2, axis=0)
            data_1_RGB = np.vstack((data_1, data_1, data_1))
            data_2_RGB = np.vstack((data_2, data_2, data_2))
            data = np.concatenate((data_1_RGB, data_2_RGB), axis=0)
            dataset.append(data)
    return dataset

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

def unsup_loss(pred_flows, wraped_imgs, frame1, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
    if len(pred_flows) < 5:
        weights = [0.005]*len(pred_flows)
    bce = 0
    smooth = 0
    for i in range(len(weights)):
        bce += weights[i] * photometric_loss(wraped_imgs[i], frame1)
        smooth += weights[i] * smoothness_loss(pred_flows[i])

    loss = bce + smooth
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
    lr = 1.6e-3 #0.000016
    batch_size = 4
    model = Unsupervised()
    model.to(device)
    save_path = os.path.join('Unsupervised', type(model.predictor).__name__)
    loss_fnc = unsup_loss
    optim = torch.optim.Adam(model.parameters(), lr)

    os.makedirs(os.path.join("Checkpoints", save_path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", save_path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs", save_path), flush_secs=20)

    NATL_dataset = NATL_Dataset('./NATL_Data.npy')
    data_loader = data.DataLoader(NATL_dataset, batch_size=4, shuffle=True)
    for e in range(200):
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        smooth_loss, bce_loss, total_loss = epoch(model, data_loader, loss_fnc, optim)
        if e == 0:
            best_loss = total_loss
        
        if total_loss < best_loss:
            best_loss = total_loss
            print('Saving checkpoints')
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'best_loss': total_loss,
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join("Checkpoints", save_path, 'training_state.pt'))
        
        tb.add_scalars('loss', {"train": total_loss}, e)
        tb.add_scalars('smooth_loss', {"train": smooth_loss}, e)
        tb.add_scalars('bce_loss', {"train": bce_loss}, e)
        
        e = e + 1