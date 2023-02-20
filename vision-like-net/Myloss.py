import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k



class L_spa4(nn.Module):

    def __init__(self):
        super(L_spa4, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E




class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3

class L_median(nn.Module):

    def __init__(self):
        super(L_median, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_00= torch.FloatTensor( [[1,0,0],[0,0,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_01 = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_02 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_10 = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_11 = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_12 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_20 = torch.FloatTensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_21 = torch.FloatTensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_22 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_00 = nn.Parameter(data=kernel_00, requires_grad=False)
        self.weight_01 = nn.Parameter(data=kernel_01, requires_grad=False)
        self.weight_02 = nn.Parameter(data=kernel_02, requires_grad=False)
        self.weight_10 = nn.Parameter(data=kernel_10, requires_grad=False)
        self.weight_11 = nn.Parameter(data=kernel_11, requires_grad=False)
        self.weight_12 = nn.Parameter(data=kernel_12, requires_grad=False)
        self.weight_20 = nn.Parameter(data=kernel_20, requires_grad=False)
        self.weight_21 = nn.Parameter(data=kernel_21, requires_grad=False)
        self.weight_22 = nn.Parameter(data=kernel_22, requires_grad=False)


    def forward(self, org  ):


        # org_mean = torch.mean(org,1,keepdim=True)
        org_R = org[:, 0:1, :, :]
        org_G = org[:, 1:2, :, :]
        org_B = org[:, 2:3, :, :]



        D_00_R = F.conv2d(org_R, self.weight_00, padding=1)
        D_01_R = F.conv2d(org_R, self.weight_01, padding=1)
        D_02_R = F.conv2d(org_R, self.weight_02, padding=1)
        D_10_R = F.conv2d(org_R, self.weight_10, padding=1)
        D_11_R = F.conv2d(org_R, self.weight_11, padding=1)
        D_12_R = F.conv2d(org_R, self.weight_12, padding=1)
        D_20_R = F.conv2d(org_R, self.weight_20, padding=1)
        D_21_R = F.conv2d(org_R, self.weight_21, padding=1)
        D_22_R = F.conv2d(org_R, self.weight_22, padding=1)
        D_all_R = torch.cat([D_00_R,D_01_R,D_02_R,D_10_R,D_11_R,D_12_R,D_20_R,D_21_R,D_22_R],dim=1)

        D_median_R = torch.median(D_all_R, 1, True)
        D_median_R = D_median_R[0]

        D_00_G = F.conv2d(org_G, self.weight_00, padding=1)
        D_01_G = F.conv2d(org_G, self.weight_01, padding=1)
        D_02_G = F.conv2d(org_G, self.weight_02, padding=1)
        D_10_G = F.conv2d(org_G, self.weight_10, padding=1)
        D_11_G = F.conv2d(org_G, self.weight_11, padding=1)
        D_12_G = F.conv2d(org_G, self.weight_12, padding=1)
        D_20_G = F.conv2d(org_G, self.weight_20, padding=1)
        D_21_G = F.conv2d(org_G, self.weight_21, padding=1)
        D_22_G = F.conv2d(org_G, self.weight_22, padding=1)
        D_all_G = torch.cat([D_00_G, D_01_G, D_02_G, D_10_G, D_11_G, D_12_G, D_20_G, D_21_G, D_22_G], dim=1)

        D_median_G = torch.median(D_all_G, 1, True)
        D_median_G = D_median_G[0]

        D_00_B = F.conv2d(org_B, self.weight_00, padding=1)
        D_01_B = F.conv2d(org_B, self.weight_01, padding=1)
        D_02_B = F.conv2d(org_B, self.weight_02, padding=1)
        D_10_B = F.conv2d(org_B, self.weight_10, padding=1)
        D_11_B = F.conv2d(org_B, self.weight_11, padding=1)
        D_12_B = F.conv2d(org_B, self.weight_12, padding=1)
        D_20_B = F.conv2d(org_B, self.weight_20, padding=1)
        D_21_B = F.conv2d(org_B, self.weight_21, padding=1)
        D_22_B = F.conv2d(org_B, self.weight_22, padding=1)
        D_all_B = torch.cat([D_00_B, D_01_B, D_02_B, D_10_B, D_11_B, D_12_B, D_20_B, D_21_B, D_22_B], dim=1)

        D_median_B = torch.median(D_all_B, 1, True)
        D_median_B = D_median_B[0]











        E = torch.pow(D_median_R-org_R, 2) + torch.pow(D_median_G-org_G, 2) + torch.pow(D_median_B-org_B, 2)


        return E