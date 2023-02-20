import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device('cuda:0')
class medium_net(nn.Module):

    def __init__(self):
        super(medium_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 8
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)

        # self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.AdaptiveMaxPool2d(1)



        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cond_shift1 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift2 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift3 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift4 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift5 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift6 = nn.Conv2d(2 * number_f, 3, kernel_size=1, bias=True)
        self.cond_shift7 = nn.Conv2d(48, 3, kernel_size=1, bias=True)



    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(x4))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(x5))

        x7 = F.tanh(self.e_conv7(x6))


        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.split(x7, 3, dim=1)



        x1_max = self.maxpool(x1)
        x1_avg = self.avgpool(x1)
        s1 = torch.cat([x1_max, x1_avg], dim=1)
        s1 = self.cond_shift1(s1)
        r_7 = F.tanh(s1 * r_7)

        x2_max = self.maxpool(x2)
        x2_avg = self.avgpool(x2)
        s2 = torch.cat([x2_max, x2_avg], dim=1)
        s2 = self.cond_shift2(s2)
        r_6 = F.tanh(s2 * r_6)

        x3_max = self.maxpool(x3)
        x3_avg = self.avgpool(x3)
        s3 = torch.cat([x3_max, x3_avg], dim=1)
        s3 = self.cond_shift3(s3)
        r_5 = F.tanh(s3 * r_5)

        x4_max = self.maxpool(x4)
        x4_avg = self.avgpool(x4)
        s4 = torch.cat([x4_max, x4_avg], dim=1)
        s4 = self.cond_shift4(s4)
        r_4 = F.tanh(s4 * r_4)

        x5_max = self.maxpool(x5)
        x5_avg = self.avgpool(x5)
        s5 = torch.cat([x5_max, x5_avg], dim=1)
        s5 = self.cond_shift5(s5)
        r_3 = F.tanh(s5 * r_3)

        x6_max = self.maxpool(x6)
        x6_avg = self.avgpool(x6)
        s6 = torch.cat([x6_max, x6_avg], dim=1)
        s6 = self.cond_shift6(s6)
        r_2 = F.tanh(s6 * r_2)

        x7_max = self.maxpool(x7)
        x7_avg = self.avgpool(x7)
        s7 = torch.cat([x7_max, x7_avg], dim=1)
        s7 = self.cond_shift7(s7)
        r_1 = F.tanh(s7 * r_1)

        x = x + r_1 * (torch.pow(x, 2) - x)
        enhance_1 = x + r_2 * (torch.pow(x, 2) - x)
        x = enhance_1 + r_3 * (torch.pow(enhance_1, 2) - enhance_1)
        x = x + r_4 * (torch.pow(x, 2) - x)
        enhance_2 = x + r_5 * (torch.pow(x, 2) - x)
        x = x + r_6 * (torch.pow(enhance_2, 2) - enhance_2)
        x = x + r_7 * (torch.pow(x, 2) - x)
        enhance_image = x + r_8 * (torch.pow(x, 2) - x)
        r = torch.cat([r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8], 1)
        return enhance_1, enhance_2, enhance_image, r







if __name__ == "__main__":
    device = torch.device('cuda:0')
    img = Image.open('data/test_data/LIME/1.bmp')
    img=img.resize((512,512))
    transf = transforms.ToTensor()

    img_tensor = transf(img).unsqueeze(0).to(device)
    ts = torch.ones([1,64*3,512,512]).to(device)
    # print(img_tensor.size())
    net = medium_net()

    net = net.to(device)

    # enhance_image_1, enhance_image, r, enhance_image_2 = net(img_tensor)
    enhance_1, enhance_2, enhance_image, r = net(img_tensor)
    # print(img_tensor.size())
    print(enhance_image.size())