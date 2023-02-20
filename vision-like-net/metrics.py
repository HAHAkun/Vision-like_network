import torch
from torchvision import transforms
from PIL import Image
import os

from utils import AverageMeter
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

# test_data = MyData
# Model = CSRNet9().cuda()
epoch_ssim = AverageMeter()
epoch_mse = AverageMeter()
epoch_psrn = AverageMeter()
hr_path = 'data/eval15/high'
predict_path = 'data/result_gradient/low'
hr_img_path = os.listdir(hr_path)
predict_img_path = os.listdir(predict_path)
hr_img_path.sort()
predict_img_path.sort()

for idx in range(len(hr_img_path)):
    predict_img_name = predict_img_path[idx]
    hr_img_name = hr_img_path[idx]
    predict_img_item_path = os.path.join(predict_path, predict_img_name)
    hr_img_item_path = os.path.join(hr_path, hr_img_name)
    predict_img = Image.open(predict_img_item_path).convert('RGB')
    tensor = transforms.ToTensor()
    ts_img = tensor(predict_img)
    array_predict_img = np.array(predict_img)
    hr_img = Image.open(hr_img_item_path).convert('RGB')
    ts_hr = tensor(hr_img)
    array_hr_img = np.array(hr_img)
    # h = (array_hr_img.shape[0] // 12) * 12
    # w = (array_hr_img.shape[1] // 12) * 12
    # array_hr_img = array_hr_img[0:h, 0:w, :]
    epoch_psrn.update(psnr(array_predict_img, array_hr_img), 1)
    epoch_ssim.update(ssim(array_predict_img, array_hr_img, multichannel=True,win_size=11),1)
    epoch_mse.update(mse(array_predict_img, array_hr_img),1)
print("ssim={}".format(epoch_ssim.avg))
print('mse={}'.format(epoch_mse.avg))
print('psnr={}'.format(epoch_psrn.avg))
