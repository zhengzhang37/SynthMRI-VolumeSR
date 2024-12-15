#!/usr/bin/python3
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch
from resnet3d import *
import SimpleITK as sitk


def split_image_d_only(imgtensor, crop_size=128, overlap_size=10):
    _, C, D, H, W = imgtensor.shape
    # 计算在D维上的切割起始位置
    dstarts = [x for x in range(0, D, crop_size - overlap_size)]
    while dstarts[-1] + crop_size > D:
        dstarts.pop()
    dstarts.append(max(D - crop_size, 0))  # 确保至少有一个完整的切片

    # 切割图像，并存储每个切片及其在D维上的起始位置
    split_data = []
    starts = []
    for ds in dstarts:
        cimgdata = imgtensor[:, :, ds:ds + crop_size, :, :]  # 切割操作
        starts.append(ds)
        split_data.append(cimgdata)
    return split_data, starts

def get_scoremap_d_only(D, B=1, C=1):
    center_d = D / 2
    score = torch.ones((B, C, D))
    for d in range(D):
        score[:, :, d] = 1.0 / (abs(d - center_d) + 1e-3)  # 使用距离中心的倒数作为权重，避免除零错误
    return score

def merge_image_d_only(split_data, starts, crop_size=128, resolution=(1, 1, 500, 256, 256)):
    B, C, D, H, W = resolution
    tot_score = torch.zeros((B, C, D, H, W))
    merge_img = torch.zeros((B, C, D, H, W))
    for simg, ds in zip(split_data, starts):
        scoremap = get_scoremap_d_only(crop_size, B, C).unsqueeze(-1).unsqueeze(-1)  # 调整形状以匹配H和W维
        merge_img[:, :, ds:ds + crop_size, :, :] += scoremap * simg[:,:,:D,:H,:W]
        tot_score[:, :, ds:ds + crop_size, :, :] += scoremap
    merge_img = merge_img / tot_score
    return merge_img

def preprocess(img, max_value):
    img = img.astype(np.float32) / max_value.astype(np.float32)
    img = (img - 0.5) * 2.0 
    return img.astype(np.float32)

def inference(opt, epoch=None):
    model = ResnetGenerator3DwithSpectralNorm().cuda()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # replace file path with your own
    input_list = ['xx.nii.gz', 'yy.nii.gz']
    matrix = {
        'FLAIR': {'PSNR': 0, 'SSIM': 0}
        }
    # replace file path with your own
    save_path = '.../save_path'
    with torch.no_grad():
        for input_file in input_list:

            # replace file path with your own
            flair_lr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('.../LR-FLAIR', input_file)))
            flair_hr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('.../HR-FLAIR', input_file)))
            tof_hr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('.../HR-TOF', input_file)))

            max_flair_lr = flair_lr_nii_img.max().astype(np.float32)
            max_flair_hr = flair_hr_nii_img.max().astype(np.float32)
            max_tof_hr = tof_hr_nii_img.max().astype(np.float32)

            flair_lr = preprocess(flair_lr_nii_img, max_flair_lr)
            flair_hr = preprocess(flair_hr_nii_img, max_flair_hr)
            tof_hr = preprocess(tof_hr_nii_img, max_tof_hr)

            flair_lr = torch.from_numpy(flair_lr).unsqueeze(0).unsqueeze(0).cuda()
            flair_hr = torch.from_numpy(flair_hr).unsqueeze(0).unsqueeze(0).cuda()
            tof_hr = torch.from_numpy(tof_hr).unsqueeze(0).unsqueeze(0).cuda()

            input_image, target_image = torch.cat((flair_lr, tof_hr), dim=1), tof_hr

            output = model(input_image)

            b, c, d, h, w = target_image.shape
            split_data, starts = split_image_d_only(input_image)
            for i, data in enumerate(split_data):
                split_data[i] = model(input_image).cpu()
            output = merge_image_d_only(split_data, starts, resolution=(b, c, d, h, w))
            output = output.detach().cpu().numpy()
            output[output<-1.0] = -1.0
            output[output>1.0] = 1.0

            output = sitk.GetImageFromArray(np.array(output[0, 0,:d,:h,:w] * 0.5 + 0.5) * max_flair_lr.numpy())
            sitk.WriteImage(output, os.path.join(save_path, input_file))
            matrix['FLAIR']['PSNR'] += compare_psnr(output[0,0,:d,:h,:w], target_image[0][0], data_range=2.) / len(input_list)
            matrix['FLAIR']['SSIM'] += compare_ssim(output[0,0,:d,:h,:w], target_image[0][0], data_range=2.) / len(input_list)

    print('FLAIR SSIM: {:.4f}, PSNR: {:.4f}'.format(matrix['FLAIR']['SSIM'], matrix['FLAIR']['PSNR']))