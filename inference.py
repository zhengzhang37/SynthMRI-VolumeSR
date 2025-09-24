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
    dstarts = [x for x in range(0, D, crop_size - overlap_size)]
    while dstarts[-1] + crop_size > D:
        dstarts.pop()
    dstarts.append(max(D - crop_size, 0))  


    split_data = []
    starts = []
    for ds in dstarts:
        cimgdata = imgtensor[:, :, ds:ds + crop_size, :, :]  
        starts.append(ds)
        split_data.append(cimgdata)
    return split_data, starts

def get_scoremap_d_only(D, B=1, C=1):
    center_d = D / 2
    score = torch.ones((B, C, D))
    for d in range(D):
        score[:, :, d] = 1.0 / (abs(d - center_d) + 1e-3)  
    return score

def merge_image_d_only(split_data, starts, crop_size=128, resolution=(1, 1, 500, 256, 256)):
    B, C, D, H, W = resolution
    tot_score = torch.zeros((B, C, D, H, W))
    merge_img = torch.zeros((B, C, D, H, W))
    for simg, ds in zip(split_data, starts):
        scoremap = get_scoremap_d_only(crop_size, B, C).unsqueeze(-1).unsqueeze(-1)  
        merge_img[:, :, ds:ds + crop_size, :, :] += scoremap * simg[:,:,:D,:H,:W]
        tot_score[:, :, ds:ds + crop_size, :, :] += scoremap
    merge_img = merge_img / tot_score
    return merge_img

def padding(img):
    d, h, w = img.shape
    padding_ratio_d = d // 16 + 1
    padding_ratio_h = h // 16 + 1
    padding_ratio_w = w // 16 + 1

    result = -np.ones((padding_ratio_d * 16, padding_ratio_h * 16, padding_ratio_w * 16)) 
    result[:d, :h, :w] = img
    return result.astype(np.float32)

def preprocess(img, max_value):
    img = img.astype(np.float32) / max_value.astype(np.float32)
    img = (img - 0.5) * 2.0 
    return img.astype(np.float32)

def inference():
    model = ResNet3D()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # replace file path with your own
    sample_path = '/home/zhangzheng/work/SynthMRI-VolumeSR/test_case'
    matrix = {
        'FLAIR': {'PSNR': 0, 'SSIM': 0}
        }
    # replace file path with your own
    save_path = '../xxx'
    with torch.no_grad():

        # replace file path with your own
        flair_lr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sample_path, '2D-FLAIR.nii.gz')))
        flair_hr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sample_path, '3D-FLAIR.nii.gz')))
        t1_hr_nii_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sample_path, '3D-T1W.nii.gz')))

        max_flair_lr = flair_lr_nii_img.max().astype(np.float32)
        max_flair_hr = flair_hr_nii_img.max().astype(np.float32)
        max_t1_hr = t1_hr_nii_img.max().astype(np.float32)

        flair_lr = padding(preprocess(flair_lr_nii_img, max_flair_lr))
        flair_hr = padding(preprocess(flair_hr_nii_img, max_flair_hr))
        t1_hr = padding(preprocess(t1_hr_nii_img, max_t1_hr))

        flair_lr = torch.from_numpy(flair_lr).unsqueeze(0).unsqueeze(0)
        flair_hr = torch.from_numpy(flair_hr).unsqueeze(0).unsqueeze(0)
        t1_hr = torch.from_numpy(t1_hr).unsqueeze(0).unsqueeze(0)

        input_image, target_image = torch.cat((flair_lr, t1_hr), dim=1), flair_hr

        b, c, d, h, w = target_image.shape
        split_data, starts = split_image_d_only(input_image)
        for i, data in enumerate(split_data):
            split_data[i] = model(data).cpu()
        output = merge_image_d_only(split_data, starts, resolution=(b, c, d, h, w))
        output = output.detach().cpu().numpy()
        output[output<-1.0] = -1.0
        output[output>1.0] = 1.0
        target_image = target_image.detach().cpu().numpy()

        # output = sitk.GetImageFromArray(np.array(output[0, 0,:d,:h,:w] * 0.5 + 0.5) * max_flair_lr.numpy())
        # sitk.WriteImage(output, os.path.join(save_path, input_file))
        matrix['FLAIR']['PSNR'] = compare_psnr(output[0,0,:d,:h,:w], target_image[0][0], data_range=2.)
        matrix['FLAIR']['SSIM'] = compare_ssim(output[0,0,:d,:h,:w], target_image[0][0], data_range=2.)

        print('FLAIR SSIM: {:.4f}, PSNR: {:.4f}'.format(matrix['FLAIR']['SSIM'], matrix['FLAIR']['PSNR']))


if __name__ == '__main__':
    inference()
