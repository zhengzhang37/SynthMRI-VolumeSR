import os
import os.path as osp
import pydicom
import numpy as np
import SimpleITK as sitk
import ants
import shutil
import random

# dcm to nii
def dcm2nii(dicom_dir, save_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dicom_dir)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, seriesIDs[i])
        lens[i] = len(dicom_names)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)

    img = reader.Execute()

    img_array = sitk.GetArrayFromImage(img)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    res = sitk.GetImageFromArray(img_array)
    res.SetSpacing(spacing)
    res.SetDirection(direction)
    res.SetOrigin(origin)
    sitk.WriteImage(res, save_dir)

root_path = '../root_path'
save_path = '../save_path'
tabs = ['3D-T1']
registration_path = '../shly_reg_nif'




for patient_path in os.listdir(root_path):
    patient = osp.join(root_path, patient_path)
    if not osp.exists(osp.join(save_path, patient_path)):
        os.makedirs(osp.join(save_path, patient_path))
    for tab in os.listdir(patient):
        dcm2nii(osp.join(root_path, patient_path, tab), osp.join(save_path, patient_path, tab + '.nii.gz'))

for patient_path in os.listdir(save_path):
    if not osp.exists(osp.join(registration_path, patient_path)):
        os.makedirs(osp.join(registration_path, patient_path))
    if os.path.exists(os.path.join(save_path, patient_path, '3D-T1.nii.gz')):
        target_path = os.path.join(save_path, patient_path, '3D-T1.nii.gz')
        target_img = ants.image_read(target_path)
        ants.image_write(target_img, os.path.join(registration_path, patient_path, '3D-T1.nii.gz'))
        for tab in os.listdir(os.path.join(save_path, patient_path)):
            if tab != '3D-T1.nii.gz':
                source_path = os.path.join(save_path, patient_path, tab)
                save_tab_path = os.path.join(registration_path, patient_path, tab)
                fix_img = ants.image_read(target_path)
                move_img = ants.image_read(source_path)
                outs = ants.registration(fix_img, 
                                        move_img, 
                                        type_of_transform = 'Affine',
                                        grad_step=0.1,
                                        flow_sigma=3,
                                        total_sigma=0,
                                        aff_metric='mattes',
                                        aff_sampling=32,
                                        syn_metric='mattes',
                                        syn_sampling=32,
                                        reg_iterations=(40, 20, 0),
                                        write_composite_transform=False,
                                        verbose=False,
                                        multivariate_extras=None,)
                reg_img = outs['warpedmovout']  
                ants.image_write(reg_img, save_tab_path)
