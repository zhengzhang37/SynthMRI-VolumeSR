# SynthMRI-VolumeSR

This project aims to achieve multi-contrast volume super-resolution for LR FLAIR and HR TOF to generate HR-like FLAIR volume.
![image](https://github.com/zhengzhang37/SynthMRI-VolumeSR/blob/main/results.jpg)

## Data Preparation

### Acquired Information
478 subjects with acquired 2D FLAIR, HR 3D FLAIR and 3D TOF images
![image](https://github.com/zhengzhang37/SynthMRI-VolumeSR/blob/main/contrast.jpg)

3D TOF images were used as the reference to provide structural details and align the number of slices from other contrast 3D scans during cross-contrast registration. During data pre-processing, 2D images were linearly interpolated to match the slice number as its corresponding 3D acquisition but still in LR, while the 3D acquisitions served as HR targets.

### Data Splitting
![image](https://github.com/zhengzhang37/SynthMRI-VolumeSR/blob/main/split.jpg)

## Train

## Inference
Inference code of multi-contrast volume super-resolution for LR FLAIR + HR TOF -> HR FLAIR
```
python inference.py
```

## Transform Nii volume to Dicom Files

```
nii2dcm(nii_image_path='your_nii_image', version='save_name', root_path='reference_dcm_file_path', save_path='your_save_path')
```

## Pre-trained model
The model can be downloaded [here](https://drive.google.com/file/d/1MIv3F7bpDnw27ya-pgcFKTTWX9PX23xs/view?usp=sharing).
