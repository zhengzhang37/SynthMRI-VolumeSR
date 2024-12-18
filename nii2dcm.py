import os
import copy
import pydicom
import numpy as np
import SimpleITK as sitk
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.pixel_data_handlers.util import get_expected_length
from pydicom.uid import ImplicitVRLittleEndian

def nii2dcm(nii_img_path, version, root_path, save_path):
    series_uid = pydicom.uid.generate_uid()
    uid_pool = set()
    uid_pool.add(series_uid)
    nii_img = sitk.GetArrayFromImage(sitk.ReadImage(nii_img_path)).astype(np.int16)
    print(f"Number of slices: {len(nii_img)}")
    for i in range(len(nii_img)):
        dcm_path = os.path.join(root_path, f'{i+1:04d}.dcm')
        save_dcm_path = os.path.join(save_path, f'{i:03d}.dcm')

        ds = pydicom.dcmread(dcm_path)
        output = copy.deepcopy(ds)
        data = np.asarray(output.pixel_array)
        data = data.astype(np.int16)
        data = nii_img[i]

        series_number_offset = int(float(ds.SeriesNumber) + 100)
        sop_uid = pydicom.uid.generate_uid()
        while sop_uid in uid_pool:
            sop_uid = pydicom.uid.generate_uid()
        uid_pool.add(sop_uid)

        bits_stored = output.get("BitsStored", 16)
        if output.get("PixelRepresentation", 0) != 0:
            t_min, t_max = (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1)
        else:
            t_min, t_max = 0, (1 << bits_stored) - 1
        data[data < t_min] = t_min
        data[data > t_max] = t_max

        output.PixelData = data.tobytes()
        if (
            output.file_meta.TransferSyntaxUID.is_compressed  
            and get_expected_length(output) == len(output.PixelData)
        ):
            output.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            output.is_implicit_VR = True
        
        series_number_offset = int(float(ds.SeriesNumber) + 100)
        output.SeriesInstanceUID = series_uid
        output.SOPInstanceUID = sop_uid
        output.SeriesNumber = int(float(output.SeriesNumber)) + series_number_offset
        output.SeriesDescription = version
        
        output.save_as(save_dcm_path)
