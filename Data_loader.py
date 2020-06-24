import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

HGG_path = "C:/Data/BraTS18Train/HGG"
# bratslgg_path = "D:\Data\\brats18\LGG"
flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

# img = sitk.ReadImage()
# imgArray = sitk.GetArrayFromImage(img)

#Spacing [default is one]: Distance between adjacent pixels/voxels in each dimension given in physical units.

def getImageSizeandSpacing():
    pathhgg_list = file_name_path(HGG_path)
    for subsetindex in range(len(pathhgg_list)):
        brats_subset_path = HGG_path + "/" + str(pathhgg_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        print('subsetindex:', subsetindex)
        print("flair_src image size,spacing:", (flair_src.GetSize(), flair_src.GetSpacing()))
        print("t1_src image size, spacing:", (t1_src.GetSize(), t1_src.GetSpacing()))
        print("t1ce_src image size, spacing:", (t1ce_src.GetSize(), t1ce_src.GetSpacing()))
        print("t2_src image size, spacing:", (t2_src.GetSize(), t2_src.GetSpacing()))


def file_name_path(file_dir, dir=True, file=False):
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs

getImageSizeandSpacing()


