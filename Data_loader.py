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
