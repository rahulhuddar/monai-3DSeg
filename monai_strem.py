import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from testing import MONAI
import os
import torch
from glob import glob

device = torch.device('cpu')
in_dir = "Task04_Hippocampus"

test_images = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))
test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(test_images, test_segmentation)]
# import pdb;pdb.set_trace()
test_files = test_files[2:5]

st.title('Medical Image Classification')
uploaded_file = st.file_uploader("Choose an image", ["jpg","jpeg","png","nii.gz"]) #image uploader
st.write('Or')
use_default_image = st.checkbox('Use default Image')

if use_default_image:
    # images = [MONAI('best_metric_model2.pth'), MONAI('best_metric_model2.pth')]
    # st.image(images, width=300, caption = ['orignal','output'])
    # st.image(MONAI('best_metric_model2.pth'), channels="GRAY", width = 300)
    st.image(MONAI('best_metric_model2.pth', device, test_files), channels='GRAY', width=300)

# elif uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)
#     if opencv_image is not None:
#         image = st.image(opencv_image, channels="BGR", width=800)
#     else:
#         image = st.image(default_image, channels="BGR", width=800)
#     if st.button('Solve'):
#         with st.spinner('Solving your maze'):
#             results = model(opencv_image)
#             st.write(results[0].names[results[0].probs.top1])
            