import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from testing import MONAI
import os
import torch
from glob import glob
import nibabel as nib

device = torch.device('cpu')

st.title('Medical Image Segmentation',)
st.header('Segmentation of Hippocampus')
st.write("""Nifti is a medical images format, to store both images, and companied data, the images are usually in grayscale, and 
         they are taken as slices, each slice with a different cross-section of the body.""")
uploaded_file = st.file_uploader("Choose an image", ["nii.gz",'jpg'])
st.write('Or')
use_default_images = st.checkbox('Use default Nifti Image')

if use_default_images:
    nifti_no = st.selectbox('choose any one', (1,2,3,4,5,6,7,8))
    st.write('You selected nifti file number:',nifti_no)
    st.write('There are 64 slices in this nifti file')
    slice_no = st.slider('choose the slice number', min_value=1, max_value=64)
    st.write('You selected a slice number:',slice_no)
    in_dir = "Task04_Hippocampus"

    test_images = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(test_images, test_segmentation)]
    test_files = test_files[nifti_no:10]

    instance = MONAI('best_metric_model2.pth', device, test_files, slice_no)

    orignal_image = instance.ViewImage()
    output = instance.Inference()
    images = [orignal_image, output]

    col1, col2 = st.columns(2)

    col1.image(images[0], width=300, caption='Image')
    col2.image(images[1], width=300, caption='Output')

elif uploaded_file is not None:
    # import pdb;pdb.set_trace()
    nifti_img = nib.load(uploaded_file)
    img_data = nifti_img.get_fdata()
    st.write(img_data)

# elif uploaded_file is not None:
#     file = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(uploaded_file, uploaded_file)]
#     import pdb;pdb.set_trace()
#     instance2 = MONAI('best_metric_model2.pth', device, file)
    
#     uploded_image = instance2.ViewImage()
    
#     uploded_image_output = instance2.Inference()
#     u_images = [uploded_image, uploded_image_output]

#     col1, col2 = st.columns(2)

#     col1.image(u_images[0], width=300, caption='Image')
#     col2.image(u_images[1], width=300, caption='Output')

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


    # st.image(images, width = 300)
    # st.image(instance.Inference(), width=300)

    # images = [MONAI('best_metric_model2.pth'), MONAI('best_metric_model2.pth')]
    # st.image(images, width=300, caption = ['orignal','output'])
    # st.image(MONAI('best_metric_model2.pth'), channels="GRAY", width = 300)
            