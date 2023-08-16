import nibabel as nib
import matplotlib.pyplot as plt

# Change the path to your path
my_img = '/home/lnv-68/Work/research/streamlit/monai/Task04_Hippocampus/TestSegmentation/hippocampus_001.nii.gz'
Nifti_img  = nib.load(my_img)
nii_data = Nifti_img.get_fdata()
nii_aff  = Nifti_img.affine
nii_hdr  = Nifti_img.header
print(nii_aff ,'\n',nii_hdr)
print(nii_data.shape)
if(len(nii_data.shape)==3):
   for slice_Number in range(nii_data.shape[2]):
       plt.imshow(nii_data[:,:,slice_Number ])
       plt.show()
if(len(nii_data.shape)==4):
   for frame in range(nii_data.shape[3]):
       for slice_Number in range(nii_data.shape[2]):
           plt.imshow(nii_data[:,:,slice_Number,frame])
           plt.show()