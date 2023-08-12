from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
from PIL import Image

import os
from glob import glob
import numpy as np

from monai.inferers import sliding_window_inference

class MONAI:
    def __init__(self, weights, device, test_files):
        self.weights = weights
        self.device = device
        self.test_files = test_files

    def ViewImage(self):
        test_transforms = Compose(
                [
                    LoadImaged(keys=["vol", "seg"]),
                    AddChanneld(keys=["vol", "seg"]),
                    Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")), 
                    Orientationd(keys=["vol", "seg"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["vol"], a_min=0, a_max=1800,b_min=0.0, b_max=1.0, clip=True), 
                    CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
                    Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),   
                    ToTensord(keys=["vol", "seg"]),
                ]
            )
    
        test_ds = Dataset(data=self.test_files, transform=test_transforms)
        self.test_loader = DataLoader(test_ds, batch_size=1)
        test_patient = first(self.test_loader)

        return np.asarray(test_patient["vol"][0, 0, :, :, 20])

    def Inference(self):
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

        model.load_state_dict(torch.load(self.weights, map_location=self.device))
        model.eval()

        sw_batch_size = 4
        roi_size = (128, 128, 64)
        with torch.no_grad():
            test_patient = first(self.test_loader)

            t_volume = test_patient['vol']
            
            test_outputs = sliding_window_inference(t_volume.to(self.device), roi_size, sw_batch_size, model)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.53

            return Image.fromarray(np.asarray(test_outputs.detach().cpu()[0, 1, :, :, 20]))
    







    # import pdb;pdb.set_trace()

        # for i in range(60):
        #     # plot the slice [:, :, 80]
        #     plt.figure("check", (18, 6))
        #     plt.subplot(1, 3, 1)
        #     plt.title(f"image {i}")
        # x = plt.imshow(test_patient["vol"][0, 0, :, :, 20], cmap="gray")
        #     plt.subplot(1, 3, 2)
        #     plt.title(f"label {i}")
        #     y = plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)
        #     plt.subplot(1, 3, 3)
        #     plt.title(f"output {i}")
        #     z = plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
        #     plt.show()
        # return Image.fromarray(np.asarray(test_outputs.detach().cpu()[0, 1, :, :, 20]))     ****************
        # x = plt.imshow(test_patient["vol"][0, 0, :, :, 20], cmap="gray")
        # x = x.convert("L")
        

    