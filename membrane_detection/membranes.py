import numpy as np

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    ScaleIntensity,
    EnsureType,
)


class NumpyDataset(monai.data.Dataset):
    """
    Dataset to load images directly from Numpy arrays rather than from files
    """
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, idx):
        img = self.data[:, :, idx]
        return self.transform(img)


def segment_membranes(vol, pth="best_metric_model_segmentation2d_dict.pth", device_s='cpu'):
    """Segment membranes from electron microscopy images
        Parameters
        ----------
        vol : 3-dimensional Numpy array
            Volume of electron microscopy images
        pth : str, optional
            Path to U-net weights pth file
        Returns
        -------
        val_outputs : uint8 3-dimensional Numpy array
            Membrane segmentation volume (0 - no membrane, 255 - membrane)
    """
    if device_s == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    val_transforms = Compose(
        [
            AddChannel(),
            ScaleIntensity(),
            EnsureType(),
        ]
    )

    val_ds = NumpyDataset(data=vol, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    if device_s == 'cpu':
        model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(pth))
    model.eval()
    val_outputs = []
    with torch.no_grad():
        for val_data in val_loader:
            val_image = val_data.to(device)
            roi_size = (512, 512)
            sw_batch_size = 4
            val_output = sliding_window_inference(val_image, roi_size, sw_batch_size, model)
            val_output = [post_trans(i) for i in decollate_batch(val_output)]
            val_outputs.append(np.squeeze(val_output[0].cpu()))
    val_outputs = np.stack(val_outputs, axis=2).astype('uint8')

    return val_outputs


if __name__ == "__main__":
    from cloudvolume import CloudVolume
    from matplotlib import pyplot as plt

    vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    data = np.squeeze(vol[100000:101000, 100000:101000, 20000:20100])
    seg = segment_membranes(data, pth='model_CREMI_2d.pth', device_s='gpu')

    plt.figure()
    plt.subplot(121)
    plt.imshow(data[:, :, 50])
    plt.subplot(122)
    plt.imshow(seg[:, :, 50])
    plt.show()