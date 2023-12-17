from rich import print

import os

from dataset_synapse import Synapse_dataset

from torch.utils.data import DataLoader

import numpy as np

from scipy.ndimage import zoom

import torch

DATASET_SPLIT = "test"
BASE_DIR = f"./datasets/Synapse/{DATASET_SPLIT}_vol_h5"
SPLIT = f"{DATASET_SPLIT}_vol"
LIST_DIR = "./lists/Synapse"
SLICES_DIR = f"./datasets/Synapse/{DATASET_SPLIT}_npz"
os.makedirs(SLICES_DIR) if not os.path.exists(SLICES_DIR) else None
PATCH_SIZE = (224, 224)


dataset = Synapse_dataset(base_dir=BASE_DIR, split=SPLIT, list_dir=LIST_DIR)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

slices_str = ""

for batch_id, batch in enumerate(dataloader):
    print(f"Working on batch: {batch_id}")

    vol_image, vol_label, case_name = batch["image"], batch["label"], batch['case_name'][0]

    for slice_idx in range(vol_image.shape[1]):
        slice_image, slice_label = vol_image[:, slice_idx, :, :], vol_label[:, slice_idx, :, :]
        slice_image = slice_image.squeeze(0)

        # print(f"slice_image.shape: {slice_image.shape}")

        x, y = slice_image.shape

        slice_image = zoom(slice_image, (PATCH_SIZE[0] / x, PATCH_SIZE[1] / y), order=3)

        # print(f"slice_image.shape: {slice_image.shape}")

        slice_image = np.expand_dims(slice_image, axis=0)
        slice_image = torch.tensor(slice_image)

        print(f"slice_image.shape: {slice_image.shape}")
        print(f"slice_label.shape: {slice_label.shape}")

        exit()

        np.savez_compressed(
            file=f"{SLICES_DIR}/{case_name}_slice{slice_idx:03d}",
            image=slice_image,
            label=slice_label,
            case_name=case_name,
            slice_num=slice_idx
        )

        slices_str += f"{case_name}_slice{slice_idx:03d}\n"


with open(f"{LIST_DIR}/{DATASET_SPLIT}.txt", 'w') as file:
    file.write(slices_str)



