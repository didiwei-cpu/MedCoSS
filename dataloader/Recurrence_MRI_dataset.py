import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize


class RecurrenceMRIDataset(Dataset):
    """Dataset for binary recurrence classification using paired T1/T2 MR volumes.

    The ROI union is cropped and resized to ``crop_size`` (depth, height, width).
    Channels correspond to normalized T1 and T2 volumes.
    """

    def __init__(self, root_dir, split="train", label_file="CC_end.xlsx", crop_size=(8, 64, 64)):
        self.root = os.path.join(root_dir, split)
        self.crop_size = crop_size
        label_path = os.path.join(root_dir, label_file)
        df = pd.read_excel(label_path)
        self.labels = {str(k): int(v) for k, v in zip(df["SampleID"], df["Recurrence"])}
        self.samples = []
        for sid in sorted(os.listdir(self.root)):
            if sid in self.labels:
                base = os.path.join(self.root, sid)
                self.samples.append({
                    "t1": os.path.join(base, "t1_image.nii.gz"),
                    "t1_roi": os.path.join(base, "t1_roi.nii.gz"),
                    "t2": os.path.join(base, "t2_image.nii.gz"),
                    "t2_roi": os.path.join(base, "t2_roi.nii.gz"),
                    "label": self.labels[sid],
                })
        self.label_list = [s["label"] for s in self.samples]
        print(f"{split}: loaded {len(self.samples)} samples from {self.root}")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def _normalize(self, vol):
        mean = vol.mean()
        std = vol.std() + 1e-8
        vol = (vol - mean) / std
        return np.clip(vol, -5.0, 5.0)

    def _crop_to_bbox(self, vol, bbox):
        return vol[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

    def __getitem__(self, idx):
        item = self.samples[idx]
        t1 = self._load_volume(item["t1"])
        t2 = self._load_volume(item["t2"])
        roi1 = self._load_volume(item["t1_roi"]) > 0
        roi2 = self._load_volume(item["t2_roi"]) > 0
        mask = roi1 | roi2
        pos = np.where(mask)
        zmin, zmax = pos[0].min(), pos[0].max() + 1
        ymin, ymax = pos[1].min(), pos[1].max() + 1
        xmin, xmax = pos[2].min(), pos[2].max() + 1
        t1 = self._crop_to_bbox(t1, (zmin, zmax, ymin, ymax, xmin, xmax))
        t2 = self._crop_to_bbox(t2, (zmin, zmax, ymin, ymax, xmin, xmax))
        t1 = self._normalize(t1)
        t2 = self._normalize(t2)
        volume = np.stack([
            t1 * mask[zmin:zmax, ymin:ymax, xmin:xmax],
            t2 * mask[zmin:zmax, ymin:ymax, xmin:xmax]
        ], axis=0)
        volume = resize(volume, (2,) + self.crop_size, order=1, preserve_range=True, anti_aliasing=True)
        return volume.astype(np.float32), np.int64(item["label"])
