import os
import glob
import time
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SPLITS = ("train", "val", "test")
DOMAINS = ("A_corrupted", "B_motion_free")
PSC_CLIP = 100.0

def load_volume(path):
    """
    Load a single fMRI volume from a nifti file.

    Args:
        path: str, path to a .nii.gz volume.

    Returns:
        np.ndarray of shape (X, Y, Z), float32.
    """
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def resize_volume(volume, target_shape):
    """
    Resample a volume to a new spatial resolution with trilinear interpolation.

    Args:
        volume: np.ndarray of shape (X, Y, Z).
        target_shape: tuple(int, int, int) or None. None keeps the native resolution.

    Returns:
        np.ndarray, resized to target_shape (or unchanged if target_shape is None).
    """
    if target_shape is None or tuple(target_shape) == volume.shape:
        return volume
    vol = torch.from_numpy(volume)[None, None]
    vol = F.interpolate(vol, size=target_shape, mode="trilinear", align_corners=False)
    return vol[0, 0].numpy()



def psc_normalize(volume):
    """
    Normalize a volume to percent signal change relative to its own brain-mean
    intensity, then clip to +/-100% of that baseline (0x-2x baseline, which covers
    virtually all brain tissue) and rescale to [0, 1]. True PSC needs a temporal
    baseline; since each sample here is a single volume, the spatial mean over
    brain (nonzero) voxels is used as the baseline instead. [0, 1] is used, rather
    than raw percent units, because the existing generator decoder ends in a
    Sigmoid: the reconstruction loss can only be satisfied if its target actually
    lies in [0, 1].

    Args:
        volume: np.ndarray of shape (X, Y, Z).

    Returns:
        tuple(np.ndarray volume rescaled to [0, 1], float32 baseline) - baseline is
        kept so the volume can be denormalized back to raw intensity later.
    """
    brain_voxels = volume[volume > 0]
    baseline = brain_voxels.mean() if brain_voxels.size > 0 else volume.mean()
    baseline = np.float32(baseline) + 1e-6
    psc = np.clip((volume - baseline) / baseline * 100.0, -PSC_CLIP, PSC_CLIP)
    normalized = ((psc / PSC_CLIP + 1.0) / 2.0).astype(np.float32)
    return normalized, baseline


def psc_denormalize(volume, baseline):
    """
    Invert psc_normalize back to raw signal intensity. Lossy for voxels that were
    clipped beyond +/-100% baseline during normalization.

    Args:
        volume: np.ndarray or torch.Tensor, normalized [0, 1] volume.
        baseline: float or broadcastable tensor, the baseline from psc_normalize.

    Returns:
        same type as volume, in raw intensity units.
    """
    psc = (volume * 2.0 - 1.0) * PSC_CLIP
    return psc / 100.0 * baseline + baseline


def flip_augment(volume, p=0.2):
    """
    Randomly mirror the volume left-right (sagittal flip) - the one geometric
    augmentation that keeps brain anatomy plausible without distorting the
    motion-artifact signal the model is meant to learn.

    Args:
        volume: np.ndarray of shape (X, Y, Z).
        p: float, probability of flipping.

    Returns:
        np.ndarray, flipped or unchanged.
    """
    if np.random.rand() < p:
        volume = np.ascontiguousarray(np.flip(volume, axis=0))
    return volume


class FmriVolumeDataset(Dataset):

    def __init__(self, data_root, split, domain, target_shape=None, augment=False):
        """
        Dataset over one split/domain folder of the chunk1 fMRI volumes.

        Args:
            data_root: str, root of the cycleGANS_experimental_chunk1 dataset.
            split: str, one of "train", "val", "test".
            domain: str, one of "A_corrupted", "B_motion_free".
            target_shape: tuple(int, int, int) or None, resample resolution.
            augment: bool, apply flip_augment (intended for training only).
        """
        self.files = sorted(glob.glob(os.path.join(data_root, split, domain, "*.nii.gz")))
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        """Return the number of volumes in this split/domain."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load, resample and normalize one volume.

        Args:
            idx: int, sample index.

        Returns:
            dict with "image" (1, X, Y, Z) float32 tensor in PSC units, "baseline"
            (scalar float tensor, for denormalization) and "path" (str).
        """
        path = self.files[idx]
        volume = load_volume(path)
        volume = resize_volume(volume, self.target_shape)
        volume, baseline = psc_normalize(volume)
        if self.augment:
            volume = flip_augment(volume)
        image = torch.from_numpy(volume).unsqueeze(0)
        return {"image": image, "baseline": torch.tensor(baseline), "path": path}


def get_dataloaders(data_root, batch_size=2, target_shape=None, augment_train=True,
                     num_workers=4, persistent_workers=True):
    """
    Build train/val/test dataloaders for both domains from one dataset implementation.

    Args:
        data_root: str, root of the cycleGANS_experimental_chunk1 dataset.
        batch_size: int, batch size used by every loader.
        target_shape: tuple(int, int, int) or None, resample resolution, e.g. (80, 96, 72).
        augment_train: bool, whether the training loaders apply flip_augment.
        num_workers: int, worker processes per loader.
        persistent_workers: bool, keep workers alive across epochs. Saves respawn cost
            in a real multi-epoch training loop, but for a short-lived script (like the
            sanity check below) it just leaves worker processes to tear down at exit -
            pass False for one-shot use.

    Returns:
        dict[str, dict[str, DataLoader]], indexed as loaders[split][domain].
    """
    loaders = {}
    for split in SPLITS:
        loaders[split] = {}
        for domain in DOMAINS:
            dataset = FmriVolumeDataset(data_root, split, domain, target_shape=target_shape,
                                         augment=augment_train and split == "train")
            loaders[split][domain] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=persistent_workers and num_workers > 0,
            )
    return loaders


if __name__ == "__main__":
    data_root = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cycleGANS_experimental_chunk1"
    target_shape = (80, 96, 72)

    loaders = get_dataloaders(data_root, batch_size=2, target_shape=target_shape,
                               num_workers=4, persistent_workers=False)

    for split in SPLITS:
        for domain in DOMAINS:
            batch = next(iter(loaders[split][domain]))
            image, baseline, paths = batch["image"], batch["baseline"], batch["path"]
            assert tuple(image.shape[2:]) == target_shape
            denorm = psc_denormalize(image, baseline.view(-1, 1, 1, 1, 1))

            raw = load_volume(paths[0])
            resized = resize_volume(raw, target_shape)
            print(f"{split}/{domain}: {tuple(image.shape)} "
                  f"raw[{raw.min():.1f}, {raw.max():.1f}] "
                  f"resized[{resized.min():.1f}, {resized.max():.1f}] "
                  f"psc[{image[0].min():.2f}, {image[0].max():.2f}] "
                  f"denorm[{denorm[0].min():.1f}, {denorm[0].max():.1f}]")

    n_batches = 5
    loader = loaders["train"]["A_corrupted"]
    start = time.time()
    for i, _ in enumerate(loader):
        if i + 1 == n_batches:
            break
    print(f"{n_batches} train batches (batch_size={loader.batch_size}) in {time.time() - start:.2f}s")
