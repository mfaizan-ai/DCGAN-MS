import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmri_dataset import (
    SPLITS, DOMAINS, get_dataloaders, load_volume, resize_volume, flip_augment,
)

MEDIA_DIR = "media"


def mid_slice(volume, axis=2):
    """
    Take the middle slice of a volume along an axis (default Z, the slice direction).

    Args:
        volume: np.ndarray of shape (X, Y, Z).
        axis: int, axis to slice along.

    Returns:
        np.ndarray, 2D slice.
    """
    index = volume.shape[axis] // 2
    return volume.take(index, axis=axis)


def save_stage_figure(stages, title, out_path):
    """
    Plot a sequence of named 2D slices side by side and save the figure to disk.

    Args:
        stages: list of (name, np.ndarray) pairs to display left to right.
        title: str, figure title.
        out_path: str, file path to save the PNG to.

    Returns:
        None.
    """
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, (name, img) in zip(axes, stages):
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    data_root = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cycleGANS_experimental_chunk1"
    target_shape = (80, 96, 72)
    os.makedirs(MEDIA_DIR, exist_ok=True)

    loaders = get_dataloaders(data_root, batch_size=1, target_shape=target_shape,
                               num_workers=0, persistent_workers=False)

    for split in SPLITS:
        for domain in DOMAINS:
            batch = next(iter(loaders[split][domain]))
            path = batch["path"][0]
            normalized = batch["image"][0, 0].numpy()

            raw = load_volume(path)
            resized = resize_volume(raw, target_shape)
            # augmentation is random (p=0.2) inside the dataset, so force one here
            # to actually show its effect rather than relying on chance
            augmented = flip_augment(normalized, p=1.0)

            stages = [("raw", mid_slice(raw)), ("downsampled", mid_slice(resized)),
                      ("normalized [0,1]", mid_slice(normalized)),
                      ("augmented (flipped)", mid_slice(augmented))]
            out_path = os.path.join(MEDIA_DIR, f"{split}_{domain}.png")
            save_stage_figure(stages, f"{split}/{domain}", out_path)
            print(f"saved {out_path}")
