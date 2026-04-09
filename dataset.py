import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEFENSE_POSITIONS = ["CB", "S", "FS", "SS", "LB", "MLB", "OLB", "DE", "DT", "NT"]

# 19 positions
POSITIONS = sorted(['C', 'CB', 'DB', 'DE', 'DT', 'FB', 'FS', 'G', 'ILB', 'LB', 'LS', 'MLB', 'NT', 'OLB', 'QB', 'RB', 'S', 'SS', 'T', 'TE', 'WR'])
POS_TO_IDX = {p: i for i, p in enumerate(POSITIONS)}

class MaskedPlayerDataset(Dataset):
    def __init__(self, frames_path):
        """
        frames_path: path to the saved .pt file containing dictionaries.
        """
        self.frames = torch.load(frames_path, weights_only=False)
        self.defense_positions = set(DEFENSE_POSITIONS)
        
        # Flatten dataset: instead of each index being a frame, 
        # each index is a (frame_idx, mask_player_idx).
        # We only augment by masking out the DEFENSIVE players per iteration (meaning 11 variations per frame).
        
        self.samples = []
        for f_idx, frame in enumerate(self.frames):
            is_def = filter(lambda i: frame['isDefense'][i] == 1, range(22))
            for p_idx in is_def:
                self.samples.append((f_idx, p_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, mask_idx = self.samples[idx]
        frame = self.frames[f_idx]

        xs = np.array(frame["x"], dtype=np.float32)
        ys = np.array(frame["y"], dtype=np.float32)
        pos = frame["position"]  # list of strings, len 22

        # Build numeric features: [x, y, is_mask]
        numeric = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)  # (22, 3)
        
        # apply mask
        target = np.array([xs[mask_idx], ys[mask_idx]], dtype=np.float32)
        numeric[mask_idx, 0] = 0.0
        numeric[mask_idx, 1] = 0.0
        numeric[mask_idx, 2] = 1.0  # is_mask flag

        # Position ids
        pos_ids = np.array([POS_TO_IDX.get(p, 0) for p in pos], dtype=np.int64)

        return {
            "numeric": torch.from_numpy(numeric),   # (22, 3)
            "pos_ids": torch.from_numpy(pos_ids),   # (22,)
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
            "target": torch.from_numpy(target),     # (2,)
        }

def get_dataloader(frames_path, batch_size=64, shuffle=True, num_workers=4):
    ds = MaskedPlayerDataset(frames_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
