import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEFENSE_POSITIONS = ["CB", "S", "FS", "SS", "LB", "MLB", "OLB", "DE", "DT", "NT"]
POSITIONS = sorted(['C', 'CB', 'DB', 'DE', 'DT', 'FB', 'FS', 'G', 'ILB', 'LB', 'LS', 'MLB', 'NT', 'OLB', 'QB', 'RB', 'S', 'SS', 'T', 'TE', 'WR'])
POS_TO_IDX = {p: i for i, p in enumerate(POSITIONS)}

MAX_SEQ_LEN = 50

class MaskedPlayerDataset(Dataset):
    def __init__(self, frames_path):
        raw_frames = torch.load(frames_path, weights_only=False)
        
        # Group frames into plays
        plays = {}
        for frame in raw_frames:
            # Enforce strict player ordering by nflId
            nflIds = frame['nflId']
            sort_idx = np.argsort(nflIds)
            frame['nflId'] = [frame['nflId'][i] for i in sort_idx]
            frame['position'] = [frame['position'][i] for i in sort_idx]
            frame['isDefense'] = [frame['isDefense'][i] for i in sort_idx]
            frame['x'] = [frame['x'][i] for i in sort_idx]
            frame['y'] = [frame['y'][i] for i in sort_idx]
            
            k = (frame['gameId'], frame['playId'])
            if k not in plays:
                plays[k] = []
            plays[k].append(frame)
            
        self.samples = []
        self.plays_list = []
        
        play_idx = 0
        for k in plays:
            # Sort by frameId natively (frameId increments chronologically)
            play_frames = sorted(plays[k], key=lambda f: f['frameId'])
            
            # constrain to MAX_SEQ_LEN
            play_frames = play_frames[:MAX_SEQ_LEN]
            self.plays_list.append(play_frames)
            
            is_def = [i for i in range(22) if play_frames[0]['isDefense'][i] == 1]
            for p_idx in is_def:
                self.samples.append((play_idx, p_idx))
            play_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        play_idx, mask_idx = self.samples[idx]
        play = self.plays_list[play_idx]
        seq_len = len(play)
        
        numeric_seq = np.zeros((MAX_SEQ_LEN, 22, 3), dtype=np.float32)
        pos_seq = np.zeros((MAX_SEQ_LEN, 22), dtype=np.int64)
        target_seq = np.zeros((MAX_SEQ_LEN, 2), dtype=np.int64)
        valid_mask = np.zeros((MAX_SEQ_LEN,), dtype=np.float32)
        
        for t, frame in enumerate(play):
            xs = np.array(frame["x"], dtype=np.float32)
            ys = np.array(frame["y"], dtype=np.float32)
            
            numeric = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
            target_x = min(max(int(xs[mask_idx] + 60), 0), 119)
            target_y = min(max(int(ys[mask_idx] + 30), 0), 59)
            
            target_seq[t] = [target_x, target_y]
            
            numeric[mask_idx, 0] = 0.0
            numeric[mask_idx, 1] = 0.0
            numeric[mask_idx, 2] = 1.0  
            
            numeric_seq[t] = numeric
            pos_seq[t] = np.array([POS_TO_IDX.get(p, 0) for p in frame["position"]], dtype=np.int64)
            valid_mask[t] = 1.0

        return {
            "numeric": torch.from_numpy(numeric_seq),   # (T, 22, 3)
            "pos_ids": torch.from_numpy(pos_seq),       # (T, 22)
            "mask_idx": torch.tensor(mask_idx, dtype=torch.long),
            "target": torch.from_numpy(target_seq),     # (T, 2)
            "valid_mask": torch.from_numpy(valid_mask)  # (T,)
        }

def get_dataloader(frames_path, batch_size=64, shuffle=True, num_workers=4):
    ds = MaskedPlayerDataset(frames_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
