import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import glob
import random

from dataset import POS_TO_IDX
from models import GhostFormerLitModel

def draw_field(ax):
    ax.set_facecolor('#E6F5E6')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-30, 30)
    for i in range(-40, 41, 5):
        ax.axvline(i, color='white', alpha=0.5, linestyle='-', zorder=0)
    for i in range(-40, 41, 1):
        ax.plot([i, i], [-1, 1], color='white', alpha=0.3, zorder=0)
        ax.plot([i, i], [29, 31], color='white', alpha=0.3, zorder=0)
        ax.plot([i, i], [-29, -31], color='white', alpha=0.3, zorder=0)
    ax.axis('off')

def get_ellipse(mean_x, mean_y, cur_var, n_std=1.0, **kwargs):
    width = 2 * n_std * cur_var
    height = 2 * n_std * cur_var
    return Ellipse(xy=(mean_x, mean_y), width=width, height=height, **kwargs)

def run_visualization():
    print("Loading test frames for visualization...")
    try:
        frames = torch.load("data/test_frames.pt", weights_only=False)
    except FileNotFoundError:
        print("test_frames.pt not found.")
        return
        
    plays = {}
    for frame in frames:
        nflIds = frame['nflId']
        sort_idx = np.argsort(nflIds)
        frame['nflId'] = [frame['nflId'][i] for i in sort_idx]
        frame['position'] = [frame['position'][i] for i in sort_idx]
        frame['isDefense'] = [frame['isDefense'][i] for i in sort_idx]
        frame['x'] = [frame['x'][i] for i in sort_idx]
        frame['y'] = [frame['y'][i] for i in sort_idx]
        k = (frame['gameId'], frame['playId'])
        if k not in plays: plays[k] = []
        plays[k].append(frame)
        
    valid_plays = [k for k, v in plays.items() if len(v) >= 15]
    selected_key = random.choice(valid_plays) if valid_plays else list(plays.keys())[0]
    play_frames = sorted(plays[selected_key], key=lambda f: f['frameId'])[:50]
    T_len = len(play_frames)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_files = glob.glob("checkpoints/*.ckpt")
    model = GhostFormerLitModel.load_from_checkpoint(ckpt_files[0]) if ckpt_files else GhostFormerLitModel()
    model.to(device)
    model.eval()
    
    mask_target_idx = [i for i, is_def in enumerate(play_frames[0]['isDefense']) if is_def == 1][0]
    
    history_actual_x, history_actual_y = [], []
    all_player_coords = []
    
    numeric_seq = np.zeros((1, T_len, 22, 3), dtype=np.float32)
    pos_seq = np.zeros((1, T_len, 22), dtype=np.int64)
    valid_mask = np.ones((1, T_len), dtype=np.float32)
    
    for t, frame in enumerate(play_frames):
        xs = np.array(frame['x'], dtype=np.float32)
        ys = np.array(frame['y'], dtype=np.float32)
        history_actual_x.append(xs[mask_target_idx])
        history_actual_y.append(ys[mask_target_idx])
        
        numeric = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
        numeric[mask_target_idx, 0] = 0.0
        numeric[mask_target_idx, 1] = 0.0
        numeric[mask_target_idx, 2] = 1.0
        
        numeric_seq[0, t] = numeric
        pos_seq[0, t] = np.array([POS_TO_IDX.get(p, 0) for p in frame['position']], dtype=np.int64)
        
        coords = [(xs[p], ys[p], frame['isDefense'][p]) for p in range(22) if p != mask_target_idx]
        all_player_coords.append(coords)
            
    with torch.no_grad():
        num_t = torch.from_numpy(numeric_seq).to(device)
        pos_t = torch.from_numpy(pos_seq).to(device)
        mask_t = torch.tensor([mask_target_idx], dtype=torch.long).to(device)
        valid_t = torch.from_numpy(valid_mask).to(device)
        
        logits_x, logits_y = model(num_t, pos_t, mask_t, valid_t) # (1, T, classes)
        px_arr = logits_x[0].argmax(dim=-1).cpu().numpy() - 60
        py_arr = logits_y[0].argmax(dim=-1).cpu().numpy() - 30
        
        # Uncertainty emulation
        probs_x = torch.softmax(logits_x[0], dim=-1).cpu().numpy()
        max_px = probs_x.max(axis=-1)
        var_proxy = -np.log(max_px + 1e-4) * 2
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(t):
        ax.clear()
        draw_field(ax)
        
        cur_coords = all_player_coords[t]
        off_x, off_y = [c[0] for c in cur_coords if c[2]==0], [c[1] for c in cur_coords if c[2]==0]
        def_x, def_y = [c[0] for c in cur_coords if c[2]==1], [c[1] for c in cur_coords if c[2]==1]
        
        ax.scatter(off_x, off_y, color='red', s=50, label='Offense', edgecolors='white', zorder=3)
        ax.scatter(def_x, def_y, color='blue', s=50, label='Defense', edgecolors='white', zorder=3)
        
        ax.scatter([history_actual_x[t]], [history_actual_y[t]], color='blue', s=80, marker='X', edgecolors='black', label='Target True', zorder=4)
        if t > 0:
            ax.plot(px_arr[:t+1], py_arr[:t+1], color='orange', linestyle='--', linewidth=2, zorder=2)
            
        cur_px, cur_py, cur_var = px_arr[t], py_arr[t], var_proxy[t]
        ax.scatter([cur_px], [cur_py], color='orange', s=80, marker='*', edgecolors='black', label='Predicted', zorder=5)
        ax.add_patch(get_ellipse(cur_px, cur_py, cur_var, alpha=0.3, color='orange', zorder=1))
        ax.set_title(f"Play Tracking: Game {selected_key[0]} - Play {selected_key[1]} | Frame {t+1}/{T_len}")
        ax.legend(loc='lower right')
        
    anim = FuncAnimation(fig, update, frames=T_len, interval=100)
    anim.save('outputs/visualization.mp4', writer='ffmpeg', fps=10)
    print("Saved visualization.mp4 to outputs/")

if __name__ == "__main__":
    run_visualization()
