import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import glob
import os
import random

from dataset import POS_TO_IDX
from models import GhostFormerLitModel

def draw_field(ax):
    """Draws a minimalist football field on the given ax."""
    # Field dimensions (normalized relative to middle)
    # usually plays happen within -30 to 30 yards of middle.
    ax.set_facecolor('#E6F5E6') # Very light, non-distracting green
    ax.set_xlim(-40, 40)
    ax.set_ylim(-30, 30)
    
    # Draw yard lines
    for i in range(-40, 41, 5):
        ax.axvline(i, color='white', alpha=0.5, linestyle='-', zorder=0)
        
    # Draw hash marks
    for i in range(-40, 41, 1):
        ax.plot([i, i], [-1, 1], color='white', alpha=0.3, zorder=0)
        ax.plot([i, i], [29, 31], color='white', alpha=0.3, zorder=0)
        ax.plot([i, i], [-29, -31], color='white', alpha=0.3, zorder=0)

    ax.axis('off')

def get_ellipse(mean_x, mean_y, logvar_x, logvar_y, n_std=2.0, **kwargs):
    """Returns a matplotlib patch for an ellipse representing confidence."""
    var_x = np.exp(logvar_x)
    var_y = np.exp(logvar_y)
    
    width = 2 * n_std * np.sqrt(var_x)
    height = 2 * n_std * np.sqrt(var_y)
    
    return Ellipse(xy=(mean_x, mean_y), width=width, height=height, **kwargs)

def run_visualization():
    print("Loading test frames for visualization...")
    try:
        frames = torch.load("data/test_frames.pt", weights_only=False)
    except FileNotFoundError:
        print("test_frames.pt not found. Ensure preprocess.py has run on week 9.")
        return
        
    if not frames:
        print("No frames loaded.")
        return
        
    # Pick a random play
    # frames are grouped by gameId, playId
    plays = {}
    for i, frame in enumerate(frames):
        key = (frame['gameId'], frame['playId'])
        if key not in plays:
            plays[key] = []
        plays[key].append((i, frame))
        
    # Try to find a play with at least 15 frames BEFORE_SNAP 
    valid_plays = [k for k, v in plays.items() if len(v) >= 15]
    if not valid_plays:
        selected_key = list(plays.keys())[0]
    else:
        selected_key = random.choice(valid_plays)
        
    play_frames = plays[selected_key]
    
    # Sort them by frameId just in case
    # Assuming the data is already sorted by frameId, but to be sure we don't have it explicitly accessible
    # Actually, we can rely on chronological order of the file
    print(f"Visualizing Game {selected_key[0]}, Play {selected_key[1]} ({len(play_frames)} frames)")
    
    # Load Model
    ckpt_files = glob.glob("checkpoints/*.ckpt")
    if ckpt_files:
        model = GhostFormerLitModel.load_from_checkpoint(ckpt_files[0])
    else:
        model = GhostFormerLitModel()
    model.eval()
    
    # Pick a random defensive player to mask throughout the play
    first_frame = play_frames[0][1]
    defenders = [i for i, is_def in enumerate(first_frame['isDefense']) if is_def == 1]
    if not defenders:
        print("No defenders found!")
        return
    mask_target_idx = defenders[0] 
    
    # Pre-compute model predictions and actuals
    history_pred_x = []
    history_pred_y = []
    history_actual_x = []
    history_actual_y = []
    logvars_x = []
    logvars_y = []
    
    # Extract fixed player info for the rest 21
    
    all_player_coords = [] # frame -> [ (x, y, is_def), ... ]
    
    with torch.no_grad():
        for i, (global_idx, frame) in enumerate(play_frames):
            xs = np.array(frame['x'], dtype=np.float32)
            ys = np.array(frame['y'], dtype=np.float32)
            is_defs = frame['isDefense']
            
            # Record actual target
            history_actual_x.append(xs[mask_target_idx])
            history_actual_y.append(ys[mask_target_idx])
            
            # numeric input representation
            numeric = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
            numeric[mask_target_idx, 0] = 0.0
            numeric[mask_target_idx, 1] = 0.0
            numeric[mask_target_idx, 2] = 1.0
            
            pos_ids = np.array([POS_TO_IDX.get(p, 0) for p in frame['position']], dtype=np.int64)
            
            # Convert to batch size 1
            num_t = torch.from_numpy(numeric).unsqueeze(0)
            pos_t = torch.from_numpy(pos_ids).unsqueeze(0)
            mask_t = torch.tensor([mask_target_idx], dtype=torch.long)
            
            pred_mean, pred_logvar = model(num_t, pos_t, mask_t)
            px, py = pred_mean[0].numpy()
            plvx, plvy = pred_logvar[0].numpy()
            
            history_pred_x.append(px)
            history_pred_y.append(py)
            logvars_x.append(plvx)
            logvars_y.append(plvy)
            
            # Store all other players
            coords = []
            for pidx in range(22):
                if pidx != mask_target_idx:
                    coords.append((xs[pidx], ys[pidx], is_defs[pidx]))
            all_player_coords.append(coords)
            
    # Now animate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame_idx):
        ax.clear()
        draw_field(ax)
        
        # Plot other players
        cur_coords = all_player_coords[frame_idx]
        off_x = [c[0] for c in cur_coords if c[2] == 0]
        off_y = [c[1] for c in cur_coords if c[2] == 0]
        def_x = [c[0] for c in cur_coords if c[2] == 1]
        def_y = [c[1] for c in cur_coords if c[2] == 1]
        
        ax.scatter(off_x, off_y, color='red', s=50, label='Offense', edgecolors='white', zorder=3)
        ax.scatter(def_x, def_y, color='blue', s=50, label='Defense', edgecolors='white', zorder=3)
        
        # Plot mask ground truth
        ax.scatter([history_actual_x[frame_idx]], [history_actual_y[frame_idx]], 
                   color='blue', s=80, marker='X', edgecolors='black', label='Target True', zorder=4)
                   
        # Plot prediction trajectory up to frame
        # Draw path
        if frame_idx > 0:
            ax.plot(history_pred_x[:frame_idx+1], history_pred_y[:frame_idx+1], color='orange', linestyle='--', linewidth=2, zorder=2)
            
        cur_px = history_pred_x[frame_idx]
        cur_py = history_pred_y[frame_idx]
        cur_lvx = logvars_x[frame_idx]
        cur_lvy = logvars_y[frame_idx]
        
        # Plot prediction dot
        ax.scatter([cur_px], [cur_py], color='orange', s=80, marker='*', edgecolors='black', label='Predicted', zorder=5)
        
        # Confidence ellipse
        ellipse = get_ellipse(cur_px, cur_py, cur_lvx, cur_lvy, n_std=1.0, 
                              alpha=0.3, color='orange', zorder=1)
        ax.add_patch(ellipse)
        
        # Confidence text indicator (average stdev)
        conf_val = np.sqrt(np.exp(cur_lvx)) + np.sqrt(np.exp(cur_lvy)) # rough measure
        ax.text(0.02, 0.95, f'Confidence (lower var = better): {conf_val:.2f}\nFrame: {frame_idx+1}/{len(play_frames)}', 
                transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
        # Title
        ax.set_title(f"Play Tracking: Game {selected_key[0]} - Play {selected_key[1]}")
        ax.legend(loc='lower right')
        
    anim = FuncAnimation(fig, update, frames=len(play_frames), interval=100)
    anim.save('visualization.mp4', writer='ffmpeg', fps=10)
    print("Saved visualization to visualization.mp4")

if __name__ == "__main__":
    run_visualization()
