import os
import polars as pl
import torch
import numpy as np
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_weeks(weeks, split_name):
    print(f"Preprocessing {split_name} (weeks {weeks})...")
    players = pl.read_csv(f"{DATA_DIR}/players.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
    plays = pl.read_csv(f"{DATA_DIR}/plays.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
    
    # Pre-filter players to just what we need
    players_meta = players.select(["nflId", "displayName", "position"]).unique()
    plays_meta = plays.select("gameId", "playId", "defensiveTeam")
    
    all_frames = []
    
    for w in weeks:
        print(f"Loading tracking_week_{w}.csv...")
        if not os.path.exists(f"{DATA_DIR}/tracking_week_{w}.csv"):
            continue
            
        df = pl.read_csv(f"{DATA_DIR}/tracking_week_{w}.csv", null_values=["NA", "nan", "N/A", "NaN", ""])
        
        # Filter football and only keep BEFORE_SNAP
        df = df.filter((pl.col("displayName") != "football") & (pl.col("frameType") == "BEFORE_SNAP"))
        
        # Join plays and players
        df = df.join(plays_meta, on=["gameId", "playId"], how="inner")
        df = df.join(players_meta, on=["nflId", "displayName"], how="left")
        
        # Add isDefense
        df = df.with_columns(
            isDefense=pl.when(pl.col("club") == pl.col("defensiveTeam")).then(pl.lit(1)).otherwise(pl.lit(0))
        )
        
        # Drop players with missing position
        df = df.filter(pl.col("position").is_not_null())
        
        print(f"Computing middle points for week {w}...")
        centers = df.group_by(["gameId", "playId", "frameId", "isDefense"]).agg([
            pl.col("x").mean().alias("mean_x"),
            pl.col("y").mean().alias("mean_y"),
        ])
        
        middle = centers.group_by(["gameId", "playId", "frameId"]).agg([
            pl.col("mean_x").mean().alias("mid_x"),
            pl.col("mean_y").mean().alias("mid_y"),
        ])
        
        df = df.join(middle, on=["gameId", "playId", "frameId"], how="inner")
        
        # Normalize relative to middle point
        df = df.with_columns([
            (pl.col("x") - pl.col("mid_x")).alias("x_norm"),
            (pl.col("y") - pl.col("mid_y")).alias("y_norm"),
        ])
        
        # We need frames to have EXACTLY 22 players
        counts = df.group_by(["gameId", "playId", "frameId"]).agg(pl.col("nflId").n_unique().alias("unique_len"))
        valid_frames = counts.filter(pl.col("unique_len") == 22).select(["gameId", "playId", "frameId"])
        df = df.join(valid_frames, on=["gameId", "playId", "frameId"], how="inner")
        
        # Group to lists
        frames_df = df.group_by(["gameId", "playId", "frameId"]).agg([
            pl.col("nflId"),
            pl.col("position"),
            pl.col("isDefense"),
            pl.col("x_norm").alias("x"),
            pl.col("y_norm").alias("y"),
        ]).to_dicts()
        
        all_frames.extend(frames_df)
        print(f"Week {w} processed: {len(frames_df)} frames added.")

    print(f"Total {split_name} frames: {len(all_frames)}")
    torch.save(all_frames, f"{DATA_DIR}/{split_name}_frames.pt")

if __name__ == "__main__":
    preprocess_weeks(list(range(1, 9)), "train")
    preprocess_weeks([9], "test")
