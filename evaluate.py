import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MaskedPlayerDataset
from models import GhostFormerLitModel

def evaluate():
    print("Loading test dataset (Week 9)...")
    test_dataset = MaskedPlayerDataset("data/test_frames.pt")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Assuming we have a saved checkpoint from training. 
    # If not, we will just use an untrained model to demonstrate the pipeline.
    import glob
    ckpt_files = glob.glob("checkpoints/*.ckpt")
    if ckpt_files:
        print(f"Loading model from checkpoint {ckpt_files[0]}")
        model = GhostFormerLitModel.load_from_checkpoint(ckpt_files[0])
    else:
        print("No checkpoint found. Evaluating with untrained model...")
        model = GhostFormerLitModel()
        
    model.eval()
    
    total_mse = 0
    total_mae = 0
    total_nll = 0
    count = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in test_loader:
            numeric = batch["numeric"]
            pos_ids = batch["pos_ids"]
            mask_idx = batch["mask_idx"]
            target = batch["target"]
            
            pred_mean, pred_logvar = model(numeric, pos_ids, mask_idx)
            
            mse = torch.nn.functional.mse_loss(pred_mean, target, reduction='sum').item()
            mae = torch.nn.functional.l1_loss(pred_mean, target, reduction='sum').item()
            
            # NLL metric
            var = torch.exp(pred_logvar) + 1e-6
            nll = 0.5 * (torch.log(var) + ((target - pred_mean) ** 2) / var).sum().item()
            
            total_mse += mse
            total_mae += mae
            total_nll += nll
            count += numeric.size(0) * 2 # 2 coordinates
            
    print("-"*40)
    print("EVALUATION RESULTS (WEEK 9)")
    print(f"Mean Squared Error (MSE): {total_mse/count:.4f}")
    print(f"Mean Absolute Error (MAE): {total_mae/count:.4f}")
    print(f"Gaussian NLL: {total_nll/(count/2):.4f}")
    print("-"*40)
    print("""
    Future Improvements for Spatial/Temporal Classification:
    Instead of continuous regression, the playing field can be discretized into a grid (e.g. 1x1 yard cells).
    This reframes the prediction into a spatial classification problem, which is naturally suited
    for cross-entropy loss and standard autoregressive LM objective formatting.
    Temporal relationships can be captured by adding Temporal self-attention over the sequence of frames
    instead of processing frames independently.
    """)

if __name__ == "__main__":
    evaluate()
