import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MaskedPlayerDataset
from models import GhostFormerLitModel
import glob

def evaluate():
    print("Loading test dataset (Week 9)...")
    test_dataset = MaskedPlayerDataset("data/test_frames.pt")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    ckpt_files = glob.glob("checkpoints/*.ckpt")
    if ckpt_files:
        print(f"Loading model from checkpoint {ckpt_files[0]}")
        model = GhostFormerLitModel.load_from_checkpoint(ckpt_files[0])
    else:
        model = GhostFormerLitModel()
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_mse = 0
    count = 0
    
    print("Evaluating Spatiotemporal Metrics...")
    with torch.no_grad():
        for batch in test_loader:
            numeric, pos_ids, mask_idx = batch["numeric"].to(device), batch["pos_ids"].to(device), batch["mask_idx"].to(device)
            target, valid_mask = batch["target"].to(device), batch["valid_mask"].to(device)
            
            logits_x, logits_y = model(numeric, pos_ids, mask_idx, valid_mask)
            
            # Predict coordinates
            pred_x = logits_x.argmax(dim=-1).float() - 60
            pred_y = logits_y.argmax(dim=-1).float() - 30
            pred_mean = torch.stack([pred_x, pred_y], dim=-1) # (B, T, 2)
            
            true_x = target[:, :, 0].float() - 60
            true_y = target[:, :, 1].float() - 30
            target_mean = torch.stack([true_x, true_y], dim=-1)
            
            # Mask out padding in MSE
            diff_sq = ((pred_mean - target_mean) ** 2)
            mse = (diff_sq.sum(dim=-1) * valid_mask).sum().item()
            
            total_mse += mse
            count += valid_mask.sum().item() * 2 # 2 coordinates per valid frame
            
    print("-"*40)
    print("EVALUATION RESULTS (WEEK 9)")
    print(f"Mean Squared Error (MSE): {total_mse/count:.4f}")
    print("-"*40)

if __name__ == "__main__":
    evaluate()
