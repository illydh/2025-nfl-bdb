import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

torch.set_float32_matmul_precision("medium")

class GhostFormer(nn.Module):
    def __init__(
        self,
        num_positions: int = 21,
        pos_emb_dim: int = 16,
        model_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.numeric_dim = 3
        self.pos_embedding = nn.Embedding(num_positions, pos_emb_dim)
        in_dim = self.numeric_dim + pos_emb_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial Encoder (inter-player)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_layers)
        
        # Temporal Encoder (frame-to-frame)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, 120 + 60) # X(120), Y(60)
        )

    def forward(self, numeric, pos_ids, mask_idx, valid_mask):
        # numeric: (B, T, 22, 3), pos_ids: (B, T, 22), mask_idx: (B,), valid_mask: (B, T)
        B, T, seq_len, _ = numeric.shape
        
        # Collapse B and T for spatial attention
        numeric_flat = numeric.view(B*T, seq_len, 3)
        pos_ids_flat = pos_ids.view(B*T, seq_len)
        
        pos_embs = self.pos_embedding(pos_ids_flat)
        x = torch.cat([numeric_flat, pos_embs], dim=-1)
        x = self.input_projection(x)
        
        x = self.spatial_transformer(x) # (B*T, 22, model_dim)
        
        # Gather masked token
        mask_idx_flat = mask_idx.unsqueeze(1).repeat(1, T).view(-1).to(x.device)
        batch_indices = torch.arange(B*T, device=x.device)
        masked_token_repr = x[batch_indices, mask_idx_flat] # (B*T, model_dim)
        
        # Restore T dimension for temporal processing
        masked_token_seq = masked_token_repr.view(B, T, -1) # (B, T, model_dim)
        
        # Generate padding mask for temporal transformer
        # In PyTorch, padding_mask True means IGNORE. Our valid_mask is 1 for valid, 0 for pad.
        src_key_padding_mask = (valid_mask == 0).to(x.device) 
        
        t_out = self.temporal_transformer(masked_token_seq, src_key_padding_mask=src_key_padding_mask) # (B, T, model_dim)
        
        output = self.decoder(t_out) # (B, T, 180)
        logits_x = output[:, :, :120]
        logits_y = output[:, :, 120:]
        return logits_x, logits_y

class GhostFormerLitModel(LightningModule):
    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = GhostFormer()

    def forward(self, numeric, pos_ids, mask_idx, valid_mask):
        return self.model(numeric, pos_ids, mask_idx, valid_mask)

    def classification_loss(self, logits_x, logits_y, target, valid_mask):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        # Flatten for loss computation
        logits_x_flat = logits_x.reshape(-1, 120)
        logits_y_flat = logits_y.reshape(-1, 60)
        target_flat = target.reshape(-1, 2)
        valid_flat = valid_mask.reshape(-1)
        
        loss_x = loss_fn(logits_x_flat, target_flat[:, 0])
        loss_y = loss_fn(logits_y_flat, target_flat[:, 1])
        
        # Mask out padded positions
        total_loss = ((loss_x + loss_y) * valid_flat).sum() / (valid_flat.sum() + 1e-6)
        return total_loss

    def training_step(self, batch, batch_idx):
        numeric, pos_ids, mask_idx = batch["numeric"], batch["pos_ids"], batch["mask_idx"]
        target, valid_mask = batch["target"], batch["valid_mask"]
        
        logits_x, logits_y = self(numeric, pos_ids, mask_idx, valid_mask)
        loss = self.classification_loss(logits_x, logits_y, target, valid_mask)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        numeric, pos_ids, mask_idx = batch["numeric"], batch["pos_ids"], batch["mask_idx"]
        target, valid_mask = batch["target"], batch["valid_mask"]
        
        logits_x, logits_y = self(numeric, pos_ids, mask_idx, valid_mask)
        loss = self.classification_loss(logits_x, logits_y, target, valid_mask)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
