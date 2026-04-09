import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

torch.set_float32_matmul_precision("medium")

class GhostFormer(nn.Module):
    """Transformer model for predicting masked player position and coordinates."""
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
        # numeric = [x, y, is_mask] -> 3 
        self.numeric_dim = 3
        
        self.pos_embedding = nn.Embedding(num_positions, pos_emb_dim)
        
        # Total incoming feature size
        in_dim = self.numeric_dim + pos_emb_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decodes coordinate logits (mean_x, mean_y) -> 2 values
        # we can also output log_var_x, log_var_y (4 values total) to get confidence,
        # but let's just do standard regression for now or MDN.
        # User wants "the model's confidence", so we predict 4 values:
        # [mean_x, mean_y, logvar_x, logvar_y]
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, 4)
        )

    def forward(self, numeric, pos_ids, mask_idx):
        """
        numeric: (B, 22, 3) 
        pos_ids: (B, 22)
        mask_idx: (B,)
        """
        B, seq_len, _ = numeric.shape
        
        # embed position ID
        pos_embs = self.pos_embedding(pos_ids) # (B, 22, pos_emb_dim)
        
        x = torch.cat([numeric, pos_embs], dim=-1) # (B, 22, in_dim)
        x = self.input_projection(x)
        
        x = self.transformer(x) # (B, 22, model_dim)
        
        # We only care about the masked token's representation
        # Gather the representation of the masked token for each item in the batch
        batch_indices = torch.arange(B, device=x.device)
        masked_token_repr = x[batch_indices, mask_idx] # (B, model_dim)
        
        output = self.decoder(masked_token_repr) # (B, 4)
        
        pred_mean = output[:, :2]
        pred_logvar = output[:, 2:]
        return pred_mean, pred_logvar


class GhostFormerLitModel(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GhostFormer()

    def forward(self, numeric, pos_ids, mask_idx):
        return self.model(numeric, pos_ids, mask_idx)

    def gaussian_nll_loss(self, pred_mean, pred_logvar, target):
        """
        Calculates the Gaussian Negative Log Likelihood Loss.
        Target is [B, 2].
        """
        var = torch.exp(pred_logvar) + 1e-6
        loss = 0.5 * (torch.log(var) + ((target - pred_mean) ** 2) / var)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        numeric = batch["numeric"]
        pos_ids = batch["pos_ids"]
        mask_idx = batch["mask_idx"]
        target = batch["target"]
        
        pred_mean, pred_logvar = self(numeric, pos_ids, mask_idx)
        loss = self.gaussian_nll_loss(pred_mean, pred_logvar, target)
        
        # Also log the pure MSE for interpretability
        mse = nn.functional.mse_loss(pred_mean, target)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", mse, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        numeric = batch["numeric"]
        pos_ids = batch["pos_ids"]
        mask_idx = batch["mask_idx"]
        target = batch["target"]
        
        pred_mean, pred_logvar = self(numeric, pos_ids, mask_idx)
        loss = self.gaussian_nll_loss(pred_mean, pred_logvar, target)
        
        mse = nn.functional.mse_loss(pred_mean, target)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # Using OneCycleLR can be very helpful
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
