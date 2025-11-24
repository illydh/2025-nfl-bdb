# %% [code]

"""
Model Architectures for NFL Big Data Bowl 2025 prediction task

This module defines neural network architectures for predicting ouptuts in NFL Plays. It includes a generalized transformer-based model for
processing sports tracking data and a LightningModule wrapper for training and evaluation.

Classes:
    SportsTransformer: Generalized Transformer-based model for sports tracking data.
    SportsTransformerLitModel: LightningModule wrapper for shared training functionality.
"""

from typing import Any
import torch
from torch import Tensor, nn, squeeze
from torch.optim import AdamW
from pytorch_lightning import LightningModule

torch.set_float32_matmul_precision("medium")


class GhostFormer(nn.Module):
    """Transformer model for predicting masked player position and coordinates."""

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 7,
        dropout: float = 0.3,
        num_positions: int = 19,  # Number of possible positions
    ):
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))

        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        self.feature_norm_layer = nn.BatchNorm1d(feature_len)
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        # Separate decoders for position and coordinates
        self.position_decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, num_positions),  # Predict position index
        )

        self.coordinate_decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, 2),  # Predict x and y coordinates
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, P, F = x.size()

        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.feature_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)

        position_logits = self.position_decoder(x)  # [B, num_positions]
        coordinates = self.coordinate_decoder(x)  # [B, 2]

        return position_logits, coordinates


class GhostFormerLitModel(LightningModule):
    def __init__(
        self,
        feature_len: int,
        batch_size: int,
        model_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        num_positions: int = 19,
    ):
        super().__init__()
        self.feature_len = feature_len
        self.num_positions = num_positions
        self.model = GhostFormer(
            feature_len=self.feature_len,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_positions=self.num_positions,
        )
        self.example_input_array = torch.randn((batch_size, 21, self.feature_len))
        self.learning_rate = learning_rate
        self.num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.hparams["params"] = self.num_params
        for k, v in self.model.hyperparams.items():
            self.hparams[k] = v
        self.save_hyperparameters()
        self.position_loss_fn = torch.nn.CrossEntropyLoss()
        self.coordinate_loss_fn = torch.nn.MSELoss()  # Or L1Loss

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(x, list):
            x = torch.stack(x)
        return self.model(x)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = batch
        position_target = y[:, 0].long()  # Extract position (batch_size,)
        coordinate_target = y[:, 1:]  # Extract coordinates (batch_size, 2)
        position_logits, coordinates = self.model(x)
        position_loss = self.position_loss_fn(position_logits, position_target)
        coordinate_loss = self.coordinate_loss_fn(coordinates, coordinate_target)
        # Scaling factors (adjust these based on your data and initial experiments)
        position_scale = 1.0  # You might not need to scale cross entropy as much
        coordinate_scale = (
            0.01  # Example if your MSE is much smaller. Adjust as needed.
        )
        scaled_position_loss = position_loss * position_scale
        scaled_coordinate_loss = coordinate_loss * coordinate_scale

        loss = scaled_position_loss + scaled_coordinate_loss
        self.log(
            "train_position_loss",
            position_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_coordinate_loss",
            coordinate_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = batch
        position_target = y[:, 0].long()  # Extract position (batch_size,)
        coordinate_target = y[:, 1:]  # Extract coordinates (batch_size, 2)
        position_logits, coordinates = self.model(x)
        position_loss = self.position_loss_fn(position_logits, position_target)
        coordinate_loss = self.coordinate_loss_fn(coordinates, coordinate_target)
        # Scaling factors (adjust these based on your data and initial experiments)
        position_scale = 1.0  # You might not need to scale cross entropy as much
        coordinate_scale = (
            0.01  # Example if your MSE is much smaller. Adjust as needed.
        )
        scaled_position_loss = position_loss * position_scale
        scaled_coordinate_loss = coordinate_loss * coordinate_scale

        loss = scaled_position_loss + scaled_coordinate_loss
        self.log(
            "val_position_loss",
            position_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_coordinate_loss",
            coordinate_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor]:
        x, _ = batch  # We don't need targets for prediction
        if isinstance(x, list):
            x = torch.stack(x)
        position_logits, coordinates = self.model(x)
        return position_logits, coordinates

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate)
