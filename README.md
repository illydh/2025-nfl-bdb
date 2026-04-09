# GhostFormer: Probabilistic Spatiotemporal Modeling of NFL Player Dynamics

## Abstract
This research project introduces **GhostFormer**, a Transformer-based architecture designed for the reconstruction and trajectory prediction of masked defensive agents within high-dimensional sports tracking environments. Leveraging the 2025 NFL Big Data Bowl dataset, we implement a masked language modeling (MLM) approach adapted for continuous spatial sequences. The model treats the set of 22 players as a sequence of spatiotemporal tokens, predicting the position and motion variance of a "ghosted" player leading up to the snap event.

## Key Research Contributions
- **Adaptive Spatiotemporal Attention**: Implementation of multi-head self-attention mechanisms to capture complex inter-player dependencies and defensive alignments.
- **Probabilistic Aleatoric Uncertainty**: Unlike deterministic regression models, GhostFormer utilizes a Gaussian Negative Log Likelihood (NLL) objective to estimate the model's confidence through predicted log-variance parameters.
- **Relative Positional Normalization**: Implementation of a coordinate transformation system that normalizes player positions relative to the dynamic "Line of Scrimmage" midpoint, ensuring model invariance to absolute yardage.
- **Dynamic Masked Augmentation**: A 11-fold data augmentation strategy that dynamically re-samples and masks individual defensive players during the training phase to maximize sample density and generalize spatial relationships.

## Methodology

### 1. Spatial Preprocessing & Normalization
Traditional absolute coordinates are transformed into a relative coordinate system. The origin $(0,0)$ is estimated as the geometric midpoint between the offensive and defensive clusters for each specific frame. This renders the model robust against varied field positions and directional biases.

### 2. The GhostFormer Architecture
The core model is a Transformer Encoder optimized for spatial regression. 
- **Input Encoding**: Concatenation of $(x, y, \text{mask\_flag})$ with Categorical Position Embeddings (e.g., CB, FS, MLB).
- **Encoder Layers**: Multi-layer self-attention blocks with Layer Normalization and GELU activations.
- **Prediction Head**: A multi-output MLP that regresses the bivariate mean ($\mu_x, \mu_y$) and log-variance ($\log \sigma_x^2, \log \sigma_y^2$).

### 3. Objective Function
We optimize using the **Gaussian Negative Log Likelihood (NLL)**:
$$\mathcal{L}(\theta) = \frac{1}{2} \sum \left( \log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2} \right)$$
This approach allows the model to learn to increase its predicted variance in scenarios of high complexity or ambiguity (e.g., pre-snap motion), providing a calibrated measure of model confidence.

## Repository Structure
- `preprocess.py`: High-performance data ingestion utilizing Polars for relative coordinate transformation.
- `dataset.py`: Implementation of the dynamic defensive masking and dataset iteration logic.
- `models.py`: Definition of the Git-Former architecture and LightningModule wrapper.
- `train.py`: Distributed training pipeline with experiment tracking via TensorBoard.
- `evaluate.py`: Statistical evaluation on the Week 9 hold-out set, analyzing MSE and Gaussian NLL metrics.
- `visualize.py`: Synthesis of spatiotemporal trajectories with overlayed confidence ellipses.
- `BDB_Pipeline_Colab.ipynb`: Consolidated research environment for cloud-based replication.

## Experimental Setup & Usage
1. **Prerequisites**: Ensure the `data/` directory contains the NFL BDB 2022 CSV files.
2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Pipeline Execution**:
   ```bash
   python preprocess.py  # Generate serialized tensors
   python train.py       # Execute training routine
   python evaluate.py    # Validate against Week 9
   python visualize.py   # Generate trajectory animation
   ```

## Future Work
- **Temporal Self-Attention**: Extending the spatial transformer to a 3D-Attention mechanism (Spatial + Temporal) to capture time-series evolution more effectively.
- **Grid-based Discrete Tokenization**: Reframing spatial regression as a classification task via field discretization (Spatial Tokenization) to leverage cross-entropy objectives.
- **Offense-Defense Interaction Analysis**: Applying attention-map visualization to quantify the "influence" of specific offensive players on defensive positioning.

## Acknowledgments
Dataset provided by the **NFL Big Data Bowl 2025**. Research inspired by Transformer-based sequence modeling advancements in *Attention is All You Need* (Vaswani et al.).
