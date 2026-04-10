# GhostFormer NFL Spatial Tracker (BDB 2025)

A deep learning project leveraging spatiotemporal Transformers to reconstruct player movements in NFL tracking data. Using a masked-player approach, the model predicts the trajectory of a specific player during the pre-snap window based on the physical dynamics of the other 21 players on the field.

https://github.com/user-attachments/assets/2e796baf-956c-4c6c-a811-34b34096b1e7



## Project Goals
- **Motion Reconstruction**: Predict where a specific defensive player *should* be given the alignment and motion of the rest of the field.
- **Spatiotemporal Dependency**: Explore how player movements respond to each other chronologically leading up to the snap.
- **Confidence Calibration**: Visualize predictive uncertainty using log-variance entropy proxies.

## The "GhostFormer" Pipeline

### 1. Preprocessing (`preprocess.py`)
Parses raw Kaggle NFL tracking data. It enforces strict player counts (22 players + ball removal) and normalizes coordinates relative to the line-of-scrimmage midpoint.

### 2. Dataset & Masking (`dataset.py`)
Generates 11 unique training variations per frame by masking out each defensive player. It organizes frames into sequential plays (Spatiotemporal blocks) for the model to learn chronological patterns.

### 3. Architecture (`models.py`)
- **Spatial Transformer**: Analyzes inter-player relationships within a single frame.
- **Temporal Transformer**: Analyzes the masked player's historical motion over the play.
- **Discrete Grid Head**: Classifies positions into a high-resolution field grid (120x60) for stable spatial convergence.

### 4. Training & Evaluation (`train.py`, `evaluate.py`)
Uses PyTorch Lightning for distributed training and evaluates performance on the Week 9 hold-out set using Mean Squared Error (MSE) mapped back to physical yardage.

## How to Run

1. **Initial Setup**: Install dependencies and place your BDB CSV files in the `data/` directory.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   python preprocess.py  # Prepare tensors
   python train.py       # Train the GhostFormer
   python evaluate.py    # Run stats on Week 9
   python visualize.py   # Generate a new visualization clip
   ```

3. **Colab Version**: For cloud execution (with GPU acceleration), use the provided `notebooks/BDB_Pipeline_Colab.ipynb`.

## Project Structure
- `data/`: Raw CSV data and processed `.pt` tensors.
- `notebooks/`: Consolidated Jupyter environment for cloud deployment.
- `outputs/`: Generated visualizations and model exports.
- `checkpoints/`: Model weights saved during training.

---
*Developed for the 2025 NFL Big Data Bowl.*
