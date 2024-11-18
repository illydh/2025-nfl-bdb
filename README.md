# Predicting Motion of a Masked Defensive Player Using Transformers

## Overview

This project applies transformer models to predict the motion of a masked defensive player in NFL player tracking data. It uses sequential data preprocessing, masking techniques, and deep learning to understand player dynamics and interactions.

## Goals

- Predict the x and y positions of a masked defensive player.
- Analyze player interactions and dependencies using transformer attention mechanisms.

## Approach

1. Preprocess the data:
    - Normalize player positions relative to play direction.
    - Mask the target defensive player's motion data.
    - Create sequences of player movements over time.
2. Train a transformer model:
    - Inputs: Player features as sequences with positional encodings.
    - Outputs: Predicted x, y coordinates of the masked player.
3. Evaluate performance:
    - Metrics: Mean Squared Error (MSE) for position prediction.

## How to Run
1. **Download the dataset** from [Kaggle NFL Big Data Bowl](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data).
    The compressed dataset can be downloaded directly from terminal using the following commands:
    ```
    pip install kaggle
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle competitions download nfl-big-data-bowl-2025
    unzip path/to/file/nfl-big-data-bowl-2025.zip'
    ```
    *Note: `kaggle.json` is a API token created from Kaggle creator accounts and is necessary to access Kaggle commands. Directions to downloading one can be found [here](https://www.kaggle.com/docs/api).*
2. Place the dataset in the same directory that this repository is cloned in.
3. Preprocess the data and train the model.
4. Evaluate the model.


## Acknowledgments

[1^]: NFL Big Data Bowl for providing the dataset.
[2^]: Transformer model architecture inspired by the original paper by Vaswani et al., *Attention is All You Need*.
