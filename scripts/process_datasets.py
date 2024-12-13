# %% [code]
# %% [code]
"""
Dataset Module for NFL Big Data Bowl 2025

This module defines the BDB2025_Dataset class, which is used to load and preprocess
data for training machine learning models.

Classes:
    BDB2025_Dataset: Custom dataset class for NFL tracking data

Functions:
    load_datasets: Load preprocessed datasets for a specific model type and data split
    main: Main execution function for creating and saving datasets

"""

import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

PREPPED_DATA_DIR = Path("./drive/My Drive/bdb-2025/split_prepped_data/")
DATASET_DIR = Path("./drive/My Drive/bdb-2025/working/datasets/")

# Enumerate player positions
POSITIONS_ENUM = {
    'T': 0,
    'CB': 1,
    'G': 2,
    'SS': 3,
    'FS': 4,
    'DB': 5,
    'MLB': 6,
    'LB': 7,
    'OLB': 8,
    'DT': 9,
    'ILB': 10,
    'NT': 11,
    'TE': 12,
    'WR': 13,
    'QB': 14,
    'RB': 15,
    'DE': 16,
    'FB': 17,
    'C': 18
}


class BDB2025_Dataset(Dataset):
    """
    Custom dataset class for NFL tracking data.

    This class preprocesses and stores NFL tracking data for use in machine learning models.

    Attributes:
        model_type (str): Type of model ('transformer')
        keys (list): List of unique identifiers for each data point
        feature_df_partition (pd.DataFrame): Preprocessed feature data
        tgt_df_partition (pd.DataFrame): Preprocessed target data
        tgt_arrays (dict): Precomputed target arrays
        feature_arrays (dict): Precomputed feature arrays
    """

    def __init__(
        self,
        model_type: str,
        feature_df: pl.DataFrame,
        tgt_df: pl.DataFrame,
    ):
        """
        Initialize the dataset.

        Args:
            model_type (str): Type of model ('transformer')
            feature_df (pl.DataFrame): DataFrame containing feature data
            tgt_df (pl.DataFrame): DataFrame containing target data

        Raises:
            ValueError: If an invalid model_type is provided
        """
        if model_type not in ["transformer"]:
            raise ValueError("model_type must be 'transformer'")

        self.model_type = model_type
        self.keys = list(
            feature_df.select(["gameId", "playId", "frameId"]).unique().rows()
        )

        # Convert to pandas form with index for quick row retrieval
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "frameId", "nflId"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "frameId"])
            .sort_index()
        )

        # Precompute features and store in dicts
        self.tgt_arrays: dict[tuple, np.ndarray] = {}
        self.feature_arrays: dict[tuple, np.ndarray] = {}
        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            results = pool.map(
                self.process_key,
                tqdm(
                    self.keys,
                    desc="Pre-computing feature transforms",
                    total=len(self.keys),
                ),
            )
            # Unpack results
            for key, tgt_array, feature_array in results:
                self.tgt_arrays[key] = tgt_array
                self.feature_arrays[key] = feature_array

    def process_key(self, key: tuple) -> tuple[tuple, np.ndarray, np.ndarray]:
        """
        Process a single key to generate target and feature arrays.

        Args:
            key (tuple): Key (gameId, playId, mirrored, frameId) identifying a specific data point

        Returns:
            tuple[tuple, np.ndarray, np.ndarray]: Processed key, target array, and feature array
        """
        tgt_array = self.transform_target_df(self.tgt_df_partition.loc[key])
        feature_array = self.transform_input_frame_df(
            self.feature_df_partition.loc[key]
        )
        return key, tgt_array, feature_array

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature array and target array for the specified index

        Raises:
            IndexError: If the index is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        key = self.keys[idx]
        return self.feature_arrays[key], self.tgt_arrays[key]

    def transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame to numpy array based on model type.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features

        Raises:
            ValueError: If an unknown model type is specified
        """

        return self.transformer_transform_input_frame_df(frame_df)

    def transform_target_df(self, tgt_df: pd.DataFrame) -> np.ndarray:
        """
        Transform target DataFrame to numpy array.
        Args:
            tgt_df (pd.DataFrame): Target DataFrame
        Returns:
            np.ndarray: Transformed target values as one-hot encoded array
        """
        # Create one-hot encoding with prefix
        ohe_tgt = pd.get_dummies(tgt_df["position"], prefix="position")

        # Ensure all formation types are present
        expected_columns = [
            f"position_{position}" for position in sorted(POSITIONS_ENUM.keys())
        ]
        for col in expected_columns:
            if col not in ohe_tgt.columns:
                ohe_tgt[col] = 0

        # Sort columns
        ohe_tgt = ohe_tgt[expected_columns].to_numpy()[0].astype(np.float32)

        x_tgt, y_tgt = np.array(tgt_df["x"]).flatten(), np.array(tgt_df["y"]).flatten()

        tgts = [ohe_tgt, x_tgt, y_tgt]
        y = np.array([np.concatenate(tgts, dtype=np.float32)])
        # assert y.shape == (1, len(tgts)), f"Expected y.shape (1, {len(tgts)}), got {y.shape}"
        return y

    def transformer_transform_input_frame_df(
        self, frame_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform input frame DataFrame for transformer model.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for transformer model

        Raises:
            AssertionError: If the output shape is not as expected
        """
        # Create one-hot encoding with prefix
        ohe_feature = pd.get_dummies(frame_df["position"], prefix="position")
        # Ensure all formation types are present
        expected_columns = [
            f"position_{position}" for position in sorted(POSITIONS_ENUM.keys())
        ]
        for col in expected_columns:
            if col not in ohe_feature.columns:
                ohe_feature[col] = 0

        # Sort columns
        ohe_feature = ohe_feature[expected_columns].to_numpy().astype(np.float32)
        x_feature, y_feature = (
            np.array(frame_df["x"]).flatten(),
            np.array(frame_df["y"]).flatten(),
        )

        assert len(frame_df) == 21, "NOT THE SAME LEN"

        # Reshape features to have the same first dimension
        ohe_feature = ohe_feature.reshape(len(frame_df), -1)
        x_feature = x_feature.reshape(len(frame_df), 1)
        y_feature = y_feature.reshape(len(frame_df), 1)

        features = [ohe_feature, x_feature, y_feature]
        x = np.concatenate(features, dtype=np.float32, axis=1)

        # assert x.shape == (21, len(features)), f"Expected x.shape (21, {len(features)}), got {x.shape}"
        return x


def load_datasets(model_type: str, split: str) -> BDB2025_Dataset:
    """
    Load datasets for a specific model type and data split.

    Args:
        model_type (str): Type of model ('transformer' or 'zoo')
        split (str): Data split ('train', 'val', or 'test')

    Returns:
        BDB2025_Dataset: Loaded dataset for the specified model type and split

    Raises:
        ValueError: If an unknown split is specified
        FileNotFoundError: If the dataset file is not found
    """
    ds_dir = DATASET_DIR / model_type
    file_path = ds_dir / f"{split}_dataset.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def main():
    """
    Main function to create and save datasets for different model types and splits.
    """
    for split in ["test", "val", "train"]:
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        for model_type in ["transformer"]:
            print(f"Creating dataset for {model_type=}, {split=}...")
            tic = time.time()
            dataset = BDB2025_Dataset(model_type, feature_df, tgt_df)
            out_dir = DATASET_DIR / model_type
            out_dir.mkdir(exist_ok=True, parents=True)
            with open(out_dir / f"{split}_dataset.pkl", "wb") as f:
                pickle.dump(dataset, f)
            print(f"Took {(time.time() - tic)/60:.1f} mins")


if __name__ == "__main__":
    main()
