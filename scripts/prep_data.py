# %% [code]
"""
Data Preparation Module for NFL Big Data Bowl 2025

This module processes raw NFL tracking data to prepare it for machine learning models.
It includes functions for loading, cleaning, and transforming the data, as well as
splitting it into train, validation, and test sets.

Functions:
    get_players_df: Load and preprocess player data
    get_plays_df: Load and preprocess play data
    get_tracking_df: Load and preprocess tracking data
    add_features_to_tracking_df: Add derived features to tracking data
    convert_tracking_to_cartesian: Convert polar coordinates to Cartesian
    get_masked_players: Generate target dataframe maskedPlayers prediction
    split_train_test_val: Split data into train, validation, and test sets
    main: Main execution function

"""
from argparse import ArgumentParser
from pathlib import Path

import polars as pl


# TEMPORARY CHANGE
INPUT_DATA_DIR = Path("./")
SPLIT_OUT_DIR = Path("./drive/My Drive/bdb-2025/prepped_data/split")
SPLIT_OUT_DIR.mkdir(exist_ok=True, parents=True)
TRACKING_OUT_DIR = Path("./drive/My Drive/bdb-2025/prepped_data/tracking")
TRACKING_OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_players_df() -> pl.DataFrame:
    """
    Load player-level data and preprocesses features.

    Returns:
        pl.DataFrame: Preprocessed player data with additional features.
    """
    return (
        pl.read_csv(
            INPUT_DATA_DIR / "players.csv", null_values=["NA", "nan", "N/A", "NaN", ""]
        )
        .with_columns(
            height_inches=(
                pl.col("height")
                .str.split("-")
                .map_elements(lambda s: int(s[0]) * 12 + int(s[1]), return_dtype=int)
            )
        )
        .with_columns(
            weight_Z=(pl.col("weight") - pl.col("weight").mean())
            / pl.col("weight").std(),
            height_Z=(pl.col("height_inches") - pl.col("height_inches").mean())
            / pl.col("height_inches").std(),
        )
    )


def get_plays_df() -> pl.DataFrame:
    """
    Load play-level data and preprocesses features.

    Returns:
        pl.DataFrame: Preprocessed play data with additional features.
    """
    return pl.read_csv(
        INPUT_DATA_DIR / "plays.csv", null_values=["NA", "nan", "N/A", "NaN", ""]
    ).with_columns(
        distanceToGoal=(
            pl.when(pl.col("possessionTeam") == pl.col("yardlineSide"))
            .then(100 - pl.col("yardlineNumber"))
            .otherwise(pl.col("yardlineNumber"))
        )
    )


def get_tracking_df() -> pl.DataFrame:
    """
    Load tracking data and preprocesses features. Notably, exclude rows representing the football's movement.

    Returns:
        pl.DataFrame: Preprocessed tracking data with additional features.
    """
    # don't include football rows for this project.
    # NOTE: Only processing week 1 for the sake of time.  Change "1" to "*" to process all weeks
    return pl.read_csv(
        INPUT_DATA_DIR / "tracking_week_1.csv",
        null_values=["NA", "nan", "N/A", "NaN", ""],
    ).filter(pl.col("displayName") != "football")


def add_features_to_tracking_df(
    tracking_df: pl.DataFrame,
    players_df: pl.DataFrame,
    plays_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Consolidates play and player level data into the tracking data.

    Args:
        tracking_df (pl.DataFrame): Tracking data
        players_df (pl.DataFrame): Player data
        plays_df (pl.DataFrame): Play data

    Returns:
        pl.DataFrame: Tracking data with additional features.
    """
    # add `is_ball_carrier`, `team_indicator`, and other features to tracking data
    og_len = len(tracking_df)
    tracking_df = (
        tracking_df.join(
            plays_df.select("gameId", "playId", "defensiveTeam"),
            on=["gameId", "playId"],
            how="inner",
        )
        .join(
            players_df.select(
                ["nflId", "displayName", "position"]
            ).unique(),  # select position column
            on=["nflId", "displayName"],
            how="left",
        )
        # .join(
        #    players_df.select(["nflId", "weight_Z", "height_Z"]).unique(),
        #    on="nflId",
        #    how="inner",
        # )
        .with_columns(
            isDefense=pl.when(pl.col("club") == pl.col("defensiveTeam"))
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias("isDefense"),
        )
        .drop(["defensiveTeam"])
    )

    assert (
        len(tracking_df) == og_len
    ), "Lost rows when joining tracking data with play/player data"

    return tracking_df


def remove_null_positions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows with null positions.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with null positions removed.
    """
    # Players with null positions
    null_position_players = tracking_df.filter(pl.col("position").is_null())

    # Identify unique gameId and playId combinations with null positions
    null_plays = null_position_players.select(["gameId", "playId"]).unique()

    # Filter out the plays with null positions from the tracking data
    return tracking_df.join(null_plays, on=["gameId", "playId"], how="anti")


def augment_data(tracking_df):
    """
    Augments tracking data by iterating through each game and play,
    removing one defensive player at a time and creating a new DataFrame.
    Saves each play to a parquet file.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Augmented tracking data
    """
    # Assuming 'tracking_df' is your Polars DataFrame
    # and it has columns 'gameId', 'playId', 'x', 'y'
    for game_id, play_id in tracking_df.select(["gameId", "playId"]).unique().rows():
        defensive_players = tracking_df.filter(
            (pl.col("gameId") == game_id)
            & (pl.col("playId") == play_id)
            & (pl.col("isDefense") == 1)
        )
        unique_names = defensive_players["displayName"].unique().to_list()

        rel_tracking_df = []

        for player_name in unique_names:
            # Filter out the selected player from tracking_df
            # filtered_df = tracking_df.filter(
            #         (pl.col("gameId") == game_id)
            #         & (pl.col("playId") == play_id)
            #         & (pl.col("displayName") != player_name)
            # )
            filtered_df = tracking_df.filter(
                (pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
            ).with_columns(
                pl.when(pl.col("displayName") == player_name)
                .then(pl.lit("MASKED"))
                .otherwise(pl.col("position"))
                .alias("position"),
                pl.when(pl.col("displayName") == player_name)
                .then(pl.lit(-1))
                .otherwise(pl.col("x"))
                .alias("x"),
                pl.when(pl.col("displayName") == player_name)
                .then(pl.lit(-1))
                .otherwise(pl.col("y"))
                .alias("y"),
            )

            # Assert that the masked player's position is now "MASKED"
            masked_player_df = filtered_df.filter(pl.col("displayName") == player_name)
            assert masked_player_df["position"].unique().to_list() == [
                "MASKED"
            ], f"Masked player {player_name} position is not 'MASKED'"
            rel_tracking_df.append(filtered_df)

        # Concatenate all DataFrames in the list
        rel_tracking_df = pl.concat(rel_tracking_df)

        # Select only the specified columns
        rel_tracking_df = rel_tracking_df.select(
            ["gameId", "playId", "frameId", "displayName", "position", "x", "y"]
        )

        # Save the DataFrame to the specified directory
        rel_tracking_df.write_parquet(
            TRACKING_OUT_DIR / f"game_{game_id}_play_{play_id}.parquet"
        )


def get_target_variable(tracking_df):
    """
    Get the target variable for the model.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Target variable
    """
    return tracking_df.filter(pl.col("isDefense") == 1).select(
        ["gameId", "playId", "frameId", "displayName", "position", "x", "y"]
    )


def split_train_test_val(target_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Split is 70-15-15 for train-test-val respectively. Notably, we split at the play levle and not frame level.
    This ensures no target contamination between splits.

    Args:
        tracking_df (pl.DataFrame): Tracking data
        target_df (pl.DataFrame): Target data

    Returns:
        dict: Dictionary containing train, validation, and test dataframes.
    """
    target_df = target_df.sort(["gameId", "playId"])

    test_val_ids = (
        target_df.select(["gameId", "playId"])
        .unique(maintain_order=True)
        .sample(fraction=0.3, seed=42)
    )
    train_tgt_df = target_df.join(test_val_ids, on=["gameId", "playId"], how="anti")
    train_ids = train_tgt_df.select(["gameId", "playId"]).unique(maintain_order=True)
    train_tracking_df = [
        pl.read_parquet(TRACKING_OUT_DIR / f"game_{game_id}_play_{play_id}.parquet")
        for game_id, play_id in train_ids.rows()
    ]
    train_tracking_df = pl.concat(train_tracking_df)
    print(
        f"Train set: {train_tracking_df.n_unique(['gameId', 'playId'])} plays,",
        f"{train_tracking_df.n_unique(['gameId', 'playId', 'frameId'])} frames",
    )

    test_ids = test_val_ids.sample(fraction=0.5, seed=42)  # 70-15-15 split
    test_tgt_df = target_df.join(test_ids, on=["gameId", "playId"], how="inner")
    test_tracking_df = [
        pl.read_parquet(TRACKING_OUT_DIR / f"game_{game_id}_play_{play_id}.parquet")
        for game_id, play_id in test_ids.rows()
    ]
    test_tracking_df = pl.concat(test_tracking_df)
    print(
        f"Test set: {test_tracking_df.n_unique(['gameId', 'playId'])} plays,",
        f"{test_tracking_df.n_unique(['gameId', 'playId', 'frameId'])} frames",
    )

    val_ids = test_val_ids.join(test_ids, on=["gameId", "playId"], how="anti")
    val_tgt_df = target_df.join(val_ids, on=["gameId", "playId"], how="inner")
    val_tracking_df = [
        pl.read_parquet(TRACKING_OUT_DIR / f"game_{game_id}_play_{play_id}.parquet")
        for game_id, play_id in val_ids.rows()
    ]
    val_tracking_df = pl.concat(val_tracking_df)
    print(
        f"Validation set: {val_tracking_df.n_unique(['gameId', 'playId'])} plays,",
        f"{val_tracking_df.n_unique(['gameId', 'playId','frameId'])} frames",
    )

    len_plays_tracking_df = (
        train_tracking_df.n_unique(["gameId", "playId"])
        + test_tracking_df.n_unique(["gameId", "playId"])
        + val_tracking_df.n_unique(["gameId", "playId"])
    )
    len_frames_tracking_df = (
        train_tracking_df.n_unique(["gameId", "playId", "frameId"])
        + test_tracking_df.n_unique(["gameId", "playId", "frameId"])
        + val_tracking_df.n_unique(["gameId", "playId", "frameId"])
    )
    print(
        f"Total set: {len_plays_tracking_df} plays,",
        f"{len_frames_tracking_df} frames",
    )

    return {
        "train_features": train_tracking_df,
        "train_targets": train_tgt_df,
        "test_features": test_tracking_df,
        "test_targets": test_tgt_df,
        "val_features": val_tracking_df,
        "val_targets": val_tgt_df,
    }


def main():
    """
    Main execution function for data preparation.

    This function orchestrates the entire data preparation process, including:
    1. Loading raw data
    2. Adding features and transforming coordinates
    3. Generating target variables
    4. Splitting data into train, validation, and test sets
    5. Saving processed data to parquet files
    """
    # Load in raw data
    print("Load players")
    players_df = get_players_df()
    print("Load plays")
    plays_df = get_plays_df()
    print("Load tracking")
    tracking_df = get_tracking_df()
    print("tracking_df rows:", len(tracking_df))
    print("Add features to tracking")
    tracking_df = add_features_to_tracking_df(tracking_df, players_df, plays_df)
    del players_df
    print("Remove null positions")
    tracking_df = remove_null_positions(tracking_df)
    print("tracking_df rows:", len(tracking_df))

    print(f"Augment tracking data")
    augment_data(tracking_df)

    print("Generate target - maskedPlayers")
    maskedPlayers_df = get_target_variable(tracking_df)
    maskedPlayers_df

    print("Split train/test/val")
    split_dfs = split_train_test_val(maskedPlayers_df)

    for key, df in split_dfs.items():
        sort_keys = ["gameId", "playId", "frameId"]
        df.sort(sort_keys).write_parquet(SPLIT_OUT_DIR / f"{key}.parquet")


if __name__ == "__main__":
    main()
