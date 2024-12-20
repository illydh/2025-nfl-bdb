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

import random

# TEMPORARY CHANGE
INPUT_DATA_DIR = Path("./")
OUT_DIR = Path("./drive/My Drive/bdb-2025/split_prepped_data/")


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
    # tracking_dfs = []
    # for week in range(1, 5):
    #     print(f"Load tracking week {week}")
    #     tracking_dfs.append(
    #         pl.read_csv(
    #             INPUT_DATA_DIR / f"tracking_week_{week}.csv",
    #             null_values=["NA", "nan", "N/A", "NaN", ""],
    #         ).filter(pl.col("displayName") != "football")
    #     )
    # return pl.concat(tracking_dfs)
    return pl.read_csv(
        INPUT_DATA_DIR / f"tracking_week_1.csv",
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


def convert_tracking_to_cartesian(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert polar coordinates to Unit-circle Cartesian format.

    Args:
        tracking_df (pl.DataFrame): Tracking data

    Returns:
        pl.DataFrame: Tracking data with Cartesian coordinates.
    """
    return (
        tracking_df.with_columns(
            dir=((pl.col("dir") - 90) * -1) % 360,
            o=((pl.col("o") - 90) * -1) % 360,
        )
        # convert polar vectors to cartesian ((s, dir) -> (vx, vy), (o) -> (ox, oy))
        .with_columns(
            vx=pl.col("s") * pl.col("dir").radians().cos(),
            vy=pl.col("s") * pl.col("dir").radians().sin(),
            ox=pl.col("o").radians().cos(),
            oy=pl.col("o").radians().sin(),
        )
    )


def get_masked_players(tracking_df):
    """
    Randomly selects a player from a game and play, and filters their data from each frame.

    Args:
        tracking_df: The tracking DataFrame.

    Returns:
        tuple: A tuple containing the filtered tracking DataFrame and the masked players DataFrame.
    """
    masked_players_df = tracking_df.filter(pl.col("isDefense") == 1)
    rel_tracking_df = []

    unique_plays = tracking_df.select(["gameId", "playId"]).unique().rows()
    len_plays, cnt = len(unique_plays), 0

    for game_id, play_id in unique_plays:
        cnt += 1
        # The defensive players in a given game + play
        defensive_df = tracking_df.filter(
            (pl.col("gameId") == game_id)
            & (pl.col("playId") == play_id)
            & (pl.col("isDefense") == 1)
        )
        defense_names = defensive_df["displayName"].unique().to_list()
        assert (
            len(defense_names) == 11
        ), "No players found for gameId: {}, playId: {}".format(game_id, play_id)

        # Retrieve masked player
        for selected_player in defense_names:
            # Filter out the selected player from tracking_df
            filtered_tracking_df = tracking_df.filter(
                ~(
                    (pl.col("gameId") == game_id)
                    & (pl.col("playId") == play_id)
                    & (pl.col("displayName") == selected_player)
                )
            )
            rel_tracking_df.append(filtered_tracking_df)

        percent_masked = cnt / len_plays * 100
        if percent_masked % 10 == 0:
            print(f"Masking {percent_masked}% complete")  #   Log %age masked

    print(
        f"rel_tracking_df rows: {len(rel_tracking_df)}\nmasked_players_df rows: {masked_players_df}"
    )
    return pl.concat(rel_tracking_df, how="vertical"), masked_players_df


def split_train_test_val(
    tracking_df: pl.DataFrame, target_df: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Split is 70-15-15 for train-test-val respectively. Notably, we split at the play levle and not frame level.
    This ensures no target contamination between splits.

    Args:
        tracking_df (tuple[pl.DataFrame, pl.DataFrame]): Tracking data
        target_df (tuple[pl.DataFrame, pl.DataFrame]): Target data

    Returns:
        dict: Dictionary containing train, validation, and test dataframes.
    """
    print(
        f"Total set (weeks 1-4): {tracking_df.n_unique(['gameId', 'playId'])} plays,",
        f"{tracking_df.n_unique(['gameId', 'playId', 'frameId'])} frames",
    )
    # Use weeks 1-8 for training
    train_tracking_df = tracking_df
    train_tgt_df = target_df
    print(
        f"Train set 1: {train_tracking_df.n_unique(['gameId', 'playId'])} plays,",
        f"{train_tracking_df.n_unique(['gameId', 'playId', 'frameId'])} frames",
    )
    return {
        "train_features_1": train_tracking_df,
        "train_targets_1": train_tgt_df,
    }


def remove_null_positions(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows with null positions from the tracking data.

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
    filtered_df = tracking_df.join(null_plays, on=["gameId", "playId"], how="anti")
    assert not filtered_df.filter(pl.col("position").is_null()).shape[
        0
    ], "There are still null positions"
    print(f"Lost {len(tracking_df) - len(filtered_df)} rows")

    return filtered_df


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
    print("Convert tracking to cartesian")
    tracking_df = convert_tracking_to_cartesian(tracking_df)

    print("Remove plays with null for positions")
    tracking_df = remove_null_positions(tracking_df)

    print("Generate target for tracking weeks 1-8")
    rel_tracking_df, maskedPlayers_df = get_masked_players(tracking_df)

    # Writing out splits to OUT_DIR
    print("Split train/test/val")
    split_dfs = split_train_test_val(rel_tracking_df, maskedPlayers_df)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    for key, df in split_dfs.items():
        sort_keys = ["gameId", "playId", "frameId"]
        df.sort(sort_keys).write_parquet(out_dir / f"{key}.parquet")


if __name__ == "__main__":
    main()
