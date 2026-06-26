
import pandas as pd
import tsdb
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

from benchpots.utils.logging import logger, print_final_dataset_info
from benchpots.utils.sliding import sliding_window

from data_processing.data_processing_uitls import get_mask, padding
from data_processing.data_processing_uitls import create_missing

def preprocess_electricity_load_diagrams(
    seq_len,
) -> Tuple[dict, dict, dict, dict]:
    """Load and preprocess the dataset Electricity Load Diagrams.

    Parameters
    ----------
    rate:
        The missing rate.

    n_steps:
        The number of time steps to in the generated data samples.
        Also the window size of the sliding window.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed Electricity Load Diagrams.

    """

    data = tsdb.load("electricity_load_diagrams")
    df = data["X"]

    feature_names = df.columns.tolist()
    df["datetime"] = pd.to_datetime(df.index)

    unique_months = df["datetime"].dt.to_period("M").unique()
    selected_as_test = unique_months[:10]  # select first 10 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[
        10:20
    ]  # select the 11th - the 20th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[20:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]

    # Fit normalization statistics on the training split only, then reuse them for val/test.
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_X_intact = sliding_window(train_set_X, seq_len)
    val_X_intact = sliding_window(val_set_X, seq_len)
    test_X_intact = sliding_window(test_set_X, seq_len)

    dataset_info = {
        # general info
        "n_steps": seq_len,
        "n_features": train_X_intact.shape[-1],
        "scaler": scaler}

    train_data = dict(
        X_intact=train_X_intact,
    )
    val_data = dict(
        X_intact=val_X_intact,
    )
    test_data = dict(
        X_intact=test_X_intact,
    )

    return dataset_info, train_data, val_data, test_data
