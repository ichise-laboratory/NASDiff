
import pandas as pd
import tsdb
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

from benchpots.utils.logging import logger, print_final_dataset_info
from benchpots.utils.sliding import sliding_window

from data_processing.data_processing_uitls import get_mask, padding
from data_processing.data_processing_uitls import create_missing


def preprocess_italy_air_quality(
    seq_len,
) -> Tuple[dict, dict, dict, dict]:
    """Load and preprocess the dataset Italy Air Quality.

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
        A dictionary containing the processed Italy Air Quality.
    """

    assert seq_len > 0, f"sample_n_steps must be larger than 0, but got {seq_len}"

    data = tsdb.load("italy_air_quality")
    df = data["X"]
    df = df.drop(columns=["Date", "Time"])
    features = df.columns
    df = df.to_numpy()

    # split the dataset into train, validation, and test sets
    all_n_steps = len(df)
    val_test_len = round(all_n_steps * 0.2)
    train_set = df[: -2 * val_test_len]
    val_set = df[-2 * val_test_len : -val_test_len]
    test_set = df[-val_test_len:]

    # Fit normalization statistics on the training split only, then reuse them for val/test.
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set)
    val_set_X = scaler.transform(val_set)
    test_set_X = scaler.transform(test_set)

    train_X_intact = sliding_window(train_set_X, seq_len)
    val_X_intact = sliding_window(val_set_X, seq_len)
    test_X_intact = sliding_window(test_set_X, seq_len)

    # assemble the final processed data into a dictionary
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
