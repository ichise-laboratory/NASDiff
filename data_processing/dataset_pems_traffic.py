
import pandas as pd
import tsdb
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

from benchpots.utils.logging import logger, print_final_dataset_info
from benchpots.utils.sliding import sliding_window

from data_processing.data_processing_uitls import get_mask, padding
from data_processing.data_processing_uitls import create_missing


def preprocess_pems_traffic(
    seq_len,
) -> Tuple[dict, dict, dict, dict]:
    """Load and preprocess the dataset PeMS traffic.

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
        A dictionary containing the processed PeMS traffic.
    """

    assert seq_len > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    # read the raw data
    data = tsdb.load("pems_traffic")
    df = data["X"]

    feature_names = df.columns.tolist()
    feature_names.remove("date")
    df["date"] = pd.to_datetime(df["date"])

    unique_months = df["date"].dt.to_period("M").unique()
    selected_as_train = unique_months[:15]  # use the first 15 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[15:19]  # select the following 4 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[
        19:
    ]  # select the left 6 months as test set, 2018-07 has only 2 days, so can be rounded to 5 months
    logger.info(f"months selected as test set are {selected_as_test}")

    test_set = df[df["date"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date"].dt.to_period("M").isin(selected_as_train)]

    # Fit normalization statistics on the training split only, then reuse them for val/test.
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

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



