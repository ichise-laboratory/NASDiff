
import pandas as pd
import tsdb
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

from benchpots.utils.logging import logger, print_final_dataset_info
from benchpots.utils.sliding import sliding_window

from data_processing.data_processing_uitls import get_mask, padding
from data_processing.data_processing_uitls import create_missing


def preprocess_ett(
    seq_len,
    subset='ETTh1',
) -> Tuple[dict, dict, dict, dict]:
    """Load and preprocess the dataset ETT.

    Parameters
    ----------
    subset:
        The name of the subset dataset to be loaded.
        Must be one of ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'].

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
        A dictionary containing the processed ETT.
    """

    all_subset_names = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
    assert (
        subset in all_subset_names
    ), f"subset_name should be one of {all_subset_names}, but got {subset}"


    data = tsdb.load("electricity_transformer_temperature")  # load all 4 sub datasets
    df = data[subset]
    feature_names = df.columns.tolist()
    df["datetime"] = pd.to_datetime(df.index)

    unique_months = df["datetime"].dt.to_period("M").unique()

    selected_as_train = unique_months[:14]  # use the first 14 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[14:19]  # select the following 5 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[19:]  # select the left 5 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")

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

