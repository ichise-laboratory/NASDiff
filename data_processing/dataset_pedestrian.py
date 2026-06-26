
import tsdb
from typing import Union, Tuple
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from benchpots.utils.logging import logger, print_final_dataset_info

from data_processing.data_processing_uitls import get_mask, padding
from data_processing.data_processing_uitls import create_missing


def preprocess_ucr_uea_datasets(
    seq_len=None,
    dataset_name: str = 'ucr_uea_MelbournePedestrian',
) -> Tuple[dict, dict, dict, dict]:
    """Load and preprocess the dataset from UCR&UEA.

    Parameters
    ----------
    dataset_name:
        The name of the UCR_UEA dataset to be loaded. Must start with 'ucr_uea_'.
        Use tsdb.list() to get all available datasets. `

    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed UCR&UEA dataset.

    """

    assert dataset_name.startswith(
        "ucr_uea_"
    ), f"set_name must start with 'ucr_uea_', but got {dataset_name}"
    assert dataset_name in tsdb.list(), f"{dataset_name} is not in TSDB database."

    data = tsdb.load(dataset_name)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    le = None
    if is_string_dtype(y_train):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    n_X_train = len(X_train)

    train_ids, val_ids = train_test_split(list(range(n_X_train)), test_size=0.2)
    X_train, X_val = X_train[train_ids], X_train[val_ids]
    y_train, y_val = y_train[train_ids], y_train[val_ids]

    X_train_shape = X_train.shape
    X_val_shape = X_val.shape
    X_test_shape = X_test.shape

    X_train = X_train.reshape(X_train_shape[0], -1)
    X_val = X_val.reshape(X_val_shape[0], -1)
    X_test = X_test.reshape(X_test_shape[0], -1)
    # Fit normalization statistics on the training split only, then reuse them for val/test.
    scaler = StandardScaler()
    train_X = scaler.fit_transform(X_train)
    val_X = scaler.transform(X_val)
    test_X = scaler.transform(X_test)

    train_X_intact = train_X.reshape(X_train_shape).transpose(0, 2, 1)
    val_X_intact = val_X.reshape(X_val_shape).transpose(0, 2, 1)
    test_X_intact = test_X.reshape(X_test_shape).transpose(0, 2, 1)

    # assemble the final processed data into a dictionary
    dataset_info = {
        # general info
        "n_steps": train_X.shape[1],
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
