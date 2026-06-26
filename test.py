import os
import time

import pandas as pd
import torch
import yaml
from pypots.nn.functional import calc_mae, calc_mre, calc_mse

from data_processing.data_base import Dataset_Base
from utils.log import My_logger
from utils.random_seed import setup_seed
from utils.result_processing import Result_base
from utils.device import resolve_device
from utils.quiet import suppress_third_party_output
from utils.train_test_func import get_dataloader, get_model_dataset_class, get_net_object


def test_val_one_trail(db, net, stage, logger, exp_config, console=False, console_prefix=None, log_result=True):
    result_base = Result_base()
    assert stage in ["val", "test"], f'stage should be either "val" or "test" but got {stage}!'

    test_sampling_times = exp_config["test"].get("test_sampling_times", 1)
    kwargs = dict(stage=stage)
    if test_sampling_times != 1:
        kwargs["test_sampling_times"] = test_sampling_times

    loader = db.val_loader if stage == "val" else db.test_loader
    net.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        if db.DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=db.DEVICE)
            torch.cuda.synchronize(device=db.DEVICE)

        start = time.time()
        for _, input in enumerate(loader):
            output = net.predict(inputs=input, **kwargs)

            if "imputed_result" in output:
                result_base.record_epoch_data(input=input, output=output["imputed_result"])
            else:
                epoch_loss += output["val_loss"]

        if db.DEVICE.type == "cuda":
            torch.cuda.synchronize(device=db.DEVICE)
        time_consume = time.time() - start

        if "imputed_result" in output:
            MIT_mae, MIT_mse, MIT_mre = result_base.cal_epoch_loss(stage=stage)
            result_base.print_result(
                MIT_mae=MIT_mae,
                MIT_mse=MIT_mse,
                MIT_mre=MIT_mre,
                time_consume=time_consume,
                stage=stage,
                logger=logger,
                console=console,
                log_result=log_result,
                console_prefix=console_prefix,
            )
        else:
            logger.debug(f"Validation epoch loss: {round(epoch_loss, 4)}", pos="blank")
            if console:
                prefix = f"{console_prefix} " if console_prefix else ""
                print(f"{prefix}{stage.upper()}: Loss:{round(epoch_loss, 4)}")

        if stage == "test":
            return MIT_mae, MIT_mse, MIT_mre, round(time_consume, 4)
        if "imputed_result" in output:
            return MIT_mae, MIT_mse, MIT_mre
        return epoch_loss, None, None


def test_with_trained_model(
    exp_config,
    config_root,
    dataset_config,
    models_root,
    logger,
    file_name: str,
    saved_root="./saved_model",
    test_dataset_info: dict = None,
):
    parts = file_name.split(" ")
    infos = dict(
        dataset=parts[0],
        model_name=parts[1],
        padding_mode=parts[2],
        seed=parts[3].replace("seed", ""),
        train_pattern=parts[6],
        train_missing_ratio=parts[7].replace(".pt", ""),
    )

    if "/" in infos["dataset"]:
        infos["dataset"] = infos["dataset"].split("/")[-1]

    use_best_config = exp_config["global"]["use_best_config"]
    if use_best_config:
        path = os.path.join(config_root, infos["model_name"] + "_best.yaml")
        if not os.path.exists(path):
            path = os.path.join(config_root, infos["model_name"] + ".yaml")
    else:
        path = os.path.join(config_root, infos["model_name"] + ".yaml")

    with open(path, "r") as f:
        model_config = yaml.safe_load(f)

    if test_dataset_info is not None:
        assert test_dataset_info.get("pattern") is not None, "test_dataset_info requires pattern"
        assert test_dataset_info.get("missing_ratio") is not None, "test_dataset_info requires missing_ratio"
        exp_config["test"]["pattern"] = test_dataset_info["pattern"]
        exp_config["test"]["missing_ratio"] = test_dataset_info["missing_ratio"]

    DEVICE = resolve_device(device=exp_config["global"].get("DEVICE"))
    try:
        exp_config["global"]["padding_mode"] = float(infos["padding_mode"])
    except ValueError:
        pass
    exp_config["train"]["pattern"] = infos["train_pattern"]
    exp_config["train"]["missing_ratio"] = float(infos["train_missing_ratio"])
    setup_seed(seed=int(infos["seed"]) if infos["seed"].isdigit() else 30)

    with suppress_third_party_output():
        Net_Class, Dataset_Class = get_model_dataset_class(models_root, infos["model_name"])
    logger.info(f"Testing checkpoint: {file_name}", pos="head")

    with suppress_third_party_output():
        db = get_dataloader(
            model_config=model_config,
            dataset_config=dataset_config,
            exp_config=exp_config,
            dataset=infos["dataset"],
            Dataset_Class=Dataset_Class,
            DEVICE=DEVICE,
        )
        net = get_net_object(
            model_config=model_config,
            dataset=infos["dataset"],
            net_class=Net_Class,
            seq_len=db.seq_len,
            d_feature=db.d_feature,
            DEVICE=DEVICE,
        )

    state_dict = torch.load(
        os.path.join(saved_root, infos["model_name"], file_name),
        map_location=DEVICE,
        weights_only=True,
    )
    net.load_state_dict(state_dict)
    MIT_mae, MIT_mse, MIT_mre, time_consume = test_val_one_trail(
        db=db,
        net=net,
        stage="test",
        logger=logger,
        exp_config=exp_config,
        console=True,
    )
    return MIT_mae, MIT_mse, MIT_mre, time_consume


def test_naive(model_name: str, dataset_config, exp_config, logger):
    assert model_name in ["MEAN", "LINEAR", "SAMPLE_MEAN"]
    datasets = exp_config["train"]["datasets"]
    seeds = exp_config["global"]["seeds"]
    exp_config["global"]["padding_mode"] = model_name.lower()
    DEVICE = resolve_device(device=exp_config["global"].get("DEVICE"))

    logger.info(
        f"Naive method: {model_name}\r\n"
        f"Datasets: {datasets}\r\n"
        f"Seeds: {seeds}",
        pos="head",
    )

    overall_result = dict()
    for dataset in datasets:
        dataset_results = pd.DataFrame(columns=["MAE", "MSE", "MRE", "Time_Usage"], index=seeds)

        for seed in seeds:
            setup_seed(seed)
            model_config = {dataset: {"setting": {"BATCHSIZE": 32}}}
            with suppress_third_party_output():
                db = get_dataloader(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    exp_config=exp_config,
                    dataset=dataset,
                    Dataset_Class=Dataset_Base,
                    DEVICE=DEVICE,
                )

            imputed_result = db.test_dataset.data_dict["X"]
            X_intact = db.test_dataset.data_dict["X_intact"]
            indicating_mask = db.test_dataset.data_dict["indicating_mask"]

            MIT_mae = calc_mae(imputed_result, X_intact, indicating_mask).item()
            MIT_mse = calc_mse(imputed_result, X_intact, indicating_mask).item()
            MIT_mre = calc_mre(imputed_result, X_intact, indicating_mask).item()

            logger.debug(
                f"{model_name} on {dataset}: MAE={round(MIT_mae, 4)}, "
                f"MSE={round(MIT_mse, 4)}, MRE={round(MIT_mre, 4)}",
                pos="blank",
            )

            dataset_results.loc[seed] = dict(MAE=MIT_mae, MSE=MIT_mse, MRE=MIT_mre, Time_Usage=None)

        overall_result[dataset] = dataset_results.to_string()

    logger.info(My_logger.format_message_multicolumn(overall_result, content="results"), pos="key")
    logger.info("Run complete.", pos="tail")
