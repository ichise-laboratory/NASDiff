import pandas as pd
import time
from tqdm import trange

from test import test_val_one_trail
from utils.early_stop import EarlyStop
from utils.log import My_logger
from utils.random_seed import setup_seed
from utils.result_processing import Result_base
from utils.save_model import save_model
from utils.device import resolve_device
from utils.quiet import suppress_third_party_output
from utils.train_test_func import get_dataloader, get_log_info, get_net_object, get_optim


def train_multiple(exp_config, dataset_config, Net_Class, Dataset_Class, model_config, logger):
    DEVICE = resolve_device(device=exp_config["global"]["DEVICE"])

    logger.info(
        f'Training mode: {exp_config["global"]["mode"]}\r\n'
        f'Training datasets: {exp_config["train"]["datasets"]}\r\n'
        f'Training seeds: {exp_config["global"]["seeds"]}\r\n'
        f'Missing pattern: {exp_config["train"]["pattern"]}\r\n'
        f'Missing ratio: {exp_config["train"]["missing_ratio"]}\r\n'
        f'Using device: {DEVICE}\r\n'
        f'Early-stop patience: {exp_config["global"]["patience"]}',
        pos="head",
    )

    datasets = exp_config["train"]["datasets"]
    seeds = exp_config["global"]["seeds"]
    overall_result = dict()

    try:
        for dataset in datasets:
            dataset_results = pd.DataFrame(columns=["MAE", "MSE", "MRE", "Time_Usage"], index=seeds)

            for seed in seeds:
                setup_seed(seed)
                print(f"Preparing dataset={dataset}, seed={seed} on device={DEVICE}...", flush=True)
                with suppress_third_party_output():
                    db = get_dataloader(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        exp_config=exp_config,
                        dataset=dataset,
                        Dataset_Class=Dataset_Class,
                        DEVICE=DEVICE,
                    )
                    net = get_net_object(
                        model_config=model_config,
                        dataset=dataset,
                        net_class=Net_Class,
                        seq_len=db.seq_len,
                        d_feature=db.d_feature,
                        DEVICE=DEVICE,
                    )
                print(
                    f"Data/model ready: dataset={dataset}, seed={seed}, "
                    f"train_batches={len(db.train_loader)}, val_batches={len(db.val_loader)}, "
                    f"test_batches={len(db.test_loader)}, seq_len={db.seq_len}, d_feature={db.d_feature}",
                    flush=True,
                )
                total_params = sum(p.numel() for p in net.parameters())
                logger.info(
                    f"Training dataset: {dataset}\r\n"
                    f"Seed: {seed}\r\n"
                    f"Trainable parameters: {total_params}",
                    pos="blank",
                )

                optim = get_optim(model_config=model_config, dataset=dataset, net=net)
                logger.info(get_log_info(model_config=model_config, exp_config=exp_config, dataset=dataset, seed=seed))

                net, _ = train_one_trail(
                    db=db,
                    net=net,
                    optim=optim,
                    exp_config=exp_config,
                    model_config=model_config,
                    logger=logger,
                    dataset=dataset,
                    seed=seed,
                )

                MIT_mae, MIT_mse, MIT_mre, time_consume = test_val_one_trail(
                    db=db,
                    net=net,
                    stage="test",
                    logger=logger,
                    exp_config=exp_config,
                )
                dataset_results.loc[seed] = dict(
                    MAE=MIT_mae,
                    MSE=MIT_mse,
                    MRE=MIT_mre,
                    Time_Usage=time_consume,
                )
                logger.info(
                    f"Finished seed {seed} on {dataset}.\r\n"
                    f"Current results:\r\n{dataset_results}",
                    pos="blank",
                )

                if exp_config["global"]["save_model"]:
                    saved_name = (
                        f"{dataset} {exp_config['global']['used_model']} {exp_config['global']['padding_mode']} "
                        f"seed{seed} Test {round(MIT_mae, 4)} "
                        f"{exp_config['train']['pattern']} {exp_config['train']['missing_ratio']}"
                    )
                    saved_root = model_config["info"].get(
                        "saved_path",
                        f"./saved_models/{exp_config['global']['used_model']}",
                    )
                    save_model(saved_name=saved_name, net=net, logger=logger, overwrite=False, saved_root=saved_root)
                    logger.info(f"Saved file name: {saved_name}", pos="blank")

            overall_result[dataset] = dataset_results.to_string()

    except KeyboardInterrupt:
        logger.info("Training was interrupted manually. Partial results are shown below.", pos="blank")
    finally:
        logger.info(My_logger.format_message_multicolumn(overall_result, content="results"), pos="key")
        logger.info("Run complete.", pos="tail")


def train_one_trail(db, net, optim, exp_config, model_config, logger, dataset, seed, test_on=False):
    val_interval = exp_config["global"]["val_interval"]
    patience = exp_config["global"]["patience"]
    EPOCH = model_config[dataset]["setting"]["EPOCH"]
    model_name = model_config["info"]["model_name"]

    result_base = Result_base()
    early_stop = EarlyStop(patience=patience)

    for e_idx in trange(EPOCH, desc="Training Epochs", dynamic_ncols=True, leave=False):
        epoch_prefix = f"[Epoch {e_idx + 1}/{EPOCH}]"
        print(
            f"{epoch_prefix} START: model={model_name}, dataset={dataset}, seed={seed}, "
            f"missing={exp_config['train']['pattern']}:{exp_config['train']['missing_ratio']}",
            flush=True,
        )
        if exp_config["train"]["pattern"] in ["block", "subseq"]:
            exp_config["train"]["missing_ratio"] = 0.5
            exp_config["test"]["missing_ratio"] = 0.5

        logger.debug(
            f"Used model: {model_name}\tDataset: {dataset}\t"
            f"Missing ratio: {exp_config['train']['missing_ratio']}\t"
            f"Missing pattern: {exp_config['train']['pattern']}\t"
            f"Padding mode: {exp_config['global']['padding_mode']}\tSeed: {seed}",
            pos="blank",
        )

        output = None
        epoch_loss = 0.0
        if db.DEVICE.type == "cuda":
            import torch
            torch.cuda.synchronize(device=db.DEVICE)
        train_start = time.time()
        for _, input in enumerate(db.train_loader):
            net.train()
            optim.zero_grad()
            output = net.forward(input)
            loss = net.loss_func(outputs=output, inputs=input)
            loss.backward()
            optim.step()

            if "imputed_result" in output:
                result_base.record_epoch_data(input=input, output=output["imputed_result"])
            else:
                epoch_loss += loss.item()
        if db.DEVICE.type == "cuda":
            torch.cuda.synchronize(device=db.DEVICE)
        train_time_consume = time.time() - train_start

        if "imputed_result" in output:
            MIT_mae, MIT_mse, MIT_mre = result_base.cal_epoch_loss(stage="train")
            result_base.print_result(
                MIT_mae=MIT_mae,
                MIT_mse=MIT_mse,
                MIT_mre=MIT_mre,
                time_consume=train_time_consume,
                stage="train",
                logger=logger,
                console=True,
                log_result=False,
                console_prefix=epoch_prefix,
            )
        else:
            logger.debug(f"Train epoch loss: {round(epoch_loss, 4)}", pos="blank")
            print(f"{epoch_prefix} TRAIN: Loss:{round(epoch_loss, 4)}\tTime Consume:{round(train_time_consume, 4)}", flush=True)

        assert val_interval > 0, "val_interval should be greater than 0"
        if e_idx % val_interval == 0:
            MIT_mae_val, _, _ = test_val_one_trail(
                db=db,
                net=net,
                stage="val",
                logger=logger,
                exp_config=exp_config,
                console=True,
                console_prefix=epoch_prefix,
                log_result=False,
            )
            if test_on:
                test_val_one_trail(db=db, net=net, stage="test", logger=logger, exp_config=exp_config)

            previous_best = early_stop.best_result
            if early_stop.if_stop(result=MIT_mae_val, net=net):
                print(
                    f"{epoch_prefix} EarlyStop: monitored_val={round(MIT_mae_val, 6)} "
                    f"best={round(early_stop.best_result, 6)} "
                    f"patience_remaining=0/{patience} -> stop",
                    flush=True,
                )
                net.load_state_dict(early_stop.best_model_dict)
                MIT_mae_val, _, _ = test_val_one_trail(
                    db=db,
                    net=net,
                    stage="val",
                    logger=logger,
                    exp_config=exp_config,
                    console=True,
                    console_prefix=f"{epoch_prefix} [Best]",
                    log_result=False,
                )
                return net, MIT_mae_val
            improved = early_stop.best_result < previous_best
            status = "best updated" if improved else "no improvement"
            print(
                f"{epoch_prefix} EarlyStop: monitored_val={round(MIT_mae_val, 6)} "
                f"best={round(early_stop.best_result, 6)} "
                f"patience_remaining={early_stop.current_patience}/{patience} -> {status}",
                flush=True,
            )

    net.load_state_dict(early_stop.best_model_dict)
    print(f"Training reached max epochs. Loading best model with best_val={round(early_stop.best_result, 6)}.", flush=True)
    MIT_mae_val, _, _ = test_val_one_trail(db=db, net=net, stage="val", logger=logger, exp_config=exp_config, log_result=False)
    return net, MIT_mae_val
