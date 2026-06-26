import argparse
import os
import traceback

import yaml


SUPPORTED_MODELS = ("NASDiff", "CSDI", "Transformer", "MEAN", "LINEAR", "SAMPLE_MEAN")


def add_boolean_argument(parser, name, default=None, help_text=None):
    parser.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=argparse.SUPPRESS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run NASDiff and selected baselines.")
    parser.add_argument("--models_root", type=str, default="model/", help=argparse.SUPPRESS)
    parser.add_argument("--saved_models_root", type=str, default="saved_models/", help=argparse.SUPPRESS)
    parser.add_argument("--config_root", type=str, default="configs/", help=argparse.SUPPRESS)
    parser.add_argument("--model_name", type=str, metavar="MODEL", default=None)
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--pattern", type=str, choices=["point", "subseq", "block"], default=None)
    parser.add_argument("--missing_ratio", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", type=str, choices=["train_multiple", "test_with_trained"], default=None, help=argparse.SUPPRESS)
    add_boolean_argument(parser, "use_best_config", default=None, help_text=argparse.SUPPRESS)
    parser.add_argument("--test_sampling_times", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=None)
    add_boolean_argument(parser, "save_model", default=None, help_text=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(exp_config, args):
    if args.model_name is not None:
        if args.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {args.model_name}")
        exp_config["global"]["used_model"] = args.model_name
    if args.mode is not None:
        exp_config["global"]["mode"] = args.mode
    if args.use_best_config is not None:
        exp_config["global"]["use_best_config"] = args.use_best_config
    if args.device is not None:
        exp_config["global"]["DEVICE"] = args.device
    if args.save_model is not None:
        exp_config["global"]["save_model"] = args.save_model
    if args.seeds is not None:
        exp_config["global"]["seeds"] = args.seeds
    if args.datasets is not None:
        exp_config["train"]["datasets"] = args.datasets
    if args.pattern is not None:
        exp_config["train"]["pattern"] = args.pattern
        exp_config["test"]["pattern"] = args.pattern
    if args.missing_ratio is not None:
        exp_config["train"]["missing_ratio"] = args.missing_ratio
        exp_config["test"]["missing_ratio"] = args.missing_ratio
    if args.test_sampling_times is not None:
        exp_config["test"]["test_sampling_times"] = args.test_sampling_times
    return exp_config


def model_config_path(config_root, model_name, use_best_config):
    if use_best_config:
        best_path = os.path.join(config_root, f"{model_name}_best.yaml")
        if os.path.exists(best_path):
            return best_path
    return os.path.join(config_root, f"{model_name}.yaml")


def apply_model_overrides(model_config, args):
    if args.epochs is None:
        return model_config
    for config in model_config.values():
        if isinstance(config, dict) and "setting" in config:
            config["setting"]["EPOCH"] = args.epochs
    return model_config


def main():
    try:
        args = parse_args()
        from utils.log import My_logger
        from utils.quiet import suppress_third_party_output
        with suppress_third_party_output():
            from test import test_naive, test_with_trained_model
            from train import train_multiple
            from utils.train_test_func import get_model_dataset_class

        exp_config = load_yaml(os.path.join(args.config_root, "exp_config.yaml"))
        dataset_config = load_yaml(os.path.join(args.config_root, "dataset_config.yaml"))
        exp_config = apply_cli_overrides(exp_config, args)

        model_name = exp_config["global"]["used_model"]
        mode = exp_config["global"]["mode"]
        use_best_config = exp_config["global"]["use_best_config"]
        logger = My_logger(logger_name=model_name)

        if model_name in ["MEAN", "LINEAR", "SAMPLE_MEAN"]:
            test_naive(model_name=model_name, dataset_config=dataset_config, exp_config=exp_config, logger=logger)
            return

        config_path = model_config_path(args.config_root, model_name, use_best_config)
        model_config = load_yaml(config_path)
        model_config = apply_model_overrides(model_config, args)
        logger.info(f"Using model config: {config_path}", pos="blank")

        with suppress_third_party_output():
            Net_Class, Dataset_Class = get_model_dataset_class(args.models_root, model_name)

        if mode == "train_multiple":
            train_multiple(exp_config, dataset_config, Net_Class, Dataset_Class, model_config, logger)
        elif mode == "test_with_trained":
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required when --mode test_with_trained")
            test_with_trained_model(
                exp_config=exp_config,
                config_root=args.config_root,
                dataset_config=dataset_config,
                models_root=args.models_root,
                logger=logger,
                saved_root=args.saved_models_root,
                file_name=args.checkpoint,
            )

    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
