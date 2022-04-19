import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from meta_bilstm.train_utils.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_path",
        "-e",
        type=str,
        default="data/embeddings/model.txt",
        help="path to pretrained embeddings",
    )
    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        default="config/model_config.json",
        help="path to model config",
    )
    parser.add_argument(
        "--train-data",
        "-t",
        type=str,
        default="data/porttinari-base-train.conllu",
        help="path to training data in conllu format",
    )
    parser.add_argument(
        "--test-data",
        "-v",
        type=str,
        default="data/porttinari-base-test.conllu",
        help="path to training data in conllu format",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="device to use for model training/inference",
    )
    parser.add_argument(
        "--runs_path", type=str, default="runs", help="Path to store trained model"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        model_config = json.load(f)

    trainer = Trainer(model_config, args.device, args.train_data, args.test_data)
    trainer.train_model(5)
    run_name = "{}-{}.pt".format(
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        args.train_data.split("data/")[1].split("-")[0],
    )
    torch.save(trainer.model, Path(args.runs_path).joinpath(run_name))


if __name__ == "__main__":
    main()
