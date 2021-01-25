import argparse
import json

from meta_bilstm.train_utils.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_path", "-e",
        type=str,
        default="data/embeddings/model.txt",
        help="path to pretrained embeddings",
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        default="config/model_config.json",
        help="path to model config",
    )
    parser.add_argument(
        "--train-data", "-t",
        type=str,
        default="data/ru_syntagrus-ud-train.conllu",
        help="path to training data in conllu format"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="device to use for model training/inference",
    )
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        model_config = json.load(f)

    trainer = Trainer(model_config, args.device, args.train_data)
    trainer.train_model(5)


if __name__ == "__main__":
    main()
