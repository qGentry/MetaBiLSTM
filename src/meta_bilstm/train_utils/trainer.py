from pprint import pprint

import torch
from meta_bilstm.meta_wrapper import ModelWrapper
from meta_bilstm.nn_utils.losses import calc_accuracy, seq_loss
from meta_bilstm.utils.dataset import PosTagDataset, create_dataloader
from meta_bilstm.utils.preprocessing import Preprocessor
from torch.optim import lr_scheduler
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model_config: dict,
        device: str,
        data_path: str,
        batch_size: int = 32,
        test_size: float = 0.25,
        model_names=("word", "char", "meta"),
    ):

        data = self.read_and_parse_data_from_path(data_path, test_size)
        print(model_config)
        for model_name in model_config:
            model_config[model_name]["output_proj_size"] = len(
                data["tags"]["tag_to_idx"]
            )
            model_config[model_name]["device"] = device

        model_config["word_model_params"]["dataset"] = data["datasets"]["train"]
        model_config["char_model_params"]["num_embs"] = len(
            data["chars"]["char_to_idx"]
        )

        self.model = ModelWrapper(model_config)
        self.model_names = model_names
        self.device = device
        self.train_metrics = self.get_empty_metrics()
        self.model.to(device)

        self.datasets = {
            "train": PosTagDataset(
                data["datasets"]["train"],
                self.model.word_model.emb_layer.token_to_idx,
                data["chars"]["char_to_idx"],
            ),
            "test": PosTagDataset(
                data["datasets"]["test"],
                self.model.word_model.emb_layer.token_to_idx,
                data["chars"]["char_to_idx"],
            ),
        }
        self.dataloaders = {
            "train": create_dataloader(self.datasets["train"], batch_size),
            "test": create_dataloader(self.datasets["test"], batch_size),
        }

    @staticmethod
    def read_and_parse_data_from_path(data_path, test_size):
        preprocessor = Preprocessor(test_size)
        data = preprocessor.get_train_test_list_datasets(data_path)
        return data

    def train_model(
        self,
        epochs: int,
        train_log_step: int = 60,
        scheduler_step_size: int = 1,
        scheduler_gamma: float = 0.2,
    ):

        optimizers = {
            "char": torch.optim.Adam(self.model.char_model.parameters()),
            "word": torch.optim.Adam(self.model.word_model.parameters()),
            "meta": torch.optim.Adam(self.model.meta_model.parameters()),
        }

        schedulers = {
            "char": lr_scheduler.StepLR(
                optimizers["char"], step_size=scheduler_step_size, gamma=scheduler_gamma
            ),
            "word": lr_scheduler.StepLR(
                optimizers["word"], step_size=scheduler_step_size, gamma=scheduler_gamma
            ),
            "meta": lr_scheduler.StepLR(
                optimizers["meta"], step_size=scheduler_step_size, gamma=scheduler_gamma
            ),
        }
        for epoch in range(epochs):
            self.model.train()
            for j, batch in tqdm(enumerate(self.dataloaders["train"])):
                batch_metrics = self.train_step(batch, optimizers)
                if j % train_log_step == 0:
                    self.accumulate_train_metrics(batch_metrics)
                    print(f"Epochs {epoch}, train step {j}, train metrics:")
                    pprint(self.train_metrics)
                    self.train_metrics = self.get_empty_metrics()
            for model_name in schedulers:
                schedulers[model_name].step()
            valid_metrics = self.validate_model(self.dataloaders["test"])
            print(f"Epochs {epoch} is finished, test metrics:")
            pprint(valid_metrics)

    def accumulate_train_metrics(self, batch_metrics):
        for metric in self.train_metrics:
            for model_name in self.model_names:
                self.train_metrics[metric][model_name].append(
                    batch_metrics[metric][model_name].detach().item()
                )

    @staticmethod
    def get_empty_metrics():
        return {
            "loss": {
                "char": [],
                "word": [],
                "meta": [],
            },
            "acc": {
                "char": [],
                "word": [],
                "meta": [],
            },
        }

    def train_step(self, batch, optimizers):
        outputs = self.model(batch)

        metrics = {
            "loss": {
                "char": None,
                "word": None,
                "meta": None,
            },
            "acc": {
                "char": None,
                "word": None,
                "meta": None,
            },
        }
        for model_name in self.model_names:
            metrics["loss"][model_name] = seq_loss(
                outputs[model_name]["logits"],
                outputs[model_name]["lens"],
                batch["labels"].to(self.device),
            )
            metrics["loss"][model_name].backward()
            optimizers[model_name].step()

            metrics["acc"][model_name] = calc_accuracy(
                batch["word_idx"][0].to(self.device),
                outputs[model_name]["logits"],
                batch["labels"].to(self.device),
            )
        return metrics

    def validate_model(self, test_loader):
        self.model.eval()

        metrics = self.get_empty_metrics()

        for batch in tqdm(test_loader):
            outputs = self.model(batch)
            with torch.no_grad():
                for model_name in self.model_names:
                    metrics["loss"][model_name].append(
                        seq_loss(
                            outputs[model_name]["logits"],
                            outputs[model_name]["lens"],
                            batch["labels"].to(self.device),
                        )
                    )

                    metrics["acc"][model_name].append(
                        calc_accuracy(
                            batch["word_idx"][0].to(self.device),
                            outputs[model_name]["logits"],
                            batch["labels"].to(self.device),
                        )
                    )
        for metric in metrics:
            for model_name in self.model_names:
                metrics[metric][model_name] = sum(metrics[metric][model_name]) / len(
                    sum(metrics[metric][model_name])
                )
        return metrics
