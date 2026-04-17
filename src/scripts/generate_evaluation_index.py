from __future__ import annotations

from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from src.misc.import_hook import install_runtime_import_hook

# Configure beartype and jaxtyping.
with install_runtime_import_hook():
    from src.config import load_typed_config
    from src.dataset import DatasetCfg
    from src.dataset.data_module import DataLoaderCfg, DataModule
    from src.evaluation.evaluation_index_generator import (
        EvaluationIndexGenerator,
        EvaluationIndexGeneratorCfg,
    )
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    index_generator: EvaluationIndexGeneratorCfg
    seed: int


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="generate_evaluation_index",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices="auto", strategy="auto")
    data_module = DataModule(cfg.dataset, cfg.data_loader, None)
    evaluation_index_generator = EvaluationIndexGenerator(cfg.index_generator)
    trainer.test(evaluation_index_generator, datamodule=data_module)
    evaluation_index_generator.save_index()


if __name__ == "__main__":
    train()
