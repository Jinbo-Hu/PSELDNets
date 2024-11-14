# Reference: https://github.com/ashleve/lightning-hydra-template

from typing import List

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.config import get_dataset
from utils.utilities import (extras, get_pylogger, instantiate_callbacks,
                             instantiate_loggers, log_hyperparameters)

log = get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """ Train or test the model.
    training or testing.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
   
    default_dataset = list(cfg.data.train_dataset.keys())[0]
    dataset = get_dataset(dataset_name=default_dataset, cfg=cfg)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, dataset, 'fit')
    valid_meta = datamodule.paths_dict, datamodule.valid_gt_dcaseformat

    log.info(f"Instantiating model <{cfg.modelmodule._target_}> ...")
    model = hydra.utils.instantiate(cfg.modelmodule, cfg, dataset, valid_meta)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")) # type: ignore
    
    
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    
    main()