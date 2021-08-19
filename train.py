#! /usr/bin/env python3

import hydra
import json
import logging
import pytorch_lightning as pl
import os
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_module import NFModel


logger = logging.getLogger(__name__)

def prepare_config(config):
    if not os.path.isabs(config.data.path):
        config.data.data_path = os.path.join(hydra.utils.get_original_cwd(),
                                             config.data.data_path)
        logger.info(f"config.data.data_path modified to {config.data.path}")

    if not os.path.isabs(config.student.checkpoint):
        config.student.checkpoint = os.path.join(hydra.utils.get_original_cwd(),
                                                         config.student.checkpoint)
        logger.info(f"config.student.checkpoint modified to {config.student.checkpoint}")

    if not os.path.isabs(config.teacher.checkpoint):
        config.teacher.checkpoint = os.path.join(hydra.utils.get_original_cwd(),
                                                         config.teacher.checkpoint)
        logger.info(f"config.teacher.checkpoint modified to {config.teacher.checkpoint}")

    try:
        if not os.path.isabs(config.loss.perceptual.checkpoint):
            config.loss.perceptual.checkpoint = os.path.join(hydra.utils.get_original_cwd(),
                                                             config.loss.perceptual.checkpoint)
            logger.info(f"config.loss.perceptual.checkpoint modified to {config.loss.perceptual.checkpoint}")
    except KeyError as e:
        logger.info(e)

    try:
        if not os.path.isabs(config.loss.perceptual.checkpoint):
            config.loss.perceptual.checkpoint = os.path.join(hydra.utils.get_original_cwd(),
                                                             config.loss.perceptual.checkpoint)
            logger.info(f"config.loss.perceptual.checkpoint modified to {config.loss.perceptual.checkpoint}")
    except KeyError as e:
        logger.info(e)


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):

    logger.info("Setting up logger")
    neptune_logger = NeptuneLogger(
        project_name=config["neptune"]["project_name"],
        experiment_name=config["neptune"]["experiment_name"],
        tags=OmegaConf.to_container(config["neptune"]["tags"]),
        params=OmegaConf.to_container(config),
    )

    logger.info("Setting up checkpoint saver")
    model_checkpoint = ModelCheckpoint(
        save_weights_only=True,
        save_top_k=3,
        monitor="val_epoch_loss"
        if config["student"].get("is_1d", False)
        else "val_epoch_fid",
        mode="min",
        period=1,
    )

    logger.info("Fixing seed")
    pl.seed_everything(config["seed"])

    logger.info("Creating trainer")
    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        gradient_clip_val=10,
        checkpoint_callback=model_checkpoint,
        logger=neptune_logger,
        gpus=config["gpus"],
        weights_summary="full",
        track_grad_norm=2 if config["track_grad_norm"] else -1,
    )

    logger.info("Create PL model")
    model = NFModel(config)

    logger.info("Starting training")
    trainer.fit(model)

    for k in model_checkpoint.best_k_models.keys():
        model_name = os.path.join(model_checkpoint.dirpath, k.split("/")[-1])
        neptune_logger.experiment.log_artifact(k, model_name)


if __name__ == "__main__":
    main()
