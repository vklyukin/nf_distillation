#! /usr/bin/env python3

import json
import pytorch_lightning as pl
import os
from argparse import ArgumentParser
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_module import NFModel


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("config_file", help="json train config with parameters")
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    neptune_logger = NeptuneLogger(
        project_name=config["neptune"]["project_name"],
        experiment_name=config["neptune"]["experiment_name"],
        tags=config["neptune"]["tags"],
        params=config,
    )

    model_checkpoint = ModelCheckpoint(
        save_weights_only=True,
        save_top_k=3,
        monitor="val_epoch_fid",
        mode="min",
        period=1,
    )

    pl.seed_everything(config["seed"])

    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        checkpoint_callback=model_checkpoint,
        logger=neptune_logger,
        gpus=config["gpus"],
        weights_summary="full",
        track_grad_norm=2,
    )
    model = NFModel(config)
    trainer.fit(model)

    for k in model_checkpoint.best_k_models.keys():
        model_name = os.path.join(model_checkpoint.dirpath, k.split("/")[-1])
        neptune_logger.experiment.log_artifact(k, model_name)


if __name__ == "__main__":
    main()
