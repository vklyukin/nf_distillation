import logging
import matplotlib.pyplot as plt
import neptune
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import typing as tp
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data_utils import get_CIFAR10, get_RICH, postprocess
from metrics.fid import calculate_fid
from models import (
    create_glow_model,
    gaussian_sample,
    FlowStep,
    inherit_permutation_matrix,
)
from losses import IdentityLoss


class NFModel(pl.LightningModule):
    def __init__(self, config: tp.Dict[str, tp.Any]):
        super().__init__()
        self.params = config
        self.teacher = self.create_model(model_name="teacher")
        self.student = self.create_model(model_name="student")
        (
            self.nll_loss,
            self.nll_weight,
            self.kd_loss,
            self.kd_weight,
        ) = self.create_loss_func()

        if not self.params["student"].get("is_1d", False):
            mean, logs = self.student.prior(None, None)
            self.register_buffer("latent", gaussian_sample(mean, logs, 1))

        self.student_kd_indices = []
        for i, layer in enumerate(self.student.flow.layers):
            if isinstance(layer, FlowStep):
                self.student_kd_indices.append(i)

        self.teacher_kd_indices = []
        found_flowsteps = 0
        for i, layer in enumerate(self.teacher.flow.layers):
            if isinstance(layer, FlowStep):
                found_flowsteps += 1

                if found_flowsteps % 4 == 0:
                    self.teacher_kd_indices.append(i)

        if self.params["inherit_p"]:
            assert not self.params["teacher"].get(
                "is_1d", False
            ), "Teacher model must be 3-dimensional"
            assert not self.params["student"].get(
                "is_1d", False
            ), "Student model must be 3-dimensional"
            inherit_permutation_matrix(
                self.student,
                self.teacher,
                self.student_kd_indices,
                self.teacher_kd_indices,
            )

    def create_model(self, model_name, teacher=None) -> nn.Module:
        """Create model from config"""
        if self.params["architecture"].lower() == "glow":
            return create_glow_model(self.params[model_name])
        else:
            raise NameError(
                "Unknown model architecture: {}".format(self.params["architecture"])
            )

    def create_loss_func(self):
        """Create loss function from config"""
        loss_description = self.params["loss"]

        nll_loss = IdentityLoss()
        nll_weight = loss_description["nll"]["weight"]

        kd_loss_description = loss_description["kd"]
        kd_weight = kd_loss_description["weight"]

        if kd_loss_description["name"].lower() == "mse":
            kd_loss = nn.MSELoss(reduction="mean")
        else:
            raise NameError(
                "Unkown KD loss name: {}".format(kd_loss_description["name"])
            )

        return nll_loss, nll_weight, kd_loss, kd_weight

    def forward(self, batch):
        """Return latent variables and student NLL"""
        if "drop_weights" not in self.params["data"]:
            x, y = batch
        else:
            x, y, _ = batch

        if self.params["student"]["y_condition"]:
            student_z, student_nll, _ = self.student(x, y)
        else:
            student_z, student_nll, _ = self.student(x, None)

        if self.params["loss"]["kd"]["weight"] > 0:
            with torch.no_grad():
                if self.params["student"]["y_condition"]:
                    teacher_z, _, _ = self.teacher(x, y)
                else:
                    teacher_z, _, _ = self.teacher(x, None)
        else:
            teacher_z = None

        return {
            "student_nll": student_nll,
            "student_z": student_z,
            "teacher_z": teacher_z,
        }

    def loss(self, model_outputs, *args):
        """Count loss function value"""
        kd_loss_value = torch.tensor(0.0, device=self.device)

        if self.kd_weight > 0:
            student_z = model_outputs["student_z"]
            teacher_z = model_outputs["teacher_z"]

            for student_layer_id, teacher_layer_id in zip(
                self.student_kd_indices, self.teacher_kd_indices
            ):
                kd_loss_value += self.kd_loss(
                    student_z[student_layer_id], teacher_z[teacher_layer_id]
                )

        return {
            "nll": model_outputs["student_nll"].mean(),
            "kd": kd_loss_value,
        }

    def generate(self, batch):
        """Generate x from noise conditioning on batch"""

        if "drop_weights" not in self.params["data"]:
            _, condition = batch
        else:
            _, condition, _ = batch

        if self.params["student"]["y_condition"]:
            generated = self.student(reverse=True, y_onehot=condition, temperature=1)[
                -1
            ]
        else:
            generated = self.student(reverse=True, temperature=1)[-1]

        return generated

    def configure_optimizers(self):
        if self.params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.student.parameters(), lr=self.params["learning_rate"]
            )
        else:
            raise NameError("Unknown optimizer name")
        return optimizer

    def training_step(self, batch, batch_idx):
        # TODO: take weights for RICH dataset into account
        model_outputs = self.forward(batch)
        train_losses = self.loss(model_outputs)

        self.log("train_batch_nll", train_losses["nll"], on_step=True)
        self.log("train_batch_kd", train_losses["kd"], on_step=True)

        result_loss = (
            self.nll_weight * train_losses["nll"] + self.kd_weight * train_losses["kd"]
        )

        self.log("train_batch_loss", result_loss, on_step=True, prog_bar=True)

        return {
            "nll": train_losses["nll"],
            "kd": train_losses["kd"],
            "loss": result_loss,
        }

    def validation_step(self, batch, batch_idx):
        model_outputs = self.forward(batch)
        train_losses = self.loss(model_outputs)
        generated = self.generate(batch)

        result_loss = (
            self.nll_weight * train_losses["nll"] + self.kd_weight * train_losses["kd"]
        )

        return {
            "nll": train_losses["nll"],
            "kd": train_losses["kd"],
            "loss": result_loss,
            "generated": generated,
            "real": batch[0],
        }

    def epoch_end(self, step_outputs, fid_mode):
        epoch_loss = torch.stack([output["loss"] for output in step_outputs]).mean()
        epoch_nll = torch.stack([output["nll"] for output in step_outputs]).mean()
        epoch_kd = torch.stack([output["kd"] for output in step_outputs]).mean()

        metrics = {
            "epoch_loss": epoch_loss,
            "epoch_nll": epoch_nll,
            "epoch_kd": epoch_kd,
        }

        if not self.params["student"].get("is_1d", False):
            epoch_fid = self.calc_fid(fid_mode)
            metrics["epoch_fid"] = epoch_fid

        return metrics

    def training_epoch_end(self, outputs):
        metrics = self.epoch_end(outputs, fid_mode="train")
        self.log("train_epoch_loss", metrics["epoch_loss"])
        self.log("train_epoch_nll", metrics["epoch_nll"])
        self.log("train_epoch_kd", metrics["epoch_kd"])

        if not self.params["student"].get("is_1d", False):
            self.log("train_epoch_fid", metrics["epoch_fid"])

    def validation_epoch_end(self, outputs):
        metrics = self.epoch_end(outputs, fid_mode="val")
        self.log("val_epoch_loss", metrics["epoch_loss"])
        self.log("val_epoch_nll", metrics["epoch_nll"])
        self.log("val_epoch_kd", metrics["epoch_kd"])

        if not self.params["student"].get("is_1d", False):
            self.log("val_epoch_fid", metrics["epoch_fid"])

            self.sample_images()
            self.trainer.logger.experiment.log_image("samples.png", x=plt.gcf())
        else:
            self.get_histograms(outputs["generated"], outputs["real"])
            self.trainer.logger.experiment.log_image("histograms.png", x=plt.gcf())

    def calc_fid(self, fid_mode) -> float:
        if fid_mode == "train":
            dataset = self.train_dataset
        elif fid_mode == "val":
            dataset = self.valid_dataset

        real_data_indices = np.random.choice(
            dataset.data.shape[0], self.params["fid_samples"], replace=False
        )
        real_data_for_fid = dataset.data[real_data_indices]

        with torch.no_grad():
            per_batch_samples = 1024  # hardcoded to prevent GPU OOM

            mean, logs = self.student.prior(torch.ones((per_batch_samples,)), None)

            gen_data_for_fid = [
                postprocess(
                    self.student(
                        z=gaussian_sample(mean, logs, 1), temperature=1, reverse=True
                    )[-1]
                )
                .detach()
                .cpu()
                .numpy()
                for _ in range(self.params["fid_samples"] // per_batch_samples)
            ]

            gen_data_for_fid = np.concatenate(gen_data_for_fid)
            gen_data_for_fid = np.transpose(gen_data_for_fid, (0, 2, 3, 1))

        fid_score = calculate_fid(real_data_for_fid, gen_data_for_fid, False, 128)

        return fid_score

    def sample_images(self):
        student_samples = self.student(reverse=True, z=self.latent, temperature=1)[-1]
        images = postprocess(student_samples).cpu()
        grid = make_grid(images[:30], nrow=6).permute(1, 2, 0)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis("off")

    def get_histograms(self, generated, real):
        pass

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage: str):
        data_description = self.params["data"]

        if data_description["name"].lower() == "cifar-10":
            image_shape, num_classes, train_dataset, valid_dataset = get_CIFAR10(
                data_description["augment"],
                data_description["data_path"],
                data_description["download"],
            )
        elif data_description["name"].lower() == "rich":
            image_shape, num_classes, train_dataset, valid_dataset, scaler = get_RICH(
                data_description["particle"],
                data_description["drop_weights"],
                data_description["data_path"],
                data_description["download"],
            )
            self.scaler = scaler

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.image_shape = image_shape
        self.num_classes = num_classes

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
