import logging
import matplotlib.pyplot as plt
import neptune
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import typing as tp
from catboost import CatBoostClassifier
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data_utils import get_CelebA, get_CIFAR10, get_RICH, postprocess
from metrics import calculate_fid, calculate_roc_auc
from models import (
    create_glow_model,
    gaussian_sample,
    FlowStep,
    SqueezeLayer,
    inherit_permutation_matrix,
    VGGPerceptualLoss,
)


logger = logging.getLogger(__name__)


class NFModel(pl.LightningModule):
    def __init__(self, config: tp.Dict[str, tp.Any]):
        super().__init__()
        self.params = config
        
        logger.info("Creating teacher")
        self.teacher = self.create_model(model_name="teacher")
        logger.info("Creating student")
        self.student = self.create_model(model_name="student")
        logger.info("Creating loss functions")
        (
            self.nll_weight,
            self.kd_loss,
            self.kd_weight,
            self.perceptual_loss,
            self.perceptual_weight,
        ) = self.create_loss_func()

        logger.info("Creating fixed latent")
        if not self.params["student"].get("is_1d", False):
            mean, logs = self.student.prior(None, None)
            self.register_buffer("latent", gaussian_sample(mean, logs, 1))

        logger.info("Getting indices for KD")
        self.student_kd_indices, self.teacher_kd_indices = self._get_kd_indices()

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

        logger.info("Setting seaborn settings")
        sns.set()

    def _get_kd_indices(self) -> tp.Tuple[tp.List, tp.List]:
        if self.params["loss"]["kd"]["weight"] + self.params["loss"]["perceptual"]["weight"] == 0:
            return [], []

        student_kd_indices = []
        for i, layer in enumerate(self.student.flow.layers):
            if isinstance(layer, SqueezeLayer):
                student_kd_indices.append(i)

        teacher_kd_indices = []
        for i, layer in enumerate(self.teacher.flow.layers):
            if isinstance(layer, SqueezeLayer):
                teacher_kd_indices.append(i)

        return student_kd_indices, teacher_kd_indices

    def load_checkpoint(self, model, checkpoint_path) -> nn.Module:
        model_state = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )

        if "state_dict" in model_state:
            model_state = {
                ".".join(k.split(".")[1:]): v
                for k, v in torch.load(checkpoint_path)["state_dict"].items()
                if k.startswith("student.")
            }

        model.load_state_dict(model_state)

        return model

    def create_model(self, model_name) -> tp.Optional[nn.Module]:
        """Create model from config"""
        if model_name == "teacher" and (
            self.params["loss"]["kd"]["weight"] +\
            self.params["loss"]["perceptual"]["weight"] == 0
        ):
            return None

        if self.params["architecture"].lower() == "glow":
            model_params = OmegaConf.to_container(self.params[model_name])
            del model_params["checkpoint"]

            logger.info("Creating GLOW model")
            model = create_glow_model(model_params)

            checkpoint_path = self.params[model_name]["checkpoint"]
            if checkpoint_path:
                logger.info("Loading checkpoint")
                self.load_checkpoint(model, checkpoint_path)

            return model
        else:
            raise NameError(
                "Unknown model architecture: {}".format(self.params["architecture"])
            )

    def create_loss_func(self):
        """Create loss function from config"""
        loss_description = self.params["loss"]

        nll_weight = loss_description["nll"]["weight"]

        kd_loss_description = loss_description["kd"]
        kd_weight = kd_loss_description["weight"]

        perceptual_loss_description = loss_description["perceptual"]
        perceptual_weight = perceptual_loss_description["weight"]

        if kd_loss_description["name"].lower() == "mse":
            kd_loss = nn.MSELoss(reduction="mean")
        else:
            raise NameError(
                "Unkown KD loss name: {}".format(kd_loss_description["name"])
            )

        if perceptual_loss_description["name"].lower() == "vgg":
            perceptual_loss = VGGPerceptualLoss(
                resize=True, checkpoint_path=perceptual_loss_description["checkpoint"]
            )
        elif perceptual_loss_description["name"].lower() == "l1":
            perceptual_loss = nn.L1Loss()
        else:
            raise NameError(
                "Unkown perceptual loss name: {}".format(
                    perceptual_loss_description["name"]
                )
            )

        return (
            nll_weight,
            kd_loss,
            kd_weight,
            perceptual_loss,
            perceptual_weight,
        )

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

        if self.kd_weight > 0:
            with torch.no_grad():
                if self.params["student"]["y_condition"]:
                    teacher_z, _, _ = self.teacher(x, y)
                else:
                    teacher_z, _, _ = self.teacher(x, None)
        else:
            teacher_z = None

        if self.perceptual_weight > 0:
            y_onehot = torch.zeros((self.params["batch_size"],))
            mean, logs = self.student.prior(None, y_onehot=y_onehot)
            latent = gaussian_sample(mean, logs, 1)

            student_x = (
                torch.clamp(
                    self.student(z=latent, temperature=1, reverse=True)[-1], -0.5, 0.5
                )
                + 0.5
            )
            with torch.no_grad():
                teacher_x = (
                    torch.clamp(
                        self.teacher(z=latent, temperature=1, reverse=True)[-1],
                        -0.5,
                        0.5,
                    )
                    + 0.5
                )
        else:
            student_x = None
            teacher_x = None

        return {
            "student_nll": student_nll,
            "student_z": student_z,
            "teacher_z": teacher_z,
            "student_x": student_x,
            "teacher_x": teacher_x,
        }

    def loss(self, model_outputs, *args):
        """Count loss function value"""
        kd_loss_value = torch.tensor(0.0, device=self.device)
        perceptual_loss_value = torch.tensor(0.0, device=self.device)

        if self.kd_weight > 0:
            student_z = model_outputs["student_z"]
            teacher_z = model_outputs["teacher_z"]

            for student_layer_id, teacher_layer_id in zip(
                self.student_kd_indices, self.teacher_kd_indices
            ):
                kd_loss_value += self.kd_loss(
                    student_z[student_layer_id], teacher_z[teacher_layer_id]
                )

        if self.perceptual_weight > 0:
            student_x = model_outputs["student_x"]
            teacher_x = model_outputs["teacher_x"]

            perceptual_loss_value += self.perceptual_loss(student_x, teacher_x)

        return {
            "nll": model_outputs["student_nll"].mean(),
            "kd": kd_loss_value,
            "perceptual": perceptual_loss_value,
        }

    @torch.no_grad()
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
                self.student.parameters(), 
                lr=self.params["learning_rate"], 
                weight_decay=self.params["weight_decay"],
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
        self.log("train_batch_perceptual", train_losses["perceptual"], on_step=True)

        result_loss = (
            self.nll_weight * train_losses["nll"]
            + self.kd_weight * train_losses["kd"]
            + self.perceptual_weight * train_losses["perceptual"]
        )

        self.log("train_batch_loss", result_loss, on_step=True, prog_bar=True)

        return {
            "nll": train_losses["nll"],
            "kd": train_losses["kd"],
            "perceptual": train_losses["perceptual"],
            "loss": result_loss,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        model_outputs = self.forward(batch)
        train_losses = self.loss(model_outputs)
        generated = self.generate(batch)

        result_loss = (
            self.nll_weight * train_losses["nll"]
            + self.kd_weight * train_losses["kd"]
            + self.perceptual_weight * train_losses["perceptual"]
        )

        output = {
            "nll": train_losses["nll"],
            "kd": train_losses["kd"],
            "perceptual": train_losses["perceptual"],
            "loss": result_loss,
            "generated": generated,
            "real": batch[0],
        }

        if self.params["student"]["is_1d"]:
            output["weights"] = batch[2]

        return output

    def epoch_end(self, step_outputs, fid_mode):
        epoch_loss = torch.stack([output["loss"] for output in step_outputs]).mean()
        epoch_nll = torch.stack([output["nll"] for output in step_outputs]).mean()
        epoch_kd = torch.stack([output["kd"] for output in step_outputs]).mean()
        epoch_perceptual = torch.stack(
            [output["perceptual"] for output in step_outputs]
        ).mean()

        metrics = {
            "epoch_loss": epoch_loss,
            "epoch_nll": epoch_nll,
            "epoch_kd": epoch_kd,
            "epoch_perceptual": epoch_perceptual,
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
        self.log("train_epoch_perceptual", metrics["epoch_perceptual"])

        if not self.params["student"].get("is_1d", False):
            self.log("train_epoch_fid", metrics["epoch_fid"])

    def validation_epoch_end(self, outputs):
        metrics = self.epoch_end(outputs, fid_mode="val")
        self.log("val_epoch_loss", metrics["epoch_loss"])
        self.log("val_epoch_nll", metrics["epoch_nll"])
        self.log("val_epoch_kd", metrics["epoch_kd"])
        self.log("val_epoch_perceptual", metrics["epoch_perceptual"])

        if not self.params["student"].get("is_1d", False):
            self.log("val_epoch_fid", metrics["epoch_fid"])

            self.sample_images()
            self.trainer.logger.experiment.log_image("samples.png", x=plt.gcf())
        else:
            generated = (
                torch.cat([output["generated"] for output in outputs])
                .detach()
                .cpu()
                .numpy()
            )
            real = (
                torch.cat([output["real"] for output in outputs]).detach().cpu().numpy()
            )
            weights = (
                torch.cat([output["weights"] for output in outputs])
                .detach()
                .cpu()
                .numpy()
            )

            self.get_histograms(generated, real)
            self.trainer.logger.experiment.log_image("histograms.png", x=plt.gcf())

            roc_auc = self.calc_roc_auc(generated, real, weights)
            self.log("val_epoch_roc_auc", roc_auc)

    @torch.no_grad()
    def calc_fid(self, fid_mode) -> float:
        if fid_mode == "train":
            dataset = self.train_dataset
        elif fid_mode == "val":
            dataset = self.valid_dataset

        real_data_indices = np.random.choice(
            len(dataset), self.params["fid_samples"], replace=False
        )
        real_data_for_fid = postprocess(
            torch.stack([
                dataset[index][0] for index in real_data_indices
            ])
        ).cpu().numpy()
        real_data_for_fid = np.transpose(real_data_for_fid, (0, 2, 3, 1))

        with torch.no_grad():
            per_batch_samples = 256  # hardcoded to prevent GPU OOM

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

        fid_score = calculate_fid(
            real_data_for_fid,
            gen_data_for_fid,
            False,
            128,
            self.params["inception_checkpoint"],
        )

        return fid_score

    @torch.no_grad()
    def sample_images(self):
        student_samples = self.student(reverse=True, z=self.latent, temperature=1)[-1]
        images = postprocess(student_samples).cpu()
        grid = make_grid(images[:30], nrow=6).permute(1, 2, 0)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis("off")

    def get_histograms(self, generated, real):
        features_names = ["RichDLLe", "RichDLLk", "RichDLLmu", "RichDLLp", "RichDLLbt"]
        num_of_subplots = len(features_names)

        fig, axes = plt.subplots(num_of_subplots, 1, figsize=(10, 16))
        plt.suptitle("Histograms comparison")

        for feature_index, (feature_name, ax) in enumerate(
            zip(features_names, axes.flatten())
        ):
            ax.set_title(feature_name)

            sns.distplot(
                real[:, feature_index],
                bins=100,
                label="real",
                hist_kws={"alpha": 1.0},
                ax=ax,
                norm_hist=True,
                kde=False,
            )

            sns.distplot(
                generated[:, feature_index],
                bins=100,
                label="gen",
                hist_kws={"alpha": 0.5},
                ax=ax,
                norm_hist=True,
                kde=False,
            )

            if feature_index == 0:
                ax.legend()

    def calc_roc_auc(self, generated, real, weights):
        X = np.concatenate((generated, real))
        y = np.array([0] * generated.shape[0] + [1] * real.shape[0])
        weights = np.concatenate((weights, weights))

        (
            X_train,
            X_test,
            y_train,
            y_test,
            weights_train,
            weights_test,
        ) = train_test_split(
            X,
            y,
            weights,
            test_size=0.33,
            random_state=self.params["seed"],
            stratify=y,
            shuffle=True,
        )

        classifier = CatBoostClassifier(iterations=1000, task_type="GPU")
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        roc_auc = calculate_roc_auc(y_test, predicted, weights=weights_test)
        return roc_auc

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage: str):
        data_description = self.params["data"]
        logger.info("Creating dataset")

        if data_description["name"].lower() == "celeba":
            image_shape, num_classes, train_dataset, valid_dataset = get_CelebA(
                data_description["augment"],
                data_description["data_path"],
                data_description["download"],
            )
        elif data_description["name"].lower() == "cifar-10":
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
        logger.info("Creating train dataloader")
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        logger.info("Creating val dataloader")
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
