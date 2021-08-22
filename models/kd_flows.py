import logging
import math
import torch
import torch.nn as nn
import typing as tp

from .flows import FlowNet, Glow, FlowStep
from .layers import Split2d, gaussian_likelihood
from .utils import uniform_binning_correction


logger = logging.getLogger(__name__)


class FlowNetGetAllOutputs(FlowNet):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        is_1d=False,
        condition_features=0,
    ):
        super().__init__(
            image_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            is_1d=is_1d,
            condition_features=condition_features,
        )

    def encode(self, z, y_onehot=None, logdet=0.0):
        all_outputs = []
        layer_input = z

        for layer, shape in zip(self.layers, self.output_shapes):
            layer_output, logdet = layer(
                layer_input, y_onehot=y_onehot, logdet=logdet, reverse=False
            )
            all_outputs.append(layer_output)
            layer_input = layer_output

        return all_outputs, logdet

    def decode(self, z, y_onehot=None, temperature=None):
        all_outputs = []
        layer_input = z

        for layer in reversed(self.layers):

            if isinstance(layer, Split2d):
                layer_output, logdet = layer(
                    layer_input, logdet=0, reverse=True, temperature=temperature
                )
            else:
                layer_output, logdet = layer(
                    layer_input, y_onehot=y_onehot, logdet=0, reverse=True
                )

            all_outputs.append(layer_output)
            layer_input = layer_output

        return all_outputs


class GlowGetAllOutputs(Glow):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        y_classes,
        learn_top,
        y_condition,
        is_1d=False,
    ):
        super().__init__(
            image_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            y_classes,
            learn_top,
            y_condition,
            is_1d=is_1d,
        )

        logger.info("Creating KD flow")
        self.flow = FlowNetGetAllOutputs(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            is_1d=is_1d,
            condition_features=y_classes if y_condition else 0,
        )

    def normal_flow(self, x, y_onehot):
        if self.is_1d:
            b, c = x.shape
        else:
            b, c, h, w = x.shape

        if self.is_1d:
            logdet = torch.zeros(b, device=x.device, dtype=torch.float32)
        else:
            x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, y_onehot=y_onehot, logdet=logdet, reverse=False)

        last_z = z[-1]

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, last_z)

        if self.y_condition:
            if not self.is_1d:
                last_z = last_z.mean(dim=[2, 3])
            y_logits = self.project_class(last_z)
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        if self.is_1d:
            bpd = -objective
        else:
            bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits


def create_glow_model(
    config: tp.Dict[str, tp.Any], is_inited: bool
) -> GlowGetAllOutputs:
    model = GlowGetAllOutputs(**config)
    logger.info("Setting actnorm up")

    if is_inited:
        model.set_actnorm_init()

    return model


def inherit_permutation_matrix(
    student,
    teacher,
    student_kd_indices,
    teacher_kd_indices,
):
    current_common_layer_index = 0
    current_permutation_matrix = None
    for teacher_layer_id, teacher_layer in enumerate(teacher.flow.layers):
        if teacher_layer_id == teacher_kd_indices[current_common_layer_index]:
            student.flow.layers[
                student_kd_indices[current_common_layer_index]
            ].invconv.p = (current_permutation_matrix @ teacher_layer.invconv.p)
            current_common_layer_index += 1

            current_permutation_matrix = None
        elif isinstance(teacher_layer, FlowStep):
            if current_permutation_matrix is None:
                current_permutation_matrix = teacher_layer.invconv.p
            else:
                current_permutation_matrix @= teacher_layer.invconv.p
