import math
import torch
import torch.nn as nn
import typing as tp

from .flows import FlowNet, Glow, FlowStep
from .layers import Split2d, gaussian_likelihood
from .utils import uniform_binning_correction


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
        )

    def encode(self, z, logdet=0.0):
        all_outputs = []
        layer_input = z

        for layer, shape in zip(self.layers, self.output_shapes):
            layer_output, logdet = layer(layer_input, logdet, reverse=False)
            all_outputs.append(layer_output)
            layer_input = layer_output

        return all_outputs, logdet

    def decode(self, z, temperature=None):
        all_outputs = []
        layer_input = z

        for layer in reversed(self.layers):

            if isinstance(layer, Split2d):
                layer_output, logdet = layer(
                    layer_input, logdet=0, reverse=True, temperature=temperature
                )
            else:
                layer_output, logdet = layer(layer_input, logdet=0, reverse=True)

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
        )

        self.flow = FlowNetGetAllOutputs(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        last_z = z[-1]

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, last_z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits


def create_glow_model(config: tp.Dict[str, tp.Any]) -> GlowGetAllOutputs:
    model = GlowGetAllOutputs(**config)
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
    current_sign_s = None
    for teacher_layer_id, teacher_layer in enumerate(teacher.flow.layers):
        if teacher_layer_id == teacher_kd_indices[current_common_layer_index]:
            student.flow.layers[
                student_kd_indices[current_common_layer_index]
            ].invconv.p = (current_permutation_matrix @ teacher_layer.invconv.p)
            student.flow.layers[
                student_kd_indices[current_common_layer_index]
            ].invconv.sign_s = (current_sign_s * teacher_layer.invconv.sign_s)
            current_common_layer_index += 1

            current_permutation_matrix = None
            current_sign_s = None
        elif isinstance(teacher_layer, FlowStep):
            if current_permutation_matrix is None:
                current_permutation_matrix = teacher_layer.invconv.p
                current_sign_s = teacher_layer.invconv.sign_s
            else:
                current_permutation_matrix @= teacher_layer.invconv.p
                current_sign_s *= teacher_layer.invconv.sign_s
