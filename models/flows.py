import logging
import math
import torch
import torch.distributions as D
import torch.nn as nn

from .layers import (
    Conv2d,
    Conv2dZeros,
    ActNorm1d,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    SqueezeLayer,
    Split2d,
    MADE,
    BatchNormFlow,
    Reverse,
    gaussian_likelihood,
    gaussian_sample,
)
from .utils import split_feature, uniform_binning_correction


logger = logging.getLogger(__name__)


def get_block_2d(in_channels, out_channels, hidden_channels):
    logger.info("Creating 2D block")
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


def get_block_1d(in_features, out_features, hidden_features):
    logger.info("Creating 1D block")
    block = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_features, hidden_features),
        nn.Tanh(),
        nn.Linear(hidden_features, out_features),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        is_1d=False,
        condition_features=0,
    ):
        super().__init__()
        self.is_1d = is_1d
        self.flow_coupling = flow_coupling
        self.flow_permutation_type = flow_permutation
        self.condition_features = condition_features

        logger.info("Creating ActNorm")
        if self.is_1d:
            self.actnorm = ActNorm1d(in_channels, actnorm_scale)
        else:
            self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        logger.info("Creating permutation")
        if self.flow_permutation_type == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed, is_1d=is_1d
            )
        elif self.flow_permutation_type == "shuffle":

            if self.is_1d:
                raise RuntimeError("Permutation is not supported is 1d mode")

            self.shuffle = Permute2d(in_channels, shuffle=True)
        else:
            if self.is_1d:
                raise RuntimeError("Permutation is not supported is 1d mode")

            self.reverse = Permute2d(in_channels, shuffle=False)

        # 3. coupling
        logger.info("Creating coupling")
        if flow_coupling == "additive":
            in_block_channels = in_channels // 2 + condition_features
            out_block_channels = in_channels - in_channels // 2
            if self.is_1d:
                self.block = get_block_1d(
                    in_block_channels, out_block_channels, hidden_channels
                )
            else:
                self.block = get_block_2d(
                    in_block_channels, out_block_channels, hidden_channels
                )
        elif flow_coupling == "affine":
            in_block_channels = in_channels // 2 + condition_features
            out_block_channels = (in_channels - in_channels // 2) * 2
            if self.is_1d:
                self.block = get_block_1d(
                    in_block_channels,
                    out_block_channels,
                    hidden_channels,
                )
            else:
                self.block = get_block_2d(
                    in_block_channels,
                    out_block_channels,
                    hidden_channels,
                )
        else:
            raise NameError(f"Unknown coupling type: {flow_coupling}")

    def flow_permutation(self, z, logdet, rev):
        if self.flow_permutation_type == "invconv":
            return self.invconv(z, logdet, rev)
        elif self.flow_permutation_type == "shuffle":
            return self.shuffle(z, rev), logdet
        else:
            return self.reverse(z, rev), logdet

    def forward(self, input, y_onehot=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, y_onehot=y_onehot, logdet=logdet)
        else:
            return self.reverse_flow(input, y_onehot=y_onehot, logdet=logdet)

    def normal_flow(self, input, y_onehot, logdet):
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")

        if y_onehot is not None:
            coupling_argument = torch.cat((z1, y_onehot), dim=1)
        else:
            coupling_argument = z1

        if self.flow_coupling == "additive":
            z2 = z2 + self.block(coupling_argument)
        elif self.flow_coupling == "affine":
            h = self.block(coupling_argument)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = (
                torch.sum(torch.log(scale), dim=[1] + ([] if self.is_1d else [2, 3]))
                + logdet
            )
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, y_onehot, logdet):
        # 1.coupling
        z1, z2 = split_feature(input, "split")

        if self.condition_features:
            coupling_argument = torch.cat((z1, y_onehot), dim=1)
        else:
            coupling_argument = z1

        if self.flow_coupling == "additive":
            z2 = z2 - self.block(coupling_argument)
        elif self.flow_coupling == "affine":
            h = self.block(coupling_argument)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = (
                -torch.sum(torch.log(scale), dim=[1] + ([] if self.is_1d else [2, 3]))
                + logdet
            )
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
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
        super().__init__()

        self.is_1d = is_1d

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        if not self.is_1d:
            H, W, C = image_shape
        else:
            C = image_shape[0]

        for i in range(L):
            logger.info(f"Creating {i} block")
            # 1. Squeeze
            if not self.is_1d:
                C, H, W = C * 4, H // 2, W // 2

                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for j in range(K):
                logger.info(f"Creating {j} Flowstep")
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        is_1d=self.is_1d,
                        condition_features=condition_features,
                    )
                )

                if self.is_1d:
                    self.output_shapes.append([-1, C])
                else:
                    self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1 and not self.is_1d:
                logger.info(f"Adding Split2d")
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(
        self, input, y_onehot=None, logdet=0.0, reverse=False, temperature=None
    ):
        if reverse:
            return self.decode(input, y_onehot=y_onehot, temperature=temperature)
        else:
            return self.encode(input, y_onehot=y_onehot, logdet=logdet)

    def encode(self, z, y_onehot=None, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, y_onehot=y_onehot, logdet=logdet, reverse=False)
        return z, logdet

    def decode(self, z, y_onehot=None, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(
                    z,
                    logdet=0,
                    reverse=True,
                    temperature=temperature,
                )
            else:
                z, logdet = layer(z, y_onehot=y_onehot, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
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
        super().__init__()
        logger.info("Creating FlowNet")
        self.flow = FlowNet(
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
        self.is_1d = is_1d

        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]

            if self.is_1d:
                self.learn_top_fn = LinearZeros(C * 2, C * 2)
            else:
                self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                ]
                + (
                    []
                    if self.is_1d
                    else [
                        self.flow.output_shapes[-1][2],
                        self.flow.output_shapes[-1][3],
                    ]
                )
            ),
        )

    def prior(self, data, y_onehot=None):
        if data is not None:
            shape = [data.shape[0], 1] + ([] if self.is_1d else [1, 1])
            h = self.prior_h.repeat(*shape)
        else:
            # Hardcoded a batch size of 32 here
            if y_onehot is not None:
                batch_size = y_onehot.size(0)
            else:
                batch_size = 32
            shape = [batch_size, 1] + ([] if self.is_1d else [1, 1])
            h = self.prior_h.repeat(shape)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            shape = [h.shape[0], channels] + ([] if self.is_1d else [1, 1])
            h += yp.view(*shape)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot=None):
        if self.is_1d:
            b, c = x.shape
        else:
            b, c, h, w = x.shape

        if self.is_1d:
            logdet = torch.zeros(b, device=x.device, dtype=torch.float32)
        else:
            x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, y_onehot=y_onehot, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            if not self.is_1d:
                z = z.mean(dim=[2, 3])
            y_logits = self.project_class(z)
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        if self.is_1d:
            bpd = -objective
        else:
            bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        if z is None:
            mean, logs = self.prior(z, y_onehot)
            z = gaussian_sample(mean, logs, temperature)
        x = self.flow(z, y_onehot=y_onehot, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, module in self.named_modules():
            if isinstance(module, ActNorm2d) or isinstance(module, ActNorm1d):
                module.inited = True


# --------------------
# MAF
# --------------------


class MAF(nn.Module):
    """Masked Autoregressive Flow Model"""

    def __init__(self, input_shape, num_blocks, hidden_channels):
        super().__init__()
        self.register_buffer("placeholder", torch.randn(1))
        self.num_inputs = input_shape[0]

        flow = []
        for _ in range(num_blocks):
            flow += [
                MADE(self.num_inputs, hidden_channels),
                BatchNormFlow(self.num_inputs),
                Reverse(self.num_inputs),
            ]

        self.flow = nn.ModuleList(flow)

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=self.placeholder.device)

        for flow in self.flow:
            x, ld = flow.forward(x, y_onehot=y_onehot)
            log_det += ld

        mean, logs = self.prior(x)
        z, prior_logprob = x, gaussian_likelihood(mean, logs, x)

        log_prob = prior_logprob + log_det

        return z, -log_prob, None

    def reverse_flow(self, z, y_onehot=None, temperature=None):
        if z is None:
            mean, logs = self.prior(z)
            z = gaussian_sample(mean, logs, temperature)

        m, _ = z.shape

        for flow in reversed(self.flow):
            z, _ = flow(z, y_onehot=y_onehot, reverse=True)

        x = z
        return x

    def prior(self, data, *args):
        if data is None:
            batch_size = 32
        else:
            batch_size = data.size(0)

        device = self.placeholder.device
        means = torch.zeros(batch_size, self.num_inputs, device=device)
        logs = torch.zeros(batch_size, self.num_inputs, device=device)

        return means, logs
