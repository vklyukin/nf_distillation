import math
import torch
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
    gaussian_likelihood,
    gaussian_sample,
)
from .utils import split_feature, uniform_binning_correction


def get_block_2d(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


def get_block_1d(in_features, out_features, hidden_features):
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
    ):
        super().__init__()
        self.is_1d = is_1d
        self.flow_coupling = flow_coupling

        if self.is_1d:
            self.actnorm = ActNorm1d(in_channels, actnorm_scale)
        else:
            self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed, is_1d=is_1d
            )
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":

            if self.is_1d:
                raise RuntimeError("Permutation is not supported is 1d mode")

            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            if self.is_1d:
                raise RuntimeError("Permutation is not supported is 1d mode")

            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            if self.is_1d:
                self.block = get_block_1d(
                    in_channels // 2, in_channels - in_channels // 2, hidden_channels
                )
            else:
                self.block = get_block_2d(
                    in_channels // 2, in_channels - in_channels // 2, hidden_channels
                )
        elif flow_coupling == "affine":
            if self.is_1d:
                self.block = get_block_1d(
                    in_channels // 2,
                    (in_channels - in_channels // 2) * 2,
                    hidden_channels,
                )
            else:
                self.block = get_block_2d(
                    in_channels // 2,
                    (in_channels - in_channels // 2) * 2,
                    hidden_channels,
                )

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = (
                torch.sum(torch.log(scale), dim=[1] + [] if self.is_1d else [2, 3])
                + logdet
            )
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = (
                -torch.sum(torch.log(scale), dim=[1] + [] if self.is_1d else [2, 3])
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
            # 1. Squeeze
            if not self.is_1d:
                C, H, W = C * 4, H // 2, W // 2

                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        is_1d=self.is_1d,
                    )
                )

                if self.is_1d:
                    self.output_shapes.append([-1, C])
                else:
                    self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1 and not self.is_1d:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
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
                + []
                if self.is_1d
                else [self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3]]
            ),
        )

    def prior(self, data, y_onehot=None):
        if data is not None:
            shape = [data.shape[0], 1] + [] if self.is_1d else [1, 1]
            h = self.prior_h.repeat(*shape)
        else:
            # Hardcoded a batch size of 32 here
            if y_onehot is not None:
                batch_size = y_onehot.size(0)
            else:
                batch_size = 32
            shape = [batch_size, 1] + [] if self.is_1d else [1, 1]
            h = self.prior_h.repeat(shape)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            shape = [h.shape[0], channels] + [] if self.is_1d else [1, 1]
            h += yp.view(*shape)
        
        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        if self.is_1d:
            b, c = x.shape
        else:
            b, c, h, w = x.shape

        if self.is_1d:
            logdet = torch.zeros(b, device=x.device, dtype=torch.float32)
        else:
            x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        if self.is_1d:
            bpd = (-objective) / (math.log(2.0) * c)
        else:
            bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, module in self.named_modules():
            if isinstance(module, ActNorm2d) or isinstance(module, ActNorm1d):
                module.inited = True
