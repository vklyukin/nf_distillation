import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, checkpoint_path=None):
        super(VGGPerceptualLoss, self).__init__()

        if checkpoint_path is None:
            vgg_model = torchvision.models.vgg16(pretrained=True)
        else:
            vgg_model = torchvision.models.vgg16(pretrained=False)
            vgg_model_state = torch.load(checkpoint_path)
            vgg_model.load_state_dict(vgg_model_state)

        vgg_features = vgg_model.features

        blocks = [
            vgg_features[:4].eval(),
            vgg_features[4:9].eval(),
            vgg_features[9:16].eval(),
            vgg_features[16:23].eval(),
        ]

        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

        self.mean = torch.nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.std = torch.nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )

        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)

        return loss
