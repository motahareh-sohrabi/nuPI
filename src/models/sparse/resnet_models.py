"""ResNet model implementation. Heavily inspired by the PyTorch implementation.
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
We add support for `MaskedLayer` layers."""

from typing import Any, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

import src.models.sparse.layers as layers

from .base_model import BaseL0Model
from .utils import apply_bn_to_conv_out, choose_correct_bn_type, create_general_conv2d

BNLayer = Union[nn.BatchNorm2d, layers.MaskedBatchNorm2d]
Conv2dLayer = Union[nn.Conv2d, layers.StructuredL0Conv2d, layers.UnstructuredL0Conv2d]


# TODO(juan43ramirez): should adjust the normalization of the parameters
# acording tothe droprate_init.

# TODO(juan43ramirez): should add logging statements to the initialization of models


class Bottleneck(nn.Module):
    """
    Implements a Bottleneck block for ResNet models. The code in this class is an adaptation
    of the official Pytorch-vision ResNet code to allow for L0Conv2d layers.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[BNLayer] = nn.BatchNorm2d,
        conv_layer: Type[Conv2dLayer] = layers.StructuredL0Conv2d,
        masked_layer_kwargs: dict = {},  # include temp, detach and weight_dec
        masked_conv_ix: List[str] = [],
    ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.shortcut_conv layers downsample the input when stride != 1

        self.conv1 = create_general_conv2d(
            in_channels=inplanes,
            out_channels=width,
            kernel_size=(1, 1),
            stride=1,
            groups=1,
            conv_layer=conv_layer if "conv1" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn1_type = choose_correct_bn_type(self.conv1, norm_layer)
        self.bn1 = bn1_type(num_features=width)

        self.conv2 = create_general_conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=(3, 3),
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            conv_layer=conv_layer if "conv2" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn2_type = choose_correct_bn_type(self.conv2, norm_layer)
        self.bn2 = bn2_type(num_features=width)

        self.conv3 = create_general_conv2d(
            in_channels=width,
            out_channels=planes * self.expansion,
            kernel_size=(1, 1),
            stride=1,
            groups=1,
            conv_layer=conv_layer if "conv3" in masked_conv_ix else nn.Conv2d,
            masked_layer_kwargs=masked_layer_kwargs,
            use_bias=False,
        )
        bn3_type = choose_correct_bn_type(self.conv3, norm_layer)
        self.bn3 = bn3_type(num_features=planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.has_downsample = False
        if stride != 1 or inplanes != planes * self.expansion:
            self.has_downsample = True

            self.downsample_conv = create_general_conv2d(
                in_channels=inplanes,
                out_channels=planes * self.expansion,
                kernel_size=(1, 1),
                stride=stride,
                conv_layer=conv_layer if "downsample_conv" in masked_conv_ix else nn.Conv2d,
                use_bias=False,
                masked_layer_kwargs=masked_layer_kwargs,
            )

            down_bn_type = choose_correct_bn_type(self.downsample_conv, norm_layer)
            self.downsample_bn = down_bn_type(num_features=planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = apply_bn_to_conv_out(self.bn1, out)
        out = self.relu(out)

        out = self.conv2(out)
        out = apply_bn_to_conv_out(self.bn2, out)
        out = self.relu(out)

        out = self.conv3(out)
        out = apply_bn_to_conv_out(self.bn3, out)

        if self.has_downsample:
            identity = apply_bn_to_conv_out(self.downsample_bn, self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """
    Implements a Basic block for ResNet models. The code in this class is an adaptation
    of the official Pytorch-vision ResNet code to allow for L0Conv2d layers.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[BNLayer] = nn.BatchNorm2d,
        conv_layer: Type[Conv2dLayer] = layers.StructuredL0Conv2d,
        masked_layer_kwargs: dict = {},  # include temp, detach and weight_dec
        masked_conv_ix: List[str] = [],
    ):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.stride = stride

        # Both self.conv1 and self.shortcut_conv layers downsample the input when stride != 1

        self.conv1 = create_general_conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=stride,
            groups=groups,  # Default 1
            dilation=dilation,  # Default  1
            padding=dilation,  # Default  1
            conv_layer=conv_layer if "conv1" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn1_type = choose_correct_bn_type(self.conv1, norm_layer)
        self.bn1 = bn1_type(num_features=planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = create_general_conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=1,
            groups=groups,  # Default 1
            dilation=dilation,  # Default  1
            padding=dilation,  # Default  1
            conv_layer=conv_layer if "conv2" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn2_type = choose_correct_bn_type(self.conv2, norm_layer)
        self.bn2 = bn2_type(num_features=planes)

        # TODO(gallego-posada): This block has the exact same code as the
        # BottleNeck block. Consider removing the duplication.
        self.has_downsample = False
        if stride != 1 or inplanes != planes:
            self.has_downsample = True

            self.downsample_conv = create_general_conv2d(
                in_channels=inplanes * self.expansion,
                out_channels=planes,
                kernel_size=(1, 1),
                stride=stride,
                groups=1,
                conv_layer=conv_layer if "downsample_conv" in masked_conv_ix else nn.Conv2d,
                use_bias=False,
                masked_layer_kwargs=masked_layer_kwargs,
            )

            down_bn_type = choose_correct_bn_type(self.downsample_conv, norm_layer)
            self.downsample_bn = down_bn_type(num_features=planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = apply_bn_to_conv_out(self.bn1, out)
        out = self.relu(out)

        out = self.conv2(out)
        out = apply_bn_to_conv_out(self.bn2, out)

        if self.has_downsample:
            identity = apply_bn_to_conv_out(self.downsample_bn, self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out


class SparseResNet(BaseL0Model):
    """
    Adaptation of the official Pytorch-vision ResNet code to allow for masked
    layers.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


    Args:
        norm_layer: Type of batch normalization to use. Can be `nn.BatchNorm2d` or
            `MaskedBatchNorm2d`. If `MaskedBatchNorm2d`, then the *preceding*
            layer must be an instance of `L0Conv2d` layer.
        input_shape: Shape of the input tensor. This is used to determine the
            number of input channels.
        conv_layer: Type of convolutional layer to use. Must be of type
            `Conv2dLayer`.
        masked_conv_ix: List of strings indicating which Conv2d layers of each
            block are use the `conv_layer` class, as opposed to the standard
            nn.Conv2d dense layers. If this is empty, the model is fully dense.
            Valid values are "conv1" (resp. 2 or 3) for the convolutional layers
            and "downsample_conv" for the shortcut layers.
        masked_layer_kwargs: Keyword arguments to pass to the construction of
            the masked layers prescribed in `masked_conv_ix`. This can include
            configuration for gated layers such as `temperature`,
            `l2_detach_gates`, `droprate_init`, etc.
    """

    def __init__(
        self,
        conv1: nn.Module,
        block: Type[Union[Bottleneck, BasicBlock]],
        layers: List[int],
        output_size: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Type[BNLayer] = nn.BatchNorm2d,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        conv_layer: Type[Conv2dLayer] = nn.Conv2d,
        masked_conv_ix: List[str] = ["conv1", "conv2", "conv3"],
        masked_layer_kwargs: dict[str, Any] = {},  # TODO: unpack this
        is_last_fc_dense: bool = True,
    ):
        super(SparseResNet, self).__init__(
            input_shape=input_shape,
            output_size=output_size,
            droprate_init=masked_layer_kwargs.get("droprate_init", 0.01),
        )

        # We do NOT use bias in SparseResNet (like Pytorch implementation)

        assert input_shape[0] == 3
        self.input_shape = input_shape

        self._norm_layer = norm_layer
        self._conv_layer = conv_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv1

        bn1_class = choose_correct_bn_type(self.conv1, norm_layer)
        self.bn1 = bn1_class(num_features=self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_kwargs = {
            "norm_layer": self._norm_layer,
            "conv_layer": self._conv_layer,
            "masked_layer_kwargs": masked_layer_kwargs,
            "masked_conv_ix": masked_conv_ix,
        }

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            stride=1,
            dilate=False,
            block_kwargs=block_kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            block_kwargs=block_kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            block_kwargs=block_kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            block_kwargs=block_kwargs,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if is_last_fc_dense or conv_layer != layers.StructuredL0Conv2d:
            # If the model is dense, we enforce the last layer to be dense
            last_fc_layer = nn.Linear
        else:
            if conv_layer == layers.StructuredL0Conv2d:
                last_fc_layer = layers.StructuredL0Linear
            else:
                last_fc_layer = layers.UnstructuredL0Linear

        self.fc = last_fc_layer(in_features=512 * block.expansion, out_features=output_size, bias=True)

        self.pytorch_weight_initialization(zero_init_residual=zero_init_residual)

    def pytorch_weight_initialization(self, zero_init_residual: bool):
        """
        Initialize the weights following the scheme suggested in the Pytorch ResNet
        implementation.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layers.StructuredL0Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, layers.MaskedBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,  # out_planes
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        block_kwargs: dict[str, Any] = {},
    ) -> nn.Sequential:
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                **block_kwargs,
            )
        )

        # Set the inplanes of the next layer to the out_planes of the current layer
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **block_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = apply_bn_to_conv_out(self.bn1, x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if isinstance(self.fc, (layers.StructuredL0Linear, layers.UnstructuredL0Linear)):
            x, _ = self.fc(x)
        else:
            x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def CifarSparseResNet18(
    output_size: int = 200, sparsity_type: str = "structured", is_first_conv_dense: bool = True, **kwargs: Any
) -> SparseResNet:
    """Constructs a ResNet-18 model."""

    assert sparsity_type in ("structured", "unstructured")
    conv_layer = layers.StructuredL0Conv2d if sparsity_type == "structured" else layers.UnstructuredL0Conv2d

    conv1 = create_general_conv2d(
        conv_layer=nn.Conv2d if is_first_conv_dense else conv_layer,
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=False,
        masked_layer_kwargs={},  # TODO: is this correct?
    )

    resnet = SparseResNet(
        conv1=conv1, block=BasicBlock, layers=[2, 2, 2, 2], output_size=output_size, conv_layer=conv_layer, **kwargs
    )

    return resnet


def SparseResNet50(
    output_size: int = 1000, sparsity_type: str = "structured", is_first_conv_dense: bool = True, **kwargs: Any
) -> SparseResNet:
    """Constructs a ResNet-50 model."""

    assert sparsity_type in ("structured", "unstructured")
    conv_layer = layers.StructuredL0Conv2d if sparsity_type == "structured" else layers.UnstructuredL0Conv2d

    conv1 = create_general_conv2d(
        conv_layer=conv_layer if is_first_conv_dense else nn.Conv2d,
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        use_bias=False,
        masked_layer_kwargs={},
    )

    return SparseResNet(
        conv1=conv1, block=Bottleneck, layers=[3, 4, 6, 3], output_size=output_size, conv_layer=conv_layer, **kwargs
    )
