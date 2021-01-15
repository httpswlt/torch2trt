from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .Conv2d import *


@tensorrt_converter("torch.nn.functional.conv2d", enabled=trt_version() < '7.0')
def convert_f_conv2d(ctx):
    input_ = ctx.method_args[0]
    weights = ctx.method_args[1]
    in_channels = input_.size()[1]
    out_channels = weights.size()[0]
    kernel_size = tuple(weights.size()[2:4])
    kwargs = ctx.method_kwargs

    con = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        **kwargs)
    con.weight = torch.nn.Parameter(weights)

    ctx.method_args = (con, input_)
    convert_Conv2d(ctx)
