"""
@author Zach Stoebner
@date November 2019
@details An U-Net AE implementation with the encoder and decoder separated
"""

from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import functools

class UnetEncoder(nn.Module):

    """
    U-Net Encoder

    This class constructs a U-Net encoder aka the first half of the 'U' and saves the
    multiscale outputs into a list which will be passed to the decoder. Downsampling is executed
    via MaxPool layers, as per the original U-Net paper, and doesn't occur via a stride=2 in the
    Conv2d layers (how CycleGAN implements U-Net). This way, image dims are preserved at each scale
    and there aren't major tensor mismatches in the decoder.

    Inputs:
    <n_downsample>      Number of multiscale levels (=n_upsample in decoder)
    <input_dim>         Number of input channels (original vocab of the UNIT authors,
                        doesn't make sense to me why they don't use input_nc for clarity that this
                        refers to channels b/c torch doesn't require manually entering the image
                        dims --> I get that C_in is the second dim in the data tensor)
    <norm_layer>        Normalization function to apply to inner levels (default=BatchNorm2d)
    """

    def __init__(self,n_downsample, input_dim, norm_layer=nn.BatchNorm2d):
        super(UnetEncoder,self).__init__()
        self.cnns = nn.ModuleList()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #First layer
        inner_nc = 64
        outer_nc = input_dim
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=1, padding=1, bias=use_bias)
        outer_nc = inner_nc
        # down = [downconv,nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
        #                      stride=1, padding=1, bias=use_bias )]
        down = [downconv]
        first = nn.Sequential(*down)
        self.cnns.append(first)

        #Following layers
        outer_nc = inner_nc
        downrelu = nn.LeakyReLU(0.2)

        # per the original U-Net impl --> not used for downsampling in cycle_gan
        downsample = nn.MaxPool2d(kernel_size=4,stride=2)
        for i in range(n_downsample-1):
            downnorm = norm_layer(inner_nc)
            block1 = nn.Conv2d(outer_nc, inner_nc * 2, kernel_size=4,
                                 stride=1, padding=1, bias=use_bias)

            inner_nc *= 2
            outer_nc = inner_nc
            # block2 = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
            #                      stride=1, padding=1, bias=use_bias)

            # innermost layer
            if i == n_downsample-2:
                #block = [downrelu, block1,block2]
                block = [downrelu, downsample, block1]
            else:
                #block = [downrelu, block1,block2, downnorm]
                block = [downrelu, downsample, block1, downnorm]
            self.cnns.append(nn.Sequential(*block))

        # setting this field so that the decoder knows what the encoding dim is at the innermost layer
        self.output_dim = inner_nc

    ### NOTE Mulit-scale forward prop
    def forward(self, x):
        outputs = []
        prev = x

        # last output is the undercomplete encoding
        with torch.autograd.set_detect_anomaly(True):
            for model in self.cnns:
                prev = model(prev)
                outputs.append(prev)

            return outputs

class UnetDecoder(nn.Module):

    """
    U-Net Decoder

    This class implements the U-Net decoder aka the second of the 'U'. The dedoder primarily
    upsamples using a stride=2 via ConvTranspose layers. Since MaxPool layers are used in the encoder,
    there are slight differences in the image dims at each level so in the forward prop I interpolate
    the output up about 5-10 pixels to account for those differences and compile cleanly. The only
    layer that differs is the last layer which is Conv2d with stride=1 (MAY CHANGE to ConvTranspose2d
    w/ stride=1 to deconv better pending current experiment). A problem that was previously encountered
    was inplace Relu which caused gradient calc in trainer.py to throw an runtime error --> cannot set
    inplace=True for an torch modules in UNIT b/c decoder and encoder are separate here.

    Inputs:
    <n_upsample>        Number of multiscale levels (=n_downsample in decoder)
    <input_dim>         Number of input channels (original vocab of the UNIT authors,
                        doesn't make sense to me why they don't use input_nc for clarity that this
                        refers to channels b/c torch doesn't require manually entering the image
                        dims --> I get that C_in is the second dim in the data tensor)
    <norm_layer>        Normalization function to apply to inner levels (default=BatchNorm2d)
    """

    def __init__(self,n_upsample, input_dim, dim, norm_layer=nn.BatchNorm2d):
        super(UnetDecoder,self).__init__()
        self.cnns = nn.ModuleList()
        self.n_upsample = n_upsample
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #Lowest layer --> will just upconv the undercomplete encoding
        inner_nc = input_dim
        outer_nc = int(inner_nc // 2)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        #upsamp = nn.Upsample(scale_factor=2)
        #up = [uprelu,upconv,upnorm,upsamp]
        up = [uprelu,upconv,upnorm]
        lowest = nn.Sequential(*up)
        self.cnns.append(lowest)

        # Remainingg upsamples --> have to add corresponding outputs
        for i in range(n_upsample-1):
            inner_nc = outer_nc
            outer_nc = int(outer_nc // 2)
            upnorm = norm_layer(outer_nc)

            if i == n_upsample-2:
                # have to double inner_nc to account for addition of skip connections
                # using a normal convolution b/c conv2dTranspose matches MaxPool2d downsample in encoder
                ###NOTE try changing this to  ConvTranspose2d w/ stride=1 to deconv better
                upconv = nn.ConvTranspose2d(inner_nc * 2, dim,
                                        kernel_size=4, stride=1,
                                        padding=1)
                #block = [uprelu,upconv,upsamp,nn.Tanh()]
                block = [uprelu,upconv,nn.Tanh()]
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
                #block = [uprelu,upconv,upnorm,upsamp]
                block = [uprelu,upconv,upnorm]
            self.cnns.append(nn.Sequential(*block))

    ### NOTE Mulit-scale forward prop
    def forward(self, x):

        # reversing for innermost layer first
        # can't just reverse in encoder b/c trainer uses encoding for loss computes

        # n_upsample = n_downsample = len(prev_outs) = len(self.cnns)
        with torch.autograd.set_detect_anomaly(True):
            inner = True
            output = None
            for model,enc in zip(self.cnns,reversed(x)):
                if inner:

                    output = model(enc)
                    inner = False
                else:
                    # have to do a small upsample (5-10 pixels) to account for minor differences b/w ConvTranspose layers and Conv2d layers
                    output = nn.functional.interpolate(output,size=enc.size()[-2:])
                    output = model(torch.cat([enc, output], 1))

                # small upsample to handle minor pixel differences
                output = nn.functional.interpolate(output,size=torch.Size([256,256]))
            return output
