import torch
import torch.nn as nn
from blocks import CALayer, sub_pixel

class RCAB(nn.Module):
    def __init__(self, num_features, upscale_factor, reduction, act_type, norm_type):
        super(RCAB, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12
        output_padding = 0

        self.ca = CALayer(num_features, reduction)
        self.last_hidden = None
        self.up = nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        self.down = nn.Conv2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.conv_1 = nn.Conv2d(num_features*2, num_features, kernel_size=1)
        self.conv_2 = nn.Conv2d(num_features*2, num_features, kernel_size=1)
        self.conv_3 = nn.Conv2d(num_features*3, num_features, kernel_size=1)
        self.conv_4 = nn.Conv2d(num_features*4, num_features, kernel_size=1)
        self.conv_5 = nn.Conv2d(num_features*5, num_features, kernel_size=1)
        self.conv_l = nn.Conv2d(num_features*6, num_features, kernel_size=1)
        self.conv_h = nn.Conv2d(num_features*5, num_features, kernel_size=1)
        self.act = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x):
        
        # l0 = self.conv_1(x)
        l0 = x
        h1 = self.act(self.up(l0))
        l1 = self.act(self.down(h1))
        h2 = self.act(self.up(self.conv_2(torch.cat((l0, l1),1))))
        l2 = self.act(self.down(self.conv_2(torch.cat((h1, h2),1))))
        h3 = self.act(self.up(self.conv_3(torch.cat((l0, l1, l2),1))))
        l3 = self.act(self.down(self.conv_3(torch.cat((h1, h2, h3),1))))
        h4 = self.act(self.up(self.conv_4(torch.cat((l0, l1, l2, l3),1))))
        l4 = self.act(self.down(self.conv_4(torch.cat((h1, h2, h3, h4),1))))
        h5 = self.act(self.up(self.conv_5(torch.cat((l0, l1, l2, l3, l4),1))))
        l5 = self.act(self.down(self.conv_5(torch.cat((h1, h2, h3, h4, h5),1))))

        l = self.conv_l(torch.cat((l0, l1, l2, l3, l4, l5),1))
        h = self.conv_h(torch.cat((h1, h2, h3, h4, h5), 1))
        output_l = self.ca(l)
        output_h = self.ca(h)

        return output_l, output_h

    def reset_state(self):
        self.should_reset = True

class CAFRN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, reduction, upscale_factor, act_type = 'prelu', norm_type = None):
        super(CAFRN, self).__init__()

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        #Initial Feature Extraction
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        self.feat_in = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)

        # basic block
        self.block = RCAB(num_features, upscale_factor, reduction, act_type, norm_type)
        self.conv_out = nn.Conv2d(num_features*num_steps, num_features, kernel_size=3, padding=1, bias=True)

        # Upsampler
        self.conv_up = nn.Conv2d(num_features, num_features*upscale_factor*upscale_factor, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(upscale_factor)
        # conv 
        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):

        x = self.conv_in(x)
        x = self.feat_in(x)

        out1_l, out1_h = self.block(x)
        out2_l, out2_h = self.block(out1_l)
        out3_l, out3_h = self.block(out2_l)
        out4_l, out4_h = self.block(out3_l)
        out5_l, out5_h = self.block(out4_l)
        out6_l, out6_h = self.block(out5_l)
        out7_l, out7_h = self.block(out6_l)
        outs_l = torch.cat((out1_l, out2_l, out3_l, out4_l, out5_l, out6_l, out7_l), 1)
        outs_h = torch.cat((out1_h, out2_h, out3_h, out4_h, out5_h, out6_h, out7_h), 1)

        out_l = self.conv_out(outs_l)
        out_h = self.conv_out(outs_h)

        out = out_l + x

        y = self.conv_up(out)
        y = self.upsample(y)
        y = self.conv3(y+out_h)
        return y 

