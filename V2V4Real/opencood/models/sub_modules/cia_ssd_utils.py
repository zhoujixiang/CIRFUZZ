import torch
from torch import nn


class SSFA(nn.Module):
    def __init__(self, args):
        super(SSFA, self).__init__()
        feature_num = args['feature_num']
        hidden_num = args.get('hidden_num', feature_num * 2)

        seq = [nn.ZeroPad2d(1)]
        seq += get_conv_layers('Conv2d',
                               feature_num,
                               feature_num,
                               n_layers=3,
                               kernel_size=[3, 3, 3],
                               stride=[1, 1, 1],
                               padding=[0, 1, 1],
                               sequential=False)
        self.bottom_up_block_0 = nn.Sequential(*seq)
        self.bottom_up_block_1 = get_conv_layers('Conv2d',
                                                 feature_num,
                                                 hidden_num,
                                                 n_layers=3,
                                                 kernel_size=[3, 3, 3],
                                                 stride=[2, 1, 1],
                                                 padding=[1, 1, 1])

        self.trans_0 = get_conv_layers('Conv2d',
                                       feature_num,
                                       feature_num,
                                       n_layers=1,
                                       kernel_size=[1],
                                       stride=[1],
                                       padding=[0])
        self.trans_1 = get_conv_layers('Conv2d',
                                       hidden_num,
                                       hidden_num,
                                       n_layers=1,
                                       kernel_size=[1],
                                       stride=[1],
                                       padding=[0])

        self.deconv_block_0 = get_conv_layers('ConvTranspose2d',
                                              hidden_num,
                                              feature_num,
                                              n_layers=1,
                                              kernel_size=[3],
                                              stride=[2],
                                              padding=[1],
                                              output_padding=[1])
        self.deconv_block_1 = get_conv_layers('ConvTranspose2d',
                                              hidden_num,
                                              feature_num,
                                              n_layers=1,
                                              kernel_size=[3],
                                              stride=[2],
                                              padding=[1],
                                              output_padding=[1])

        self.conv_0 = get_conv_layers('Conv2d',
                                      feature_num,
                                      feature_num,
                                      n_layers=1,
                                      kernel_size=[3],
                                      stride=[1],
                                      padding=[1])
        self.conv_1 = get_conv_layers('Conv2d',
                                      feature_num,
                                      feature_num,
                                      n_layers=1,
                                      kernel_size=[3],
                                      stride=[1],
                                      padding=[1])

        self.w_0 = get_conv_layers('Conv2d',
                                   feature_num,
                                   1,
                                   n_layers=1,
                                   kernel_size=[1],
                                   stride=[1],
                                   padding=[0],
                                   relu_last=False)
        self.w_1 = get_conv_layers('Conv2d',
                                   feature_num,
                                   1,
                                   n_layers=1,
                                   kernel_size=[1],
                                   stride=[1],
                                   padding=[0],
                                   relu_last=False)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1),
                                 dim=1)

        x_output = x_output_0 * x_weight[:, 0:1, :, :] + \
            x_output_1 * x_weight[:, 1:, :, :]
        return x_output.contiguous()


def get_conv_layers(conv_name,
                    in_channels,
                    out_channels,
                    n_layers,
                    kernel_size,
                    stride,
                    padding,
                    relu_last=True,
                    sequential=True,
                    **kwargs):
    layers = []
    for idx in range(n_layers):
        layers.extend([
            getattr(nn, conv_name)(
                in_channels,
                out_channels,
                kernel_size[idx],
                stride=stride[idx],
                padding=padding[idx],
                bias=False,
                **{key: value[idx] for key, value in kwargs.items()}
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        ])
        if idx < n_layers - 1 or relu_last:
            layers.append(nn.ReLU())
        in_channels = out_channels

    if sequential:
        return nn.Sequential(*layers)
    return layers
