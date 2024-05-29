import torch.nn as nn


def build_lateral_connection(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, input_dim, 1, 1, 0),
        nn.Conv2d(input_dim, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, output_dim, 1, 1, 0)
    )


class ConvWithActivation(nn.Module):
    def __init__(self, conv_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='relu'):
        super(ConvWithActivation, self).__init__()
        if conv_type == 'conv':
            conv_func = nn.Conv2d 
        elif conv_type == 'deconv':
            conv_func = nn.ConvTranspose2d
        self.conv2d = conv_func(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = get_activation(activation)

        for m in self.modules():
            if isinstance(m, conv_func):
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x

    
def get_activation(type):
    if type == 'leaky relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif type == 'relu':
        return nn.ReLU(inplace=True)
    elif type == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError