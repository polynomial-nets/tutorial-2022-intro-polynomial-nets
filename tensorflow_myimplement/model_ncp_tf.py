'''Model for Π-net based 2nd degree blocks without activation functions:
https://ieeexplore.ieee.org/document/9353253 (or https://arxiv.org/abs/2006.13026). 

This file implements an NCP-based product of polynomials.
'''
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense , Conv2D, MaxPool2D, Dropout,BatchNormalization,AveragePooling2D
from tensorflow_addons.layers import AdaptiveAveragePooling2D,InstanceNormalization

def get_norm(norm_local):
    #TODO
    """ Define the appropriate function for normalization. """
    # if norm_local is None or norm_local == 0:
    #     return BatchNormalization(axis=3)
    # elif norm_local == 1:
    #     return InstanceNormalization(axis=3)
    # elif isinstance(norm_local, int) and norm_local < 0:
    #     return lambda a: lambda x: x
    # else:
    #     return norm_local
    if norm_local == 1:
        # print("norm1")
        return InstanceNormalization(axis=3)
    else:
        # print("norm3")
        return BatchNormalization(axis=3)

class SinglePoly(Model):
    def __init__(self, planes, stride=1, use_alpha=False, kernel_sz=3,
                 norm_S=None, norm_layer=None, kernel_size_S=1,
                 use_only_first_conv=False, **kwargs):
        """ This class implements a single second degree NCP model. """ 
        super(SinglePoly, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_S = get_norm(norm_S)
        self.use_only_first_conv = use_only_first_conv

        self.conv1 = Conv2D(planes, kernel_size=kernel_sz, strides=stride, padding='same', use_bias=False)
        self.bn1 = self._norm_layer
        if not self.use_only_first_conv:
            self.conv2 = Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
            self.bn2 = self._norm_layer

        self.conv_S = Conv2D(planes, kernel_size=kernel_size_S, strides=stride, padding='same', use_bias=False)
        self.bnS = self._norm_S

        self.use_alpha = use_alpha
        if self.use_alpha:
            # self.alpha = nn.Parameter(torch.zeros(1))
            self.alpha = self.add_weight(name='kernel', shape=1, initializer='zeros',trainable=True)
            self.monitor_alpha = []

    def call(self, x):
        out = self.bn1(self.conv1(x))
        if not self.use_only_first_conv:
            out = self.bn2(self.conv2(out))
        out1 = self.bnS(self.conv_S(x))
        out_so = out * out1
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 = out1 + out_so
        return out1


class ModelNCP(Model):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None,
                 pool_adapt=True, n_channels=[64, 128, 256, 512], **kwargs):
        super(ModelNCP, self).__init__()
        self._norm_layer = BatchNormalization(axis=3) if norm_layer is None else get_norm(norm_layer)
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = AdaptiveAveragePooling2D((1, 1))
        else:
            self.avg_pool =AveragePooling2D(pool_size=4)

        self.conv1 = Conv2D(n_channels[0], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = self._norm_layer
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, **kwargs)
        self.linear = Dense(num_classes, activation='softmax') #changed add activation

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        current_layers = []
        for stride in strides:
            current_layers.append(block(planes, stride, norm_layer=self._norm_layer, **kwargs))
        return Sequential(current_layers)

    def call(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out) #[16, 32, 32, 64]
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)#([16, 4, 4, 512])
        out = self.avg_pool(out)
        out = Flatten(input_shape=out.shape)(out)
        out = self.linear(out)
        return out

def ModelNCP_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return ModelNCP(SinglePoly, num_blocks, **kwargs)

def test():
    args = { 'num_blocks': [2, 2, 2, 1],'train': True, 'use_alpha': True, 'use_only_first_conv': 1, 'kernel_sz': 5,'norm_S': 1}
    net = ModelNCP_wrapper(**args)
    from tensorflow.keras.datasets import cifar10
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train / 255.0
    batchsize = 16
    inputs = x_train[:batchsize]
    print(inputs.shape) #(16, 32, 32, 3)
    y = net(inputs, training=True)
    print(y.shape) #(16, 10)

if __name__ == '__main__':
    test()
