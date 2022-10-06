# https://github.com/Eric-mingjie/network-slimming/blob/master/models/vgg.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from Dependencies.models.BaseModelScript import BaseModel
from Dependencies import UtilsScript
from Dependencies import TrainTestScript
from Dependencies import GetDataScript

__all__ = ['VggCifar10']

default_cfg = {
    # coreset: compressed model
    # RITA: cifar10-vgg19
    # full model err 6.23 (Acc=93.77) #params=20,035,018
    # c    model err 6.02 (Acc=93.98) #params= 2,366,600 88.2%
    'debug': [4, 'M', 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4],  # for debug
    'debug_c': [2, 'M', 2, 'M', 3, 2, 'M', 4, 4, 'M', 2, 4, 'M', 3, 4],  # for debug
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # our compression goal
    '19_coreset': [49, 64, 'M', 128, 128, 'M', 256, 254, 234, 198, 'M', 114, 41, 24, 11, 'M', 14, 13, 19, 104],

}


class VggCifar10(BaseModel):
    def __init__(self, vgg_type: str, path: str):
        assert vgg_type in default_cfg, 'vgg_type({}) must be one of {}'.format(vgg_type, default_cfg.keys())
        super().__init__(path)

        # convolutions number and sizes
        cfg = default_cfg[vgg_type]
        self.feature = self.make_layers(cfg, batch_norm=True)

        num_classes = 10  # cifar10 classes
        self.classifier = nn.Linear(cfg[-1], num_classes)  # fc layer at the end

        self._initialize_weights()

        full_vgg_19_params = 20035018
        self.init(full_model_params=full_vgg_19_params)
        return

    def create_mapping(self):
        # create layers mapping:
        mapping_list, layer_list = [], []
        for name, param in self.named_parameters():
            if not name.startswith('classifier'):  # in vgg: each 3 params are a conv layer except the last which is fc
                # print(name, param.shape)
                layer_list.append(name)
                if len(layer_list) == 3:  # save layer
                    mapping_list.append(layer_list)
                    layer_list = []
            else:  # fc layer - HARD CODED FOR ONE FC LAYER
                # print(name, param.shape)
                layer_list.append(name)
                if len(layer_list) == 2:  # save layer
                    mapping_list.append(layer_list)
                    break
        return mapping_list

    @staticmethod
    def make_layers(cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        # todo add softmax? output = func.log_softmax(out, dim=1) ???
        return y


def cifar10_vgg19_example():
    """
    :return:
    """
    UtilsScript.make_cuda_invisible()
    UtilsScript.set_cuda_scope_and_seed(seed=42)

    vgg_type = '19'
    model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))
    UtilsScript.model_params_print(model)
    print(UtilsScript.model_summary_to_string(model=model, input_size=(3, 32, 32), batch_size=-1))

    ds_name = 'cifar10'
    root = '../../Datasets'
    data_limit = 0
    bs_train, shf_train = 64, False
    bs_test, shf_test = 256, False
    train_loader = GetDataScript.get_data_loader(ds_name=ds_name, data_root=root, is_train=True, bs=bs_train,
                                                 shuffle=shf_train, transform=None, data_limit=data_limit)
    test_loader = GetDataScript.get_data_loader(ds_name=ds_name, data_root=root, is_train=False, bs=bs_test,
                                                shuffle=shf_test, transform=None, data_limit=data_limit)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    TrainTestScript.train_using_dls(train_loader, test_loader, model, opt, func.cross_entropy, epochs=1, use_acc=True,
                                    save_models=False)

    TrainTestScript.test_using_dl(test_loader, model, func.cross_entropy, print_res=True)
    return


def speed_test():
    """
    CUDA can't hold on my pc
    working on CPU
    Created vgg
        #params=20,035,018, path=D:\workspace\2019SGD\Dependencies\models\vgg.pt
        dataloaders
        epoch [  1/  1] train loss:1.826263,test loss=1.846838, accuracy=3000/10000 (30.00%), time so far 00:25:46 New
        datasets
        epoch [  1/  1] train loss:1.771535,test loss=1.466372, accuracy=4099/10000 (40.99%), time so far 00:24:48 New
    notice results differ - that's due to the transform which has random in it (tried to cancel it)
    """
    UtilsScript.make_cuda_invisible()
    UtilsScript.set_cuda_scope_and_seed(seed=42)
    vgg_type = '19_coreset'
    # vgg_type = 'debug_c'
    m1 = VggCifar10(vgg_type, path='./vgg_{}_1.pt'.format(vgg_type))
    m2 = UtilsScript.clone_model(m1)
    m2.set_new_path('./vgg_{}_2.pt')
    ADAM_OPT_V2 = {
        'name': 'ADAM',
        'lr': 0.001,
        'weight_decay': 0.0001
    }

    TrainTestScript.speed_test_datasets_vs_dataloaders(m1=m1, m2=m2, f=func.cross_entropy, ds_name='cifar10',
                                                       ds_root='../../Datasets', seed=42, bs_train=64, bs_test=256,
                                                       data_limit=0, epochs=1, opt_d=ADAM_OPT_V2, b_interval=100)
    return


def sim_forward(vgg_type):
    model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))
    batch_x = UtilsScript.torch_normal(shape=(16, 3, 32, 32), miu=0, std=10)
    y_pred = model(batch_x)
    print(UtilsScript.var_to_string(batch_x, 'batch_x', with_data=False))
    print(UtilsScript.var_to_string(y_pred, 'y_pred', with_data=False))
    return


def freeze_layer_except(trainable_layers_indices: list):
    vgg_type = '19_coreset'
    c_model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))

    # print mapping layer ind to layer components
    map_d = c_model.get_mapping_dict()
    for layer_ind, layer_components in map_d.items():
        print('layer {}: {}'.format(layer_ind, layer_components))

    print('layers_indices_to_train {} test:'.format(trainable_layers_indices))
    trainable_layers_values = c_model.get_layers_components(trainable_layers_indices)
    UtilsScript.freeze_layers(c_model, trainable_layers=trainable_layers_values)
    UtilsScript.model_params_print(c_model)
    return


def copy_layers_except():
    vgg_type = '19'
    p_model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))
    vgg_type = '19_coreset'
    c_model = VggCifar10(vgg_type, path='./vgg_{}.pt'.format(vgg_type))

    # print mapping layer ind to layer components
    map_d = c_model.get_mapping_dict()
    for layer_ind, layer_components in map_d.items():
        print('layer {}: {}'.format(layer_ind, layer_components))

    # in this setup, we can copy only layers 3,4,5
    # to copy we need the layer to be in the same size AND size of the previous layer (size mismatch otherwise)
    # e.g.
    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512] - vgg19
    # [XX, XX, 'M', VVV, VVV, 'M', VVV, XXX, XXX, XXX, 'M', XXX, XXX, XXX, XXX, 'M', XXX, XXX, XXX, XXX]
    # [49, 64, 'M', 128, 128, 'M', 256, 254, 234, 198, 'M', 114,  41,  24,  11, 'M',  14,  13,  19, 104] - vgg19_coreset
    # on both models: layer 2 is the same size BUT layer 1 is different - CAN'T COPY
    #       on p_model |layer 2| = [64, 64, 3, 3]
    #       on c_model |layer 2| = [64, 49, 3, 3]  # the 49 is the input of layer 2 (or the output of layer 1)
    # on both models: layer 3 is the same size AND layer 2 is the same - CAN COPY
    #       on p_model |layer 3| = [128, 64, 3, 3]
    #       on c_model |layer 3| = [128, 64, 3, 3]

    all_indices = list(map_d.keys())
    copy_indices = [3, 4]  # we can also add 5
    dont_copy_indices = [index for index in all_indices if index not in copy_indices]
    print('copy_indices {} (c_model.get_mapping_dict().keys() - copy_indices = dont_copy_indices=={})'.format(
        copy_indices, dont_copy_indices))
    dif_size_layers_values = c_model.get_layers_components(dont_copy_indices)
    UtilsScript.copy_models(p_model, c_model, dont_copy_layers=dif_size_layers_values, ack=True)
    return


if __name__ == '__main__':
    # sim_forward(vgg_type='19')
    # sim_forward(vgg_type='19_coreset')
    # cifar10_vgg19_example()
    speed_test()
    # freeze_layer_except(trainable_layers_indices=[1, 2, 3])
    # copy_layers_except()
