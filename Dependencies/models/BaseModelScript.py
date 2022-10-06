import torch.nn as nn
import os
from Dependencies import UtilsScript
import abc


class BaseModel(nn.Module):
    def __init__(self, path):
        super(BaseModel, self).__init__()
        self.path = os.path.abspath(path)
        self.title = os.path.basename(os.path.normpath(self.path)).replace('.pt', '')
        self.params = 0
        self.__mapping = [['enumerate from 1']]  # ignore index 0
        return

    def init(self, full_model_params: int = 1):
        self.params = UtilsScript.model_params_count(self)
        self.__mapping += self.create_mapping()  # already a list with item 0 ignored
        self.__model_info(full_model_params)  # call last - uses params and __mapping
        return

    def __model_info(self, full_model_params):
        if self.params == full_model_params:  # original size LeNet_300_100
            print('Created {}'.format(self.title))
            print('\t#params={:7,.0f}, path={}'.format(self.params, self.path))
        else:  # compression
            print('Created {}: path={}'.format(self.title, self.path))
            msg = '\tthis model #params={:,.0f}, full model #params={:,.0f}, compression ratio={:,.2f}%'
            print(msg.format(self.params, full_model_params, 100. * (1 - self.params / full_model_params)))
        mapping = self.get_mapping_dict()
        if len(mapping) > 0:
            print('\tmapping: {}'.format(mapping))
        return

    @abc.abstractmethod
    def create_mapping(self) -> list:
        """
        e.g. lenet 300 100 had 6 layers which are really 3 fc layers.
        this func create a list that maps layer index to it's layer components:
        in LeNet 300 100:
            list[1] = ['fc1.weight', 'fc1.bias']
            list[2] = ['fc2.weight', 'fc2.bias']
            list[3] = ['fc3.weight', 'fc3.bias']
        :return:
        """
        print('abc.abstractmethod')
        exit(22)
        return

    def get_mapping_dict(self) -> dict:
        mapping_dict = {}
        for i, layer_component in enumerate(self.__mapping):
            if i > 0:  # ignore first
                mapping_dict[i] = layer_component
        return mapping_dict

    def set_new_path(self, new_path: str):
        self.path = os.path.abspath(new_path)
        self.title = os.path.basename(os.path.normpath(self.path)).replace('.pt', '')
        return

    def get_layers_components(self, layer_indices: list) -> list:
        """
        let's say you need to compress layer 1. so the out of layer1 (which is the in of layer2)
        changes. call model.get_layers_components([1,2]) and get in LeNet 300 100:
        ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
        """
        total_names_list = []
        for layer_index in layer_indices:
            if layer_index > 0:  # ignore first if comes by mistake
                total_names_list += self.__mapping[layer_index]
        return total_names_list
