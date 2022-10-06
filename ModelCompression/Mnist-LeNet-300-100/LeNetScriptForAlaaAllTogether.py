import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import torch.utils.data as tu_data
import torchvision
from torchsummary import summary
import numpy as np
import os
import abc
import time
import math
from typing import Callable

DEFAULT_MNIST_TR = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

DEFAULT_CIFAR10_TR_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

DEFAULT_CIFAR10_TR_TEST = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


def var_to_string(var: torch.Tensor, title: str = 'XXX', with_data: bool = False) -> str:
    """  to string of a torch tensor or numpy """
    if isinstance(var, torch.Tensor):
        msg = '{:8s}: {:8s}, dtype:{}, trainable:{}, is_cuda:{}'
        msg = msg.format(title, str(var.shape), var.dtype, var.requires_grad, var.is_cuda)
    else:  # assuming numpy
        msg = '{:8s}: {:8s}, dtype:{}'.format(title, str(var.shape), var.dtype)
    if with_data:
        msg += ', data: {}'.format(np.round(var.tolist(), 2).tolist())
        # msg += ', data: {}'.format(var.tolist())
    return msg


def torch_uniform(shape: tuple, range_low: float, range_high: float) -> torch.Tensor:
    ret = torch.empty(shape).uniform_(range_low, range_high)
    return ret


def torch_normal(shape: tuple, miu: float, std: float) -> torch.Tensor:
    ret = torch.empty(shape).normal_(miu, std)
    return ret


def tensor_total_size(t: torch.Tensor, ignore_first=True) -> int:
    """ e.g. t.size() = (2,3,4). ignore_first True return 3*4=12 else 2*3*4=24"""
    total = 1
    my_shape = t.shape[1:] if ignore_first else t.shape
    for d in my_shape:
        total *= d
    return total


def model_params_count(model: nn.Module) -> int:
    total_parameters = 0
    for p in list(model.parameters()):
        total_parameters += tensor_total_size(p, False)
    return total_parameters


def model_params_print(model: nn.Module, print_values: bool = False, max_samples: int = 2):
    """
    :param model: nn model with self.title member
    :param print_values: print vars values as a list
    :param max_samples: if print_values: prints first 'max_samples' as a list
    :return:
    """
    print('{}:'.format(model.title))
    msg = '\t{:15s}: {:10s} ({:7} params), trainable:{}, is_cuda:{}'
    sum_params = 0
    for name, param in model.named_parameters():
        layer_params = 1
        for d in param.shape:
            layer_params *= d
        sum_params += layer_params
        print(msg.format(name, str(param.shape), layer_params, param.requires_grad, param.is_cuda))
        if print_values:
            print('\t\tvalues: {}'.format(param[min(max_samples, param.shape[0])].tolist()))
    print('\tTotal {:,.0f} params'.format(sum_params))
    return


class BaseModel(nn.Module):
    def __init__(self, path):
        super(BaseModel, self).__init__()
        self.path = os.path.abspath(path)
        self.title = os.path.basename(os.path.normpath(self.path)).replace('.pt', '')
        self.params = 0
        self.__mapping = [['enumerate from 1']]  # ignore index 0
        return

    def init(self, full_model_params: int = 1):
        self.params = model_params_count(self)
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


class LeNetMnist(BaseModel):
    """
    p__LeNetMnist_300_100(40 epochs Adam optimizer):
        train loss=0.021430, accuracy=59571/60000 (99.28%)
        test  loss=0.091702, accuracy=9793/10000 (97.93%)
    if fc1==300 and fc2==100: 266,610 params
    """

    def __init__(self, path, fc1=300, fc2=100):
        super().__init__(path)
        self.fc1 = nn.Linear(784, fc1)  # 28^2 to fc1. if fc1==300: 235,500 params
        self.fc2 = nn.Linear(fc1, fc2)  # fc1 to fc2. if fc1==300 and fc2==100: 30,100 params
        self.fc3 = nn.Linear(fc2, 10)  # fc2 to 10 classes if fc2==100: 1,010 params
        self.init(full_model_params=266610)
        return

    def create_mapping(self):
        # create layers mapping:
        mapping_list, layer_list = [], []
        for name, param in self.named_parameters():
            # print(name, param.shape)
            layer_list.append(name)
            if len(layer_list) == 2:  # in LeNet - each 2 are an fc layer
                mapping_list.append(layer_list)
                layer_list = []
        return mapping_list

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        h_relu1 = func.relu(self.fc1(x))
        h_relu2 = func.relu(self.fc2(h_relu1))
        logits = self.fc3(h_relu2)
        return func.log_softmax(logits, dim=1)


def get_data_set(ds_name: str, data_root: str, is_train: bool, transform=None, download: bool = False,
                 data_limit: int = 0, ack: bool = True) -> torchvision.datasets:
    """
    if transform is None - sets default transform by my choice
    if data_limit > 0: ds = ds[:data_limit]
    e.g.
        ds_name = 'mnist' # 'cifar10'
        ds_root = '../Datasets'
        is_train = True
        train_ds = get_data_set(ds_name=ds_name, data_root=ds_root, is_train=is_train, transform=None,
                                download = False, data_limit= 0, ack=True)
    """
    if ds_name == 'cifar10':
        if transform is None:  # default cifar10 transforms
            transform = DEFAULT_CIFAR10_TR_TRAIN if is_train else DEFAULT_CIFAR10_TR_TEST
        data_set = torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=download, transform=transform)
    elif ds_name == 'mnist':
        if transform is None:  # default mnist transform
            transform = DEFAULT_MNIST_TR
        data_set = torchvision.datasets.MNIST(root=data_root, train=is_train, download=download, transform=transform)
    else:
        print('data_set not valid!')
        return
    if data_limit > 0:
        data_set.data = data_set.data[:data_limit]
        data_set.targets = data_set.targets[:data_limit]

    if ack:
        prefix = 'train' if is_train else 'test'
        print('get_data_set({}: {} dataset): {}'.format(ds_name, prefix, data_set_size_to_str(data_set)))
    return data_set


def data_set_size_to_str(ds: torchvision.datasets) -> str:
    ds_len = len(ds)

    x = ds.data[0]  # load 1st sample as data loader will load
    X_size_post_tr = (ds_len,)
    for d in x.shape:
        X_size_post_tr += (d,)

    y_size = (len(ds.targets),)  # real data
    res = '|X|={}, |y|={}'.format(X_size_post_tr, y_size)
    return res


def get_data_loader(ds_name: str, data_root: str, is_train: bool, bs: int, transform=None,
                    shuffle: bool = False, data_limit: int = 0, download: bool = False) -> tu_data.DataLoader:
    """
    e.g.
        ds_name = 'mnist' # 'cifar10'
        ds_root = '../Datasets'
        is_train = True
        train_loader = GetDataScript.get_data_loader(ds_name=argv['ds_name'], data_root=argv['ds_root'], is_train=True,
                                                 bs=64, shuffle=False, transform=None, data_limit=0)
    """
    data_set = get_data_set(ds_name, data_root, is_train, transform, download, data_limit, ack=False)
    data_loader = tu_data.DataLoader(data_set, batch_size=bs, shuffle=shuffle, num_workers=2)
    prefix = 'train' if is_train else 'test'
    print('get_data_loader({}: {} dataset): {}'.format(ds_name, prefix, data_set_size_to_str(data_loader.dataset)))
    return data_loader


def get_dataloaders(ds_name: str, ds_root: str, bs_train: int, bs_test: int, shf_train: bool, shf_test: bool = False,
                    data_limit: int = 0) -> (DataLoader, DataLoader):
    assert ds_name in ['cifar10', 'mnist'], 'invalid ds name: {}'.format(ds_name)
    train_loader = get_data_loader(ds_name=ds_name, data_root=ds_root, is_train=True, bs=bs_train,
                                   shuffle=shf_train, transform=None, data_limit=data_limit, download=True)
    test_loader = get_data_loader(ds_name=ds_name, data_root=ds_root, is_train=False, bs=bs_test,
                                  shuffle=shf_test, transform=None, data_limit=data_limit, download=True)
    return train_loader, test_loader


def opt_to_str(optimizer: torch.optim) -> str:
    opt_s = str(optimizer)
    # t = repr(opt_s)
    opt_s = opt_s.replace('\n', '')
    opt_s = opt_s.replace('    ', ' ')
    return opt_s


def run_info(model, ds_train, ds_test, use_acc, epochs, train_bs, total_train_batches, test_bs, total_test_batches,
             optimizer, lr_cps):
    print('{}:'.format(model.title))
    print('\tTrain data: {}'.format(data_set_size_to_str(ds_train)))
    print('\tTest data: {}'.format(data_set_size_to_str(ds_test)))
    print('\tsaving on each {} improvment'.format('accuracy' if use_acc else 'loss'))
    msg = '\tepochs={}, bs_train={}, #trainBatches={}, bs_test={}, #testBatches={}'
    print(msg.format(epochs, train_bs, total_train_batches, test_bs, total_test_batches))
    print('\toptimizer = {}'.format(opt_to_str(optimizer)))
    if len(lr_cps) > 0:
        print('\tif epoch in {}: lr/=10'.format(lr_cps))
    return


def get_current_time() -> float:
    return time.time()


def get_time_str(start_time: float) -> str:
    hours, rem = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def get_lr(optimizer: torch.optim) -> float:
    lr = None
    if "lr" in optimizer.param_groups[0]:
        lr = optimizer.param_groups[0]['lr']
    return lr


def set_lr(optimizer: torch.optim, new_lr: float):
    optimizer.param_groups[0]['lr'] = new_lr
    return


def add_cuda(var: torch.Tensor) -> torch.Tensor:
    """ assigns the variables to GPU if available"""
    if torch.cuda.is_available() and not var.is_cuda:
        var = var.cuda()
    return var


def save_model(model: nn.Module, ack_print: bool = True, tabs: int = 0):
    """ nn model with self.title and self.path members """
    torch.save(model.state_dict(), model.path)
    if ack_print:
        print('{}{} saved to {}'.format(tabs * '\t', model.title, model.path))
    return


def load_model(model: nn.Module, ack_print: bool = True, tabs: int = 0):
    """ nn model with self.title and self.path members """
    dst_allocation = None if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model.path, map_location=dst_allocation))
    model.eval()
    if ack_print:
        print('{}{} loaded from {}'.format(tabs * '\t', model.title, model.path))
    return


def set_cuda_scope_and_seed(seed: int, dtype='FloatTensor'):
    """
    https://pytorch.org/docs/stable/tensors.html
    32-bit floating point: torch.cuda.FloatTensor
    64-bit floating point: torch.cuda.DoubleTensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        def_dtype = 'torch.cuda.' + dtype
        torch.set_default_tensor_type(def_dtype)
        torch.cuda.manual_seed(seed)
        print('working on CUDA. default dtype = {} <=> {}'.format(def_dtype, torch.get_default_dtype()))
    else:
        print('working on CPU')
    return


def train_using_dls(train_dl: DataLoader, test_dl: DataLoader, model: nn.Module, opt: torch.optim, f: Callable,
                    epochs: int, use_acc: bool, save_models: bool, b_interval: int = -1, lr_cps: list = None) -> None:
    """
    training using dataloaders - no input weight supported.
    :param train_dl: DataLoader
    :param test_dl: DataLoader
    :param model: nn model
    :param opt:
    :param f: loss function. f(pred, y, reduction='mean' or 'sum')
    :param epochs:
    :param use_acc: save model metric (loss or accuracy)
    :param save_models: if False, you will get last epoch model (not necessarily the best one)
    :param b_interval: batch interval. if batch_ind % b_interval == 0: prints batch stats. -1 is no batch stats
    :param lr_cps: list of floats. e.g. lr_cps=[0.5, 0.75]. in epoch=ceil(epochs*0.5) and ceil(epochs*0.75): lr/=10
    :return: None - train model by reference
    you should load model after training
    """
    train_time = get_current_time()

    train_bs = train_dl.batch_size
    train_size = len(train_dl.dataset)
    train_batches = len(train_dl)
    test_bs = test_dl.batch_size
    test_batches = len(test_dl)
    lr_cps = [] if lr_cps is None else [math.ceil(epochs * cp) for cp in lr_cps]

    run_info(model, train_dl.dataset, test_dl.dataset, use_acc, epochs, train_bs, train_batches, test_bs, test_batches,
             opt, lr_cps)

    # saves the stats by loss
    by_loss_lowest_loss = float("inf")
    by_loss_acc = -1
    by_loss_best_epoch = -1

    # saves the stats by acc
    by_acc_highest_acc = -1
    by_acc_loss = float("inf")
    by_acc_best_epoch = -1

    print('Training:')
    for epoch in range(1, epochs + 1):
        model.train()  # test sets model.eval()
        if b_interval > 0:
            print('Epoch {}/{}:'.format(epoch, epochs))
        if epoch in lr_cps:
            set_lr(opt, get_lr(opt) / 10)
            print('\toptimizer changed = {}'.format(opt_to_str(opt)))
        sum_batches_loss = 0.0
        for b_i, (batchX, batch_y) in enumerate(train_dl):
            batchX, batch_y = add_cuda(batchX), add_cuda(batch_y)
            batch_y_pred_vectors = model(batchX)
            loss = f(batch_y_pred_vectors, batch_y, reduction='mean')
            # ============ Backward ============
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_batches_loss += loss.item()
            if b_interval > 0 and b_i % b_interval == 0:
                msg = '\tBatch {:3}/{:3} ({:5}/{:5}) - Loss: {:.6f}'
                print(msg.format(b_i, train_batches, b_i * train_bs, train_size, loss.item()))

        # for each batch we summed its mean. divide by train_batches
        train_loss = sum_batches_loss / train_batches
        test_loss, test_acc, test_res_str = test_using_dl(test_dl, model, f)

        imp_acc, imp_loss = False, False
        if test_acc > by_acc_highest_acc:  # found new highest acc so far
            by_acc_highest_acc = test_acc
            by_acc_loss = test_loss
            by_acc_best_epoch = epoch
            imp_acc = True
        if test_loss < by_loss_lowest_loss:  # found new lowest loss so far
            by_loss_lowest_loss = test_loss
            by_loss_acc = test_acc
            by_loss_best_epoch = epoch
            imp_loss = True

        # save by use_acc value (or in first epoch)
        msg = '\tEpoch [{:3}/{:3}] train loss:{:.6f},test {}, time so far {}'
        if (use_acc and imp_acc) or (not use_acc and imp_loss) or (epoch == 1):
            msg += ' New best found'
            if save_models:
                save_model(model, ack_print=False)
        print(msg.format(epoch, epochs, train_loss, test_res_str, get_time_str(train_time)))

    end_msg1 = 'Done training by {}: On epoch {}: loss={:.6f}, accuracy = {}%'
    end_msg2 = 'If trained by {}   : On epoch {}: loss={:.6f}, accuracy = {}%'
    if use_acc:
        print(end_msg1.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
        print(end_msg2.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
    else:
        print(end_msg1.format('loss    ', by_loss_best_epoch, by_loss_lowest_loss, by_loss_acc))
        print(end_msg2.format('accuracy', by_acc_best_epoch, by_acc_loss, by_acc_highest_acc))
    return


def test_using_dl(dl: DataLoader, model: nn.Module, f: Callable, print_res: bool = False) -> (float, float, str):
    """
    test using dataloaders - no input weight supported.
    :param dl: DataLoader
    :param model: nn model
    :param f: loss function. f(pred, y, reduction='mean' or 'sum')
    :param print_res:
    """
    model.eval()
    total_images = len(dl.dataset)

    with torch.no_grad():
        sum_losses, correct = 0, 0
        for batchX, batch_y in dl:
            batchX, batch_y = add_cuda(batchX), add_cuda(batch_y)
            batch_y_pred_vectors = model(batchX)

            sum_losses += f(batch_y_pred_vectors, batch_y, reduction='sum').item()  # sum up batch loss
            predictions = batch_y_pred_vectors.argmax(dim=1, keepdim=True)
            correct += predictions.eq(batch_y.view_as(predictions)).sum().item()

    avg_loss = sum_losses / total_images
    acc = 100. * correct / total_images
    result_str = 'loss={:.6f}, accuracy={}/{} ({:.2f}%)'.format(avg_loss, correct, total_images, acc)
    if print_res:
        print(result_str)
    return avg_loss, acc, result_str


def sim_forward():
    # simulate forward
    model = LeNetMnist(path='./p_LeNetMnist_300_100.pt', fc1=300, fc2=100)
    batch_x = torch_normal(shape=(1000, 1, 28, 28), miu=0, std=10)
    y_pred = model(batch_x)
    print(var_to_string(batch_x, 'batch_x', with_data=False))  # var_to_string - useful function
    print(var_to_string(y_pred, 'y_pred', with_data=False))
    summary(model, input_size=(1, 28, 28), batch_size=-1, device='cpu')  # print model info
    layer_2_components_names_list = model.get_layers_components([2])  # Example getting layer number 2
    print(layer_2_components_names_list)
    all_names_dict = model.get_mapping_dict()  # example to get all
    print(all_names_dict)

    model_params_print(model, print_values=False)  # example to print model param with all info
    return


def mnist_lenet_example():
    model = LeNetMnist(path='./p_LeNetMnist_300_100_test2.pt', fc1=300, fc2=100)
    model_params_print(model, print_values=False)  # example to print model param with all info
    train_dl, test_dl = get_dataloaders(ds_name='mnist',
                                        ds_root='../../Datasets',
                                        bs_train=64,
                                        bs_test=1000,
                                        shf_train=False,
                                        shf_test=False,
                                        data_limit=0)
    if not os.path.exists(model.path):
        print('{} doesn\'t exists...'.format(model.title))
        # opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        train_using_dls(train_dl=train_dl,
                        test_dl=test_dl,
                        model=model,
                        opt=opt,
                        f=func.cross_entropy,
                        epochs=40,
                        use_acc=True,  # save according to acc/loss improvment
                        save_models=True,
                        b_interval=-1,  # if you need batch prints (not just post epoch)
                        lr_cps=[0.5, 0.75])

    load_model(model, ack_print=True)
    loss_p, acc_p, res_str = test_using_dl(dl=test_dl, model=model, f=func.cross_entropy)
    print('res: {}'.format(res_str))
    return


def make_cuda_invisible():
    """ disable cuda """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1, 0'
    return


if __name__ == '__main__':
    make_cuda_invisible()
    set_cuda_scope_and_seed(seed=42)
    # sim_forward()
    mnist_lenet_example()
