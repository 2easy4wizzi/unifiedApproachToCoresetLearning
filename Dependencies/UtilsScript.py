import torch
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import os
import torch.nn as nn
from sklearn.cluster import KMeans
# from scipy.spatial import distance
import torchvision
import random
import inspect
import typing
import wizzi_utils as wu


def get_pc_name() -> str:
    import platform
    return platform.uname()[1]


# TORCH UTILS
def cuda_on():
    """ check if cuda available """
    return torch.cuda.is_available()


def make_cuda_invisible():
    """ disable cuda """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1, 0'
    return


def set_cuda_scope_and_seed(seed: int, dtype='FloatTensor'):
    """
    https://pytorch.org/docs/stable/tensors.html
    32-bit floating point: torch.cuda.FloatTensor
    64-bit floating point: torch.cuda.DoubleTensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_on():
        def_dtype = 'torch.cuda.' + dtype
        torch.set_default_tensor_type(def_dtype)
        torch.cuda.manual_seed(seed)
        print('working on CUDA. default dtype = {} <=> {}'.format(def_dtype, torch.get_default_dtype()))
    else:
        print('working on CPU')
    return


def add_cuda(var: torch.Tensor) -> torch.Tensor:
    """ assigns the variables to GPU if available"""
    if cuda_on() and not is_cuda(var):
        var = var.cuda()
    return var


def is_trainable(var: torch.Tensor) -> bool:
    return var.requires_grad


def is_cuda(var: torch.Tensor) -> bool:
    return var.is_cuda


def t_size_str(var: torch.Tensor) -> str:
    size_str = str(var.size())
    size_str = size_str[size_str.find("(") + 1:size_str.find(")")]
    return size_str


def data_set_size_to_str(ds: torchvision.datasets) -> str:
    ds_len = len(ds)

    x = ds.data[0]  # load 1st sample as data loader will load
    X_size_post_tr = (ds_len,)
    for d in x.shape:
        X_size_post_tr += (d,)

    y_size = (len(ds.targets),)  # real data
    res = '|X|={}, |y|={}'.format(X_size_post_tr, y_size)
    return res


def var_to_string(var: typing.Union[torch.Tensor, np.array], title: str = 'XXX', with_data: bool = False) -> str:
    """  to string of a torch tensor or numpy """
    if isinstance(var, torch.Tensor):
        msg = '{:8s}: {:8s}, dtype:{}, trainable:{}, is_cuda:{}'
        msg = msg.format(title, t_size_str(var), var.dtype, is_trainable(var), is_cuda(var))
    else:  # assuming numpy
        msg = '{:8s}: {:8s}, dtype:{}'.format(title, str(var.shape), var.dtype)
    if with_data:
        msg += ', data: {}'.format(np.round(var.tolist(), 2).tolist())
        # msg += ', data: {}'.format(var.tolist())
    return msg


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
        print(msg.format(name, t_size_str(param), layer_params, is_trainable(param), is_cuda(param)))
        if print_values:
            print('\t\tvalues: {}'.format(param[min(max_samples, param.shape[0])].tolist()))
    print('\tTotal {:,.0f} params'.format(sum_params))
    return


def model_summary_to_string(model: nn.Module, input_size: tuple, batch_size: int) -> str:
    """
        get model info to string
        e.g.
            m = MnistModel()
            print(utils.model_summary_to_string(m, (1, 28, 28), 64))
    """
    from torchsummary import summary
    a, b = redirect_std_start()
    summary(model, input_size, batch_size)
    return redirect_std_finish(a, b)


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


def save_tensor(t, path: str, ack_print: bool = True, tabs: int = 0):
    """ t: dict or a tensor """
    torch.save(t, path)
    if ack_print:
        print('{}Saved to: {}'.format(tabs * '\t', path))
    return


def load_tensor(path: str, ack_print: bool = True, tabs: int = 0):
    """ t: dict or a tensor """
    dst_allocation = None if cuda_on() else 'cpu'
    t = torch.load(path, map_location=dst_allocation)
    if ack_print:
        print('{}Loaded: {}'.format(tabs * '\t', path))
    return t


def save_np(t: np.array, path: str, ack_print: bool = True, tabs: int = 0):
    np.save(path, t)
    if ack_print:
        if not path.endswith('.npy'):
            path += '.npy'
        print('{}Saved to: {}'.format(tabs * '\t', path))
    return


def load_np(path: str, ack_print: bool = True, tabs: int = 0) -> np.array:
    t = np.load(path)
    if ack_print:
        print('{}Loaded: {}'.format(tabs * '\t', path))
    return t


def save_model(model: nn.Module, ack_print: bool = True, tabs: int = 0):
    """ nn model with self.title and self.path members """
    torch.save(model.state_dict(), model.path)
    if ack_print:
        print('{}{} saved to {}'.format(tabs * '\t', model.title, model.path))
    return


def load_model(model: nn.Module, ack_print: bool = True, tabs: int = 0):
    """ nn model with self.title and self.path members """
    dst_allocation = None if cuda_on() else 'cpu'
    model.load_state_dict(torch.load(model.path, map_location=dst_allocation))
    model.eval()
    if ack_print:
        print('{}{} loaded from {}'.format(tabs * '\t', model.title, model.path))
    return


def set_model_status(model: nn.Module, status: bool, status_print: bool = False):
    """ set model parameters trainable status to 'status' """
    for param in model.parameters():
        param.requires_grad_(status)
    if status_print:
        print(model_status_str(model))
    return


def model_status_str(model: nn.Module) -> str:
    """
    3 options: model fully trainable, fully frozen, both
    model has self.title
    """
    saw_trainable, saw_frozen = 0, 0
    for param in model.parameters():
        if is_trainable(param):
            saw_trainable = 1
        else:
            saw_frozen = 1
    ans = saw_trainable + saw_frozen
    if ans == 2:
        msg = '{} is part trainable and part frozen'.format(model.title)
    elif saw_trainable == 1:
        msg = '{} is fully trainable'.format(model.title)
    else:
        msg = '{} is fully frozen'.format(model.title)
    return msg


def copy_models(model_source: nn.Module, model_target: nn.Module, dont_copy_layers: list, ack: bool = False):
    """
    :param model_source: copy from
    :param model_target: copy to
    :param dont_copy_layers: list of exact names as appear in model.named_parameters()[0]
           make sure before coping that dst and target tensors are in the same size
    :param ack: prints copy\not
    :return:
    """
    print('Copying model {} to model {} except {}:'.format(model_source.title, model_target.title, dont_copy_layers))
    for name, param in model_source.named_parameters():
        if name in dont_copy_layers:
            if ack:
                print('\tNOT copied Layer {}'.format(name))
        else:
            if ack:
                print('\tCopied Layer {}'.format(name))
            model_target.state_dict()[name].copy_(param)
            # print('\t\tvalues orig   : {}'.format(param.tolist()))
            # print('\t\tvalues coreset: {}'.format(model_coreset.state_dict()[name].tolist()))
            # print('\t\tvalues coreset: {}'.format(model_coreset.state_dict()[name].tolist()))
    return


def freeze_layers(model: nn.Module, trainable_layers: list):
    """
    trainable_layers: list of exact names as appear in model.named_parameters()[0]
    all params that are in trainable_layers -> trainable = True
    else -> trainable = False
    """
    print('Model {}: freezing all except {}:'.format(model.title, trainable_layers))
    for name, param in model.named_parameters():
        if name in trainable_layers:
            param.requires_grad_(True)
            # print('alive {}'.format(name))
        else:
            param.requires_grad_(False)
            # print('frozen {}'.format(name))
    return


def clone_model(model: nn.Module):
    import copy
    model_clone = copy.deepcopy(model)
    return model_clone


def torch_uniform(shape: tuple, range_low: float, range_high: float) -> torch.Tensor:
    ret = torch.empty(shape).uniform_(range_low, range_high)
    return ret


def torch_normal(shape: tuple, miu: float, std: float, to_double: bool = False) -> torch.Tensor:
    ret = torch.empty(shape).normal_(miu, std)
    if to_double:
        ret = ret.double()
    return ret


def get_uniform_dist_by_dim(A):
    lows = np.min(A, axis=0)
    highs = np.max(A, axis=0)
    return lows, highs


def get_normal_dist_by_dim(A):
    means = np.mean(A, axis=0)
    stds = np.std(A, axis=0)
    return means, stds


def np_normal(shape: tuple, mius: list, stds: list) -> np.array:
    """
    e.g.
        d = 2
        A = np.zeros((3, d))
        A[0][0], A[0][1] = 0, 1000
        A[1][0], A[1][1] = 1, 2000
        A[2][0], A[2][1] = 5, 4000
        means, stds = get_normal_dist_by_dim(A)
        print(means, stds)
        A_2 = np_normal(shape=(500, d), mius=means, stds=stds)
        print(A_2.shape)
        print(get_normal_dist_by_dim(A_2))
    """
    ret = np.random.normal(loc=mius, scale=stds, size=shape)
    return ret


def np_uniform(shape: tuple, lows: list, highs: list) -> np.array:
    """
    e.g.
        d=2
        A = np.zeros((3, d))
        A[0][0], A[0][1] = 0, 1000
        A[1][0], A[1][1] = 1, 2000
        A[2][0], A[2][1] = 5, 4000
        lows, highs = get_uniform_dist_by_dim(A)
        print(lows, highs)
        A_2 = np_uniform(shape=(10, d), lows=lows, highs=highs)
        print(A_2.shape)
        print(get_normal_dist_by_dim(A_2))
    """
    ret = np.random.uniform(low=lows, high=highs, size=shape)
    return ret


def numpy_to_torch(var_np: np.array, to_double=True, detach: bool = False) -> torch.Tensor:
    """ float is hard coded. if you need double, change the code"""
    if detach:
        if to_double:
            var_torch = add_cuda(torch.from_numpy(var_np).double()).detach()
        else:
            var_torch = add_cuda(torch.from_numpy(var_np).float()).detach()
    else:
        if to_double:
            var_torch = add_cuda(torch.from_numpy(var_np).double())
        else:
            var_torch = add_cuda(torch.from_numpy(var_np).float())
    return var_torch


def torch_to_numpy(var_torch: torch.Tensor) -> np.array:
    if is_trainable(var_torch):
        var_np = var_torch.detach().cpu().numpy()
    else:
        var_np = var_torch.cpu().numpy()
    return var_np


def opt_to_str(optimizer: torch.optim) -> str:
    opt_s = str(optimizer)
    # t = repr(opt_s)
    opt_s = opt_s.replace('\n', '')
    opt_s = opt_s.replace('    ', ' ')
    return opt_s


def get_momentum(optimizer: torch.optim) -> float:
    mom = None
    if "momentum" in optimizer.param_groups[0]:
        mom = optimizer.param_groups[0]['momentum']
    return mom


def get_weight_decay(optimizer: torch.optim) -> float:
    wd = None
    if "weight_decay" in optimizer.param_groups[0]:
        wd = optimizer.param_groups[0]['weight_decay']
    return wd


def get_lr(optimizer: torch.optim) -> float:
    lr = None
    if "lr" in optimizer.param_groups[0]:
        lr = optimizer.param_groups[0]['lr']
    return lr


def set_lr(optimizer: torch.optim, new_lr: float):
    optimizer.param_groups[0]['lr'] = new_lr
    return


def get_opt_by_name(opt_d: dict, model: nn.Module) -> torch.optim:
    """
    :param model: the target model for the optimizer
    :param opt_d: has keys:
        'name': 'ADAM', 'SGD'
        all has:
            'lr': float
            'weight_decay': float # 0 for none
        if 'ADAM' has also:
            pass
        if 'SGD'  has also:
            'momentum': float
    :return:
    """
    opt = None
    if opt_d['name'] == 'ADAM':
        opt = torch.optim.Adam(model.parameters(), lr=opt_d['lr'], weight_decay=opt_d['weight_decay'])
    elif opt_d['name'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=opt_d['lr'], momentum=opt_d['momentum'],
                              weight_decay=opt_d['weight_decay'])
    return opt


class OptimizerHandler:
    """
    example:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = OptimizerHandler(optimizer, factor=0.5, patience=15, min_lr=0.0005)

    for epoch in range(1, n_epochs + 1):
        for x,y in batches:
            y_tag = model(y)
            loss = loss_func(y, y_tag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        optimizer.update_lr()  # <- this counts the epochs and tries to change the lr
        # update_lr add +1 to counter. if counter >= patience: counter = 0 and new_lr= max(old_lr * factor, min_lr)

    """

    def __init__(self, optimizer: torch.optim, factor: float, patience: int, min_lr: float):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.epochs_passed = 0

    def step(self):
        self.optimizer.step()
        return

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, new_lr: float):
        self.optimizer.param_groups[0]['lr'] = new_lr
        return

    def zero_grad(self):
        self.optimizer.zero_grad()
        return

    def update_lr(self):
        self.epochs_passed += 1
        if self.epochs_passed >= self.patience:
            self.epochs_passed = 0
            old_lr = self.lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.set_lr(new_lr)
            # print('new lr changed to {}'.format(self.lr()))
        return


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best = None
        return

    def should_early_stop(self, loss: float):
        should_stop = False
        if self.best is None:
            self.best = loss
        elif loss < self.best:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
            # print('\t\tpatience {}/{}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                should_stop = True
        return should_stop


def random_init(c_size: int, A: torch.Tensor, trainable: bool = True, miu: float = 0, std: float = 1) -> torch.Tensor:
    # print('Random_init {} from A (|A|={})'.format(c_size, pretty_torch_size(A)))
    coreset_size = (c_size,)
    for d in A[0].shape:
        coreset_size += (d,)
    coreset_x = add_cuda(torch_normal(coreset_size, miu, std))
    coreset_x.requires_grad_(trainable)
    return coreset_x


def subset_init(c_size: int, A: torch.Tensor, trainable: bool = True, to_double: bool = False) -> torch.Tensor:
    """ TODO if c_size>|A| - not supported.
        change to np.choice to support
        a coreset that is bigger than |A|
    """
    # print('Subset_init {} from A (|A|={})'.format(c_size, pretty_torch_size(A)))
    n = A.shape[0]
    perm = np.random.permutation(n)
    idx = perm[:c_size]
    if not to_double:
        coreset_x = A[idx].clone().float()
    else:
        coreset_x = A[idx].clone().double()
    coreset_x.requires_grad_(trainable)
    return coreset_x


def augment_x_y_numpy(X: np.array, y: np.array) -> np.array:
    """ creates A=X|y """
    assert X.shape[0] == y.shape[0], 'row count must be the same'
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.reshape(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.reshape(y.shape[0], 1)
    A = np.column_stack((X, y))
    return A


def de_augment_numpy(A: np.array) -> (np.array, np.array):
    """ creates X|y=A """
    if len(A.shape) == 1:  # A is 1 point. change from size (n) to size (1,n)
        A = A.reshape(1, A.shape[0])
    X, y = A[:, :-1], A[:, -1]
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.reshape(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.reshape(y.shape[0], 1)
    return X, y


def augment_x_y_torch(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ creates A=X|y """
    assert X.shape[0] == y.shape[0], 'row count must be the same'
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.view(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.view(y.shape[0], 1)
    A = torch.cat((X, y), 1)
    return A


def de_augment_torch(A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ creates X|y=A """
    if len(A.shape) == 1:  # A is 1 point. change from size (n) to size (1,n)
        A = A.view(1, A.shape[0])
    X, y = A[:, :-1], A[:, -1]
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.view(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.view(y.shape[0], 1)
    return X, y


def split_tensor(Q: torch.Tensor, p: float = 0.9) -> (torch.Tensor, torch.Tensor):
    partition = int(p * Q.shape[0])
    Q_1 = Q[:partition]
    Q_2 = Q[partition:]
    return Q_1, Q_2


# MISC
# def create_unit_circle_points(point: np.array = np.array([0, 0], dtype=float), density: int = 4,
#                               plot: bool = False) -> np.array:
#     """
#     currently supports 2d only:
#     density: |A|=2^density,2,1
#     point: 2d point X|y
#     e.g.
#     create_unit_circle_points(x=0, y=0, density=4)
#     # >>> get 16 points around origin
#     """
#     t = np.linspace(0, np.pi * 2, 2 ** density + 1)  # +1:= first and last are the same
#     X = np.cos(t) + point[0]
#     y = np.sin(t) + point[1]
#     X, y = X[:-1], y[:-1]
#
#     A = augment_x_y_numpy(X, y)
#
#     print('t', t.shape, t.tolist())
#     print('x', X.shape, X.tolist())
#     print('y', y.shape, y.tolist())
#
#     print(A.shape)
#     # if plot:
#     #     g_A = (A, 'g', 'A')
#     #     g_point = (point.reshape(1, len(point), 1), 'black', 'point {}'.format(point))
#     #     plot_2d_scatter(groups=[g_A, g_point])
#     return A


def perms(n: int, perm_size):
    """
    e.g. A is 3 2d points (shape=(3,2))
    perms(A.shape[0], 2)
    # >>> [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    perms(A.shape[0], A.shape[0])
    # >>> [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    """
    from itertools import permutations
    range_list = np.arange(0, n, 1)
    dif_perms = list(permutations(range_list, perm_size))
    dif_perms = [list(dif_perm) for dif_perm in dif_perms]
    return dif_perms


def nCk(n: int, k: int, as_int: bool = False):
    """
    :param n:
    :param k:
    :param as_int:
    :return: if as_int True: the result of nCk, else the combinations of nCk

    e.g.
    A = np.random.randint(low=-10, high=10, size=(3, 2))
    print('A={}'.format(A.tolist()))

    # let's iterate on every 2 different indices of A
    combs_count = utils.nCk(len(A), k=2, as_int=True)
    print('|A| choose 2={}:'.format(combs_count)) # result is 3

    combs_list = utils.nCk(len(A), k=2)  # result is [[0, 1], [0, 2], [1, 2]]
    for i, comb in enumerate(combs_list):
        print('\tcomb {}={}. A[comb]={}'.format(i, comb, A[comb].tolist()))
    """
    from itertools import combinations
    range_list = np.arange(0, n, 1)
    combs = list(combinations(range_list, k))
    combs = [list(comb) for comb in combs]
    if as_int:
        combs = len(combs)  #
    return combs


def shuffle_tensor(arr: torch.Tensor) -> torch.Tensor:
    """ shuffles an array """
    if isinstance(arr, torch.Tensor):
        arr = arr[torch.randperm(arr.shape[0])]
    return arr


def shuffle_ds(ds: torchvision.datasets):
    """
    by ref
    :param ds:
    :return:
    """
    n = ds.targets.shape[0]
    perm = torch.randperm(n)
    ds.targets = ds.targets[perm]
    ds.data = ds.data[perm]
    return


def split_A(A: torch.Tensor, test_p: float) -> (torch.Tensor, torch.Tensor):
    """
    90:10 split:
    split_A(A, test_p=0.1)
    """
    test_size = int(A.shape[0] * test_p)
    A_train, A_test = A[:-test_size], A[-test_size:]
    print(var_to_string(A_train, 'A_train'))
    print(var_to_string(A_test, 'A_test'))
    return A_train, A_test


def filter_classes(X, y, keep_classes_list, count_classes=False):
    """
    :param X: np\torch nxd
    :param y: np\torch labels nx1
    :param keep_classes_list: classes list to keep
    :param count_classes: should print info of label
    :return:
    """
    if count_classes:
        count_keys(y)
    a = torch.zeros(y.shape[0], dtype=torch.uint8)

    for class_num in keep_classes_list:
        a |= add_cuda(y == class_num)
    X_tag, y_tag = X[a], y[a]
    if count_classes:
        count_keys(y_tag)
    return X_tag, y_tag


def count_keys(y):
    """
    @:param y: nx1 array (torch, list, numpy)
    e.g.
        from torchvision import datasets
        # data_root = path to the data else download
        dataset = datasets.MNIST(root=data_root, train=False, download=False)
        count_classes(dataset.targets)
    """
    from collections import Counter
    y_shape = len(y) if isinstance(y, list) else y.shape
    print('Count classes: (y shape {})'.format(y_shape))
    cnt = Counter()
    for value in y:
        ind = value.item() if isinstance(y, torch.Tensor) else value
        cnt[ind] += 1
    cnt = sorted(cnt.items())
    for item in cnt:
        print('\tClass {}: {} samples'.format(item[0], item[1]))
    return


def redirect_std_start():
    import io
    """ redirect all prints to summary_str """
    import sys
    old_stdout = sys.stdout
    sys.stdout = summary_str = io.StringIO()
    return old_stdout, summary_str


def redirect_std_finish(old_stdout, summary_str) -> str:
    """ redirect all prints bach to std out and return a string of what was captured"""
    import sys
    sys.stdout = old_stdout
    return summary_str.getvalue()


def get_current_date_hour() -> str:
    now = datetime.now()
    current_time = now.strftime('%d-%m-%Y %H:%M:%S')
    return current_time


def get_current_time() -> float:
    return time.time()


def get_mili_seconds_str(start_time: float) -> str:
    msg = "Process time: {:,.3f} seconds".format((time.time() - start_time))
    return msg


def get_time_str(start_time: float) -> str:
    hours, rem = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def start_profiler():
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    return pr


def end_profiler(pr, rows=10) -> str:
    import pstats
    import io
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(rows)
    return s.getvalue()


def find_centers(A: np.array, k: int = 1) -> np.array:
    """
    :param A: nxd data array
    :param k: how many centers
    :return: centers kxd
    np.random.seed(42)
    A = np.zeros((4, 2))  # A square with origin 0
    A[0] = [-1, -1]
    A[1] = [-1, 1]
    A[2] = [1, -1]
    A[3] = [1, 1]
    print('A:{}'.format(A.tolist()))
    centers = find_centers(A, k=1)
    print('centers: type={}, shape={}, centers values:{}'.format(type(centers), centers.shape, centers.tolist()))
    X, y = de_augment_numpy(A)
    X_c, y_c = de_augment_numpy(centers)
    plt.scatter(X, y, color='g', marker='.', label='A')
    plt.scatter(X_c, y_c, color='r', marker='.', label='k==1')
    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5, edgecolor='black')
    plt.show()
    centers = find_centers(A, k=4)
    print('centers: type={}, shape={}, centers values:{}'.format(type(centers), centers.shape, centers.tolist()))

    A = np.arange(6).reshape(6, 1)
    print('A:{}'.format(A.tolist()))
    centers = find_centers(A, k=1)
    print('centers: type={}, shape={}, centers values:{}'.format(type(centers), centers.shape, centers.tolist()))
    centers = find_centers(A, k=3)
    print('centers: type={}, shape={}, centers values:{}'.format(type(centers), centers.shape, centers.tolist()))

    A = np.ones((3, 3))  # A 3 times (1,1,1)
    print('A:{}'.format(A.tolist()))
    centers = find_centers(A, k=1)
    print('centers: type={}, shape={}, centers values:{}'.format(type(centers), centers.shape, centers.tolist()))
    """
    if isinstance(A, torch.Tensor):
        A = torch_to_numpy(A)
    k_means_obj = KMeans(n_clusters=k)
    k_means_obj.fit(A)
    centers = k_means_obj.cluster_centers_
    return centers


def get_diff_2_points(a0: np.array, a1: np.array, normed=False) -> np.array:
    v = a1 - a0
    # print('v:{}'.format(v))
    if normed:
        u = v / np.linalg.norm(v)
        # print('u:{}'.format(u))
        return u
    return v


def mid_2_points(a0: np.array, a1: np.array) -> np.array:
    mid = (a1 - a0) / 2 + a0
    return mid


def get_line_number(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.lineno)
    except IndexError:
        pass
    return ret_val


def get_function_name(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.function)
    except IndexError:
        pass
    return ret_val


def get_file_name(depth: int = 1) -> str:
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.filename)
    except IndexError:
        pass
    return ret_val


def get_function_name_and_line(depth: int = 1) -> str:
    # for scope in inspect.stack():  # see all scopes
    #     print(scope)
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{} line {}'.format(scope_1_back.function, scope_1_back.lineno)
    except IndexError:
        pass
    return ret_val


CONST_COLOR_MAP = {
    'ResetAll': "\033[0m",
    'Bold': "\033[1m",
    'Dim': "\033[2m",
    'Underlined ': "\033[4m",
    'Blink      ': "\033[5m",
    'Reverse    ': "\033[7m",
    'Hidden     ': "\033[8m",

    'ResetBold': "\033[21m",
    'ResetDim': "\033[22m",
    'ResetUnderlined': "\033[24m",
    'ResetBlink': "\033[25m",
    'ResetReverse': "\033[27m",
    'ResetHidden': "\033[28m",

    'Default': "\033[39m",
    'Black': "\033[97m",
    'Red': "\033[31m",
    'Green': "\033[32m",
    'Yellow': "\033[33m",
    'Blue': "\033[34m",
    'Magenta': "\033[35m",
    'Cyan': "\033[36m",
    'LightGray': "\033[37m",
    'DarkGray': "\033[90m",
    'LightRed': "\033[91m",
    'LightGreen': "\033[92m",
    'LightYellow': "\033[93m",
    'LightBlue': "\033[94m",
    'LightMagenta': "\033[95m",
    'LightCyan': "\033[96m",
    'White': "\033[30m",

    'BackgroundDefault': "\033[49m",
    'BackgroundBlack': "\033[107m",
    'BackgroundRed': "\033[41m",
    'BackgroundGreen': "\033[42m",
    'BackgroundYellow': "\033[43m",
    'BackgroundBlue': "\033[44m",
    'BackgroundMagenta': "\033[45m",
    'BackgroundCyan': "\033[46m",
    'BackgroundLightGray': "\033[47m",
    'BackgroundDarkGray': "\033[100m",
    'BackgroundLightRed': "\033[101m",
    'BackgroundLightGreen': "\033[102m",
    'BackgroundLightYellow': "\033[103m",
    'BackgroundLightBlue': "\033[104m",
    'BackgroundLightMagenta': "\033[105m",
    'BackgroundLightCyan': "\033[106m",
    'BackgroundWhite': "\033[40m"
}


def add_color(string: str, color: str = 'Red') -> str:
    """
    :param string:
    :param color: from color map
    # todo add option for combination (like bold, underlined ...)
    # todo change to lower case
    # todo color numbers maybe wrong
    e.g.
        my_str = 'hello colorful world!'
        print(add_color(my_str))
        print(add_color(my_str, color='LightCyan'))
        print(add_color(my_str, color='fake_color'))
        print(add_color(my_str, color='Blue'))
        print(add_color(my_str, color='Bold'))
        print(my_str)
    to see all colors:
        color_map = UtilsScript.CONST_COLOR_MAP
        for k, v in color_map.items():
            print('{}{}{}'.format(v, k, color_map['ResetAll']))
    """

    if color in CONST_COLOR_MAP:
        # concat (color, string, reset tag)
        string = '{}{}{}'.format(CONST_COLOR_MAP[color], string, CONST_COLOR_MAP['ResetAll'])
    return string


# PLOTS

def get_x_ticks_list(x_low, x_high, p=10):
    ten_percent_jump = (x_high - x_low) / p
    x_ticks = [x_low + i * ten_percent_jump for i in range(p + 1)]
    return x_ticks


def plot_2d_subplots_start(rows: int = 1, cols: int = 1, main_title: str = None):
    plt.close('all')

    fig, ax_item_or_tuple_or_arr = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False)
    if main_title is not None:
        fig.suptitle(main_title)
    return fig, ax_item_or_tuple_or_arr


def plot_2d_add_subplot(ax: Axes, points_groups: list = None, lines_groups: list = None, sub_title: str = None,
                        label_x: str = 'x', label_y: str = 'y', add_center: bool = False):
    """
    np.random.seed(42)
    A = np.random.randint(low=-10, high=10, size=(10, 2))
    C = np.random.randint(low=-10, high=10, size=(5, 2))
    U = np.random.randint(low=-10, high=10, size=(5, 2))
    V = np.random.randint(low=-10, high=10, size=(5, 2))
    Q = np.random.randint(low=-10, high=10, size=(5, 2, 1))
    q_optA = np.random.randint(low=-10, high=10, size=(2, 1))
    q_optC = np.random.randint(low=-10, high=10, size=(2, 1))

    g_A = (A, 'g', 'A')
    g_C = (C, 'r', 'C')
    g_U = (U, 'b', 'U')
    g_V = (V, 'aqua', 'V')
    g_q_opt_A = (q_optA, 'g', 'A q_opt')
    g_q_opt_C = (q_optC, 'r', 'C q_opt')
    g_Q = (Q, 'darkviolet', 'trainQ')

    fig, ax_tuple = plot_2d_subplots_start(cols=2, main_title='1x2: linear regression')  # 1 row, 2 cols
    plot_2d_add_subplot(ax=ax_tuple[0], points_groups=[g_A, g_C], lines_groups=[g_q_opt_A, g_q_opt_C], sub_title='1',
                        add_center=True)
    plot_2d_add_subplot(ax=ax_tuple[1], points_groups=[g_A, g_C, g_U], lines_groups=None, sub_title='2',
                        add_center=True)
    plot_2d_subplots_end(f=fig, zoomed=True)

    fig, ax_tuple = plot_2d_subplots_start(rows=2, main_title='2x1')  # 2 rows, 1 col
    plot_2d_add_subplot(ax=ax_tuple[0], points_groups=[g_A, g_C], lines_groups=None, sub_title='1', add_center=True)
    plot_2d_add_subplot(ax=ax_tuple[1], points_groups=[g_A, g_C, g_U, g_V], lines_groups=None, sub_title='2',
                        add_center=True)
    plot_2d_subplots_end(f=fig, zoomed=True)

    # 2x2
    fig, ax_arr = plot_2d_subplots_start(rows=2, cols=2, main_title='2x2')  # 2x2 subplots
    plot_2d_add_subplot(ax=ax_arr[0][0], points_groups=None, lines_groups=[g_q_opt_A, g_q_opt_C], sub_title='1',
                        add_center=True)
    plot_2d_add_subplot(ax=ax_arr[0][1], points_groups=[g_A, g_U], lines_groups=None, sub_title='2', add_center=True)
    plot_2d_add_subplot(ax=ax_arr[1][0], points_groups=[g_A, g_V], lines_groups=[g_Q], sub_title='3', add_center=True)
    plot_2d_add_subplot(ax=ax_arr[1][1], points_groups=[g_A, g_C, g_U, g_V], lines_groups=None, sub_title='4',
                        add_center=False)
    plot_2d_subplots_end(f=fig, zoomed=True)
    """
    assert points_groups is not None or lines_groups is not None, 'need at least 1 group'

    # casting to numpy if torch and checking d==2
    if points_groups is not None:
        for i in range(len(points_groups)):
            A, color, lbl = points_groups[i]
            if isinstance(A, torch.Tensor):
                A = torch_to_numpy(A)
                points_groups[i] = (A, color, lbl)
            assert len(A.shape) > 1 and A.shape[1] == 2, 'A is not valid or d!=1. {}: A.shape={}'.format(lbl, A.shape)
    if lines_groups is not None:
        for i in range(len(lines_groups)):
            Q, color, lbl = lines_groups[i]
            if isinstance(Q, torch.Tensor):
                Q = torch_to_numpy(Q)
                lines_groups[i] = (Q, color, lbl)
            if len(Q.shape) == 2:  # got 1 line
                Q = Q.reshape(1, 2, 1)
                lines_groups[i] = (Q, color, lbl)
            assert Q.shape[1] == 2 and Q.shape[2] == 1, '{}: Q shape {}'.format(lbl, Q.shape)

    # dist_from_mid = 0.5  # needed for lines
    all_points = np.empty((0, 2), dtype=float)

    if points_groups is not None:
        for point_group in points_groups:
            A, color, lbl = point_group
            X, y = de_augment_numpy(A)
            ax.scatter(X, y, color=color, marker='.', label=lbl)
            all_points = np.concatenate((all_points, A), axis=0)
    else:
        for line_group in lines_groups:
            Q, color, lines_label = line_group
            for i, q in enumerate(Q):
                a, b = q[0][0], q[1][0]
                two_xs = np.array([-5, 5], dtype=float)
                two_ys = a * two_xs + b
                two_points = augment_x_y_numpy(two_xs, two_ys)
                if points_groups is None and add_center:  # gather some points for circle
                    all_points = np.concatenate((all_points, two_points), axis=0)
    distance_mat = distance.cdist(all_points, all_points, 'euclidean')
    longest_dist = max(distance_mat[0])
    dist_from_mid = longest_dist

    if lines_groups is not None:
        for line_group in lines_groups:
            Q, color, lines_label = line_group
            for i, q in enumerate(Q):
                label = lines_label if i == 0 else ''  # add lines label once
                a, b = q[0][0], q[1][0]
                two_xs = np.array([-5, 5], dtype=float)
                two_ys = a * two_xs + b
                two_points = augment_x_y_numpy(two_xs, two_ys)
                mid = mid_2_points(two_points[0], two_points[1])
                u = get_diff_2_points(two_points[0], two_points[1], normed=True)
                p_up = mid + dist_from_mid * u
                p_down = mid - dist_from_mid * u
                if points_groups is None and add_center:  # gather some points for circle
                    all_points = np.concatenate((all_points, p_up.reshape(1, 2), p_down.reshape(1, 2)), axis=0)
                # print('x.shape {}, x:{}'.format(two_xs.shape, two_xs.tolist()))
                # print('y.shape {}, y:{}'.format(two_ys.shape, two_ys.tolist()))
                # print('a.shape {}, a:{}'.format(two_points.shape, two_points.tolist()))
                # print('mid_yeter point = {}'.format(mid))
                # print('p_up = {}'.format(p_up))
                # print('p_down = {}'.format(p_down))
                # print('dist(p_up, p_down)={}'.format(np.linalg.norm(p_up - p_down)))

                ax.plot([p_down[0], p_up[0]], [p_down[1], p_up[1]], color=color, marker='x', linestyle='dashed',
                        linewidth=1, markersize=12, label=label)

    if add_center:
        centers = find_centers(all_points, k=1)
        distance_mat = distance.cdist(centers, all_points, 'euclidean')
        rad = max(distance_mat[0])
        ax.scatter(centers[0][0], centers[0][1], color='orange', marker='o', label='center')
        circle_cover = plt.Circle(xy=(centers[0][0], centers[0][1]), radius=rad, color='orange', fill=False,
                                  linewidth=0.5)
        ax.add_artist(circle_cover)
    ax.grid()
    ax.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5, edgecolor='black')
    if sub_title is not None:
        ax.set_title(sub_title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if points_groups is not None:
        all_points_x, all_points_y = de_augment_numpy(all_points)
        min_x, max_x = min(all_points_x), max(all_points_x)
        min_y, max_y = min(all_points_y), max(all_points_y)
        ax.set_xticks(get_x_ticks_list(min_x, max_x, p=3))
        ax.set_yticks(get_x_ticks_list(min_y, max_y, p=3))
        ten_p_diff_x = (max_x - min_x) / 10
        ten_p_diff_y = (max_y - min_y) / 10
        ax.set_xlim(min_x - ten_p_diff_x, max_x + ten_p_diff_x)
        ax.set_ylim(min_y - ten_p_diff_y, max_y + ten_p_diff_y)
        ax.margins(0.1)
    # else:
    #     ax.autoscale()
    #     ax.margins(0.1)
    # center_x, center_y = 0, 200
    # offset = 100
    # ax.set_xlim(center_x-offset, center_x+offset)
    # ax.set_ylim(center_y-offset, center_y+offset)
    return


def plot_2d_subplots_end(save_path: str = None, show_plot: bool = True, f: Figure = None, hspace: float = 0.3,
                         wspace: float = 0.3, dpi=200, zoomed: bool = False):
    if f is not None:
        f.subplots_adjust(hspace=hspace, wspace=wspace)
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('\tsaved to {}.png'.format(save_path))
    if show_plot:
        if zoomed:
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
        plt.show()
    plt.cla()
    return


def plot_2d_lines(groups: list, title: str = '', label_x: str = 'x', label_y: str = 'y', add_center: bool = False,
                  save_path: str = None, show_plot: bool = True, zoomed: bool = False):
    """
    np.random.seed(42)
    Q = np.random.randint(low=-10, high=10, size=(5, 2, 1))
    q_opt = np.random.randint(low=-10, high=10, size=(2, 1))
    g_Q = (Q, 'darkviolet', 'Q1')
    g_q_opt = (q_opt, 'black', 'q_opt')
    plot_2d_lines(groups=[g_Q, g_q_opt], title='lines', zoomed=True, add_center=True)
    """
    fig, ax = plot_2d_subplots_start(1, 1)
    plot_2d_add_subplot(ax=ax, points_groups=None, lines_groups=groups, sub_title=title, label_x=label_x,
                        label_y=label_y, add_center=add_center)
    plot_2d_subplots_end(save_path=save_path, show_plot=show_plot, zoomed=zoomed)
    return


def plot_2d_scatter(groups: list, title: str = '', label_x: str = 'x', label_y: str = 'y', add_center: bool = False,
                    save_path: str = None, show_plot: bool = True, zoomed: bool = False):
    """
    np.random.seed(42)
    A = np.random.randint(low=-10, high=10, size=(10, 2))
    C = np.random.randint(low=-10, high=10, size=(5, 2))
    g_A = (A, 'g', 'A')
    g_C = (C, 'r', 'C')
    plot_2d_scatter(groups=[g_A, g_C], title='scatter', zoomed=True, add_center=True)
    """
    fig, ax = plot_2d_subplots_start(1, 1)
    plot_2d_add_subplot(ax=ax, points_groups=groups, lines_groups=None, sub_title=title, label_x=label_x,
                        label_y=label_y, add_center=add_center)
    plot_2d_subplots_end(save_path=save_path, show_plot=show_plot, zoomed=zoomed)
    return


def plot_2d_points_and_lines(points_groups: list = None, lines_groups: list = None, title: str = '', label_x: str = 'x',
                             label_y: str = 'y', add_center: bool = False, save_path: str = None,
                             show_plot: bool = True,
                             zoomed: bool = False):
    """
    np.random.seed(42)
    A = np.random.randint(low=-10, high=10, size=(10, 2))
    C = np.random.randint(low=-10, high=10, size=(5, 2))
    Q = np.random.randint(low=-10, high=10, size=(5, 2, 1))
    q_optA = np.random.randint(low=-10, high=10, size=(2, 1))
    q_optC = np.random.randint(low=-10, high=10, size=(2, 1))

    g_A = (A, 'g', 'A')
    g_C = (C, 'r', 'C')
    g_q_opt_A = (q_optA, 'g', 'A q_opt')
    g_q_opt_C = (q_optC, 'r', 'C q_opt')
    g_Q = (Q, 'darkviolet', 'trainQ')
    plot_2d_points_and_lines(points_groups=[g_A, g_C], lines_groups=[g_q_opt_A, g_q_opt_C, g_Q],
                             title='Linear regression', zoomed=True, add_center=True)
    """
    fig, ax = plot_2d_subplots_start(1, 1)
    plot_2d_add_subplot(ax=ax, points_groups=points_groups, lines_groups=lines_groups, sub_title=title,
                        label_x=label_x, label_y=label_y, add_center=add_center)
    plot_2d_subplots_end(save_path=save_path, show_plot=show_plot, zoomed=zoomed)
    return


def plot_x_y_std(data_x: np.array, groups: list, title: str = None, x_label: str = 'Size', y_label: str = 'Error',
                 save_path: str = None, show_plot: bool = True, with_shift: bool = False):
    """
    data_x: x values
    groups: list of groups s.t. each tuple(y values, y std, color, title)  y std could be None
    example:
        data_x = [10, 20, 30]
        C_errors = [5, 7, 1]
        C_errors_stds = [2, 1, 0.5]
        group_c = (C_errors, C_errors_stds, 'g', 'C')
        U_errors = [10, 8, 3]
        U_errors_vars = [4, 3, 1.5]
        group_u = (U_errors, U_errors_vars, 'r', 'U')
        groups = [group_c, group_u]
        title = 'bla'
        plot_x_y_std(data_x, groups, title)
    :return:
    """
    data_x_last = data_x  # in order to see all STDs, move a little on the x axis
    data_x_jump = 0.5
    data_x_offset = - int(len(groups) / 2) * data_x_jump
    line_style = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 2, "elinewidth": 1, "capsize": 4}
    for i, group in enumerate(groups):
        data_y, std_y = group[0], group[1]  # std_y could be None
        color, label = group[2], group[3]
        if with_shift:  # move x data for each set a bit so you can see it clearly
            dx_shift = [x + i * data_x_jump + data_x_offset for x in data_x]
            data_x_last = dx_shift
        plt.errorbar(data_x_last, data_y, std_y, color=color, fmt='.', label=label, **line_style)

    plt.grid()
    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
    if title is not None:
        plt.title(title)
    plt.xticks(data_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print('\tsaved to {}.png'.format(save_path))
    plt.pause(0.0001)
    if show_plot:
        plt.show(block=True)
    plt.cla()
    return


def plot_2d_regression(A, q_log, q_lin, loss_log, loss_lin):
    from scipy.special import expit

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    data_shape = A.shape
    X, y = A[:, :-1], A[:, -1]
    plt.scatter(X.ravel(), y, color='black', zorder=20, label='A{}'.format(data_shape))
    X_log_line = np.linspace(-10, 10, 300)

    loss = expit(X_log_line * q_log[0] + q_log[1]).ravel()
    plt.plot(X_log_line, loss, color='red', linewidth=3, label='logit(loss={:.3f})'.format(loss_log))

    # loss2 = expit(X_log_line * q_lin[0] + q_lin[1]).ravel()
    # plt.plot(X_log_line, loss2, linewidth=1, label='linear(loss={:.3f})'.format(loss_lin))
    plt.plot(X_log_line, q_lin[0] * X_log_line + q_lin[1], linewidth=1, label='linear(loss={:.3f})'.format(loss_lin))
    plt.axhline(.5, color='.5')

    plt.ylabel('y')
    plt.xlabel('X')

    # min_x, max_x = min(X), max(X)
    # min_y, max_y = min(y), max(y)
    # plt.xticks(get_x_ticks_list(min_x, max_x, p=3))
    # plt.yticks(get_x_ticks_list(min_y, max_y, p=3))
    # ten_p_diff_x = (max_x - min_x) / 10
    # ten_p_diff_y = (max_y - min_y) / 10
    # plt.xlim(min_x - ten_p_diff_x, max_x + ten_p_diff_x)
    # plt.ylim(min_y - ten_p_diff_y, max_y + ten_p_diff_y)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    plt.margins(0.1)

    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.show()
    return


def histogram(values: np.array, title: str, save_path: str = None, show_hist: bool = True, bins_n: int = 50):
    """ plots a histogram """
    # print(sum(values), values.tolist())
    # plt.hist(values, bins=50, facecolor='green', alpha=0.75, range=(values.min(), values.max()))
    # count_in_each_bin, bins, patches = plt.hist(values, bins_n, density=False, facecolor='blue', alpha=0.75)
    plt.hist(values, bins_n, density=False, facecolor='blue', alpha=0.75)
    # print(count_in_each_bin, bins, patches[0])
    # print(list_1d_to_str(count_in_each_bin))

    plt.xlabel('Values')
    plt.ylabel('Bin Count')
    plt.title(title)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print('\tsaved to {}.png'.format(save_path))
    if show_hist:
        plt.show()
    plt.cla()
    return


def compare_images_sets(set_a, set_b, title: str = None):
    """
    build for images BEFORE transform:
    notice images should be in the format:
        gray scale mnist: [number of images, 28, 28]
        RGB  Cifar10    : [number of images, 32, 32, 3]

    :param set_a: array (nd\torch) of images
    :param set_b: array (nd\torch) of images
    :param title: plot title
    plot set a of images in row 1 and set b in row 2
    set_a and set_b can be ndarray or torch arrays
    example:
        from torchvision import datasets
        # choose data set - both work
        # data_root = path to the data else download
        # dataset = datasets.MNIST(root=data_root, train=False, download=False)
        dataset = datasets.CIFAR10(root=data_root, train=False, download=False)
        set_a = dataset.data[:3]
        set_b = dataset.data[10:50]
        compare_images_sets(set_a, set_b)
    """
    n_cols = max(set_a.shape[0], set_b.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, sharex='all', sharey='all', figsize=(15, 4))
    for images, row in zip([set_a, set_b], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if title is not None:
        plt.title(title)
    plt.show()
    return


def compare_images_multi_sets_squeezed(sets_dict: dict, title: str = None) -> str:
    """
    build for images AFTER transform:
    notice images should be in the format:
        gray scale mnist: [number of images, 1, 28, 28]
        RGB  Cifar10    : [number of images, 3, 32, 32]

    :param sets_dict: each entry in dict is title, set of images(np/tensor)
    :param title: for plot
    :return str with details which set in each row
    plot sets of images in rows
    example:
        from torchvision import datasets
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor(), ])
        # choose data set - both work
        # data_root = path to the data else download
        # dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
        dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        images32, labels = iter(data_loader).next()

        images = images32[:16]  # imagine the first 16 are base images and predicted_images are the model output
        predicted_images = images32[16:32]
        d = {'original_data': images, 'predicted_data': predicted_images}
        print(compare_images_multi_sets_squeezed(d))
    """
    from torchvision.utils import make_grid

    for k, v in sets_dict.items():
        if isinstance(sets_dict[k], np.ndarray):
            sets_dict[k] = numpy_to_torch(sets_dict[k])

    all_sets = None
    msg = ''
    set_len = 0
    msg_base = 'row {}: {}, '

    for i, (k, v) in enumerate(sets_dict.items()):
        all_sets = v if all_sets is None else torch.cat((all_sets, v), 0)
        msg += msg_base.format(i, k)
        set_len = v.shape[0]

    grid_images = make_grid(all_sets, nrow=set_len)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.imshow(np.transpose(torch_to_numpy(grid_images), (1, 2, 0)))
    plt.show()
    return msg


def plotCU_groups(A: torch.Tensor, c_size: int, coresets_dict_per_size: dict):
    with torch.no_grad():
        main_title = '|A|={}, c_size={}'.format(t_size_str(A), c_size)
        fig, ax_tuple = plot_2d_subplots_start(rows=2, cols=3, main_title=main_title)
        print(var_to_string(A, 'A', with_data=True))
        for i, (name, coreset_list) in enumerate(coresets_dict_per_size.items()):
            (C, U) = coreset_list[0]
            CU = U.view(U.shape[0], 1) * C
            print('{}:'.format(name))
            print(var_to_string(C, '\tC', with_data=True))
            print(var_to_string(U, '\tU', with_data=True))
            print(var_to_string(CU, '\tCU', with_data=True))
            group1 = (A, 'g', 'A')
            group2 = (C, 'r', 'C')
            group3 = (CU, 'b', 'CU')
            plot_2d_add_subplot(ax=ax_tuple[0][i], points_groups=[group1, group2], sub_title=name, add_center=True)
            plot_2d_add_subplot(ax=ax_tuple[1][i], points_groups=[group1, group2, group3], sub_title=None,
                                add_center=True)
        plot_2d_subplots_end(f=fig, zoomed=True)
    return


def plot3d_scatters(X_list: list, titles: list, labels: list, colors: list, marker: str = 'o', markersize: int = 50):
    """
    dim_3d = 3
    sub_plots_n = 2
    titles = ['p1_cloud', 'c1_cloud_init']
    labels = ['{}({})'.format(0, 'random'), '{}({})'.format(1, 'random1')]
    colors = ['r', 'b']
    X_list = np.random.randint(low=-10, high=10, size=(sub_plots_n, 20, dim_3d))  # must be 3d
    plot3d_scatters(X_list, titles, labels, colors)
    """
    assert len(X_list) == len(titles) == len(labels) == len(colors), '{}=={}=={}=={} ?'.format(len(X_list), len(titles),
                                                                                               len(labels), len(colors))
    fig = plt.figure()
    for i, X in enumerate(X_list):
        ax = fig.add_subplot(1, len(X_list), i + 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colors[i], marker=marker, s=markersize, alpha=1,
                   label='{}: {} blobs'.format(labels[i], X.shape[0]))
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        # noinspection PyUnresolvedReferences
        ax.set_zlabel('Z Label')
        ax.set_title(titles[i])

    plt.show()
    return


def plot3d_scatter(X: np.array, title: str = '3dplot', label: str = 'some label', marker: str = '.',
                   markersize: int = 50, color: str = 'red'):
    """
    dim_3d = 3
    X = np.random.randint(low=-10, high=10, size=(20, dim_3d))  # must be 3d
    utils.plot3d_scatter(X, title='3dplot', label='random', marker='o')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker, s=markersize, alpha=1,
               label='{}({} blobs)'.format(label, X.shape[0]))

    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    # noinspection PyUnresolvedReferences
    ax.set_zlabel('Z Label')
    ax.set_title(title)

    plt.show()
    return


# LOGGER
def init_logger(logger_path: str = None):
    """
    example:
    init_logger('./log.txt')
    log_print('line 1')
    flush_logger()
    log_print('line 2')
    log_print('line 3')
    del_logger()
    """
    global logger
    logger = open(file=logger_path, mode='w', encoding='utf-8') if logger_path is not None else None
    return logger


def flush_logger():
    """ good for loops - writes every iteration if used """
    global logger
    if logger is not None:
        logger.flush()
    return


def log_print(string):
    global logger
    print(string)
    if logger is not None:
        logger.write('{}\n'.format(string))
    return


def log_print_dict(my_dict):
    for k, v in my_dict.items():
        log_print('\t{}: {}'.format(k, v))
    return


def del_logger():
    global logger
    if logger is not None:
        logger.close()
    return


def list_1d_to_str(list_of_floats, float_precision=4):
    """ |list_of_floats| = nx1 """
    p = '{:,.' + str(float_precision) + 'f}'
    out = '['
    for i, num in enumerate(list_of_floats):
        out += p.format(num)
        out += ',' if (i + 1) < len(list_of_floats) else ''
    out += ']'
    return out


def list_2d_to_str(list_of_lists_floats, float_precision=3):
    """ |list_of_floats| = nxm - list of lists of floats """
    p = '{:,.' + str(float_precision) + 'f}'
    out = '['
    for i, list_of_floats in enumerate(list_of_lists_floats):
        row = '['
        for j, num in enumerate(list_of_floats):
            row += p.format(num)
            row += ',' if (j + 1) < len(list_of_floats) else ''
        row += ']'
        out += row
        out += ',' if (i + 1) < len(list_of_lists_floats) else ''
    out += ']'
    return out


# CORESETS
def select_coreset_frequency(P: np.array, SP: np.array, coreset_size: int, W_in: np.array = None,
                             out_torch_double: bool = True) -> (
        np.array, np.array):
    """
    based on BEN and RITA idea to sample without repetitions
    P, SP, W_in - all the same type: numpy or torch
    :param P:  your points. if you have labels: from the shape X|y
    :param SP: your points sensitivity.
    :param coreset_size: the size of desired coreset
    :param W_in: weights of P. if not wighted input - ones(n)
    :param out_torch_double:
    :return:
    """
    n = P.shape[0]
    assert coreset_size <= n
    if W_in is None:
        W_in = np.ones(n)

    prob = SP / SP.sum()
    indices_set = set()
    indices_list = []

    cnt = 0
    while len(indices_set) < coreset_size:
        i = np.random.choice(a=n, size=1, p=prob).tolist()[0]
        indices_list.append(i)
        indices_set.add(i)
        cnt += 1

    hist = np.histogram(indices_list, bins=range(n + 1))[0].flatten()
    indices_list = np.nonzero(hist)[0]
    C = P[indices_list, :]

    W_out = (W_in[indices_list].T * hist[indices_list]).T
    W_out = (W_out.T / (prob[indices_list] * cnt)).T
    W_out = add_cuda(numpy_to_torch(W_out, to_double=out_torch_double))
    return C, W_out


def select_coreset(A: np.array, SP: np.array, coreset_size: int, out_torch: bool = True, out_torch_double=True) -> (
        np.array, np.array):
    """
    :param A: nx(d+1) points. can be torch\numpy
    :param SP: numpy array of nx1 sensitivities for each p in A
    :param coreset_size: size of the coreset
    :param out_torch:
    :param out_torch_double:  double or float
    :return:
    C: subset of A of size coreset_size x (d+1). if A on GPU so will C
    W: weights of each c in C
    """
    n = len(A)
    sp_sum = SP.sum()
    sp_normed = SP / sp_sum
    # replace: False -> unique values, True  -> with reps
    C_ind = np.random.choice(a=range(0, n), size=coreset_size, p=sp_normed, replace=True)
    C = A[C_ind]
    W = sp_sum / (coreset_size * SP[C_ind])
    if out_torch:
        W = add_cuda(numpy_to_torch(W, to_double=out_torch_double))  # NOTICE: you might need different data type
    return C, W


def create_dir_if_not_exists(data_root: str, ack: bool = True, tabs: int = 1):
    if os.path.exists(data_root):
        if ack:
            print('{}{} exists'.format(tabs * '\t', data_root))
    else:
        os.mkdir(data_root)
        if ack:
            print('{}{} created'.format(tabs * '\t', data_root))
    return


def results_by_test(test: str, data: list, fp: float = 5) -> None:
    mean = np.mean(data)
    msg = '\t{:11}[avg={:.5f}]'.format(test, mean)
    print(wu.to_str(data, title=msg, fp=fp, wm=False))
    return


def main():
    # req_file_fp = '../sgd_requirements_{}.txt'.format(wu.get_time_stamp('%Y_%m_%d'))
    # print(req_file_fp)
    # print(os.path.abspath(req_file_fp))
    # wu.generate_requirements_file(req_file_fp)
    return


if __name__ == "__main__":
    wu.main_wrapper(
        main_function=main,
        cuda_off=False,
        torch_v=True,
        tf_v=False,
        cv2_v=False,
        with_profiler=False
    )
