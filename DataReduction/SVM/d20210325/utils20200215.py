# feb 15 2020 commit c1
# https://github.com/pytorch
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
import torch
import numpy as np
from typing import List
from time import time
import cProfile
import pstats
import io
import sys
import matplotlib.pyplot as plt
import os
from torchsummary import summary


def cuda_on():
    return torch.cuda.is_available()


def make_cuda_invisible():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1, 0'
    return


def set_cuda_scope_and_seed(seed, dtype='FloatTensor'):
    """
    https://pytorch.org/docs/stable/tensors.html
    32-bit floating point: torch.cuda.FloatTensor
    64-bit floating point: torch.cuda.DoubleTensor
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda_on():
        def_dtype = 'torch.cuda.' + dtype
        torch.set_default_tensor_type(def_dtype)
        torch.cuda.manual_seed(seed)
        print('working on CUDA. default dtype = {} <=> {}'.format(def_dtype, torch.get_default_dtype()))
    else:
        print('working on CPU')
    return


def add_cuda(var: torch.Tensor):
    """ assigns the variables to GPU if available"""
    if cuda_on():
        var = var.cuda()
    return var


def var_data(title: str, var: torch.Tensor):
    msg = '{:8s}: {:8s}, dtype:{}, trainable:{}, is_cuda:{}'
    size_str = str(var.size())
    size_str = size_str[size_str.find("(") + 1:size_str.find(")")]
    return msg.format(title, size_str, var.dtype, is_trainable(var), is_cuda(var))


def redirect_std_start():
    old_stdout = sys.stdout
    sys.stdout = summary_str = io.StringIO()
    return old_stdout, summary_str


def redirect_std_finish(old_stdout, summary_str):
    sys.stdout = old_stdout
    return summary_str.getvalue()


def get_model_summary_str(p_model, input_size, batch_size):
    a, b = redirect_std_start()
    summary(p_model, input_size, batch_size)
    return redirect_std_finish(a, b)


def torch_uniform(shape, range_low, range_high):
    ret = torch.empty(shape).uniform_(range_low, range_high)
    return ret


def torch_normal(shape, mean, std):
    ret = torch.empty(shape).normal_(mean, std)
    return ret


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn_l = 1
        for s in list(p.size()):
            nn_l = nn_l * s
        pp += nn_l
    return pp


# ALL models must have title and path as members
def model_params_print(model, print_values=False):
    print('{}:'.format(model.title))
    msg = '\t{:15s}: {:15s}, trainable:{}, is_cuda:{}'

    for name, param in model.named_parameters():
        size_str = str(param.size())
        size_str = size_str[size_str.find("(") + 1:size_str.find(")")]
        print(msg.format(name, size_str, is_trainable(param), is_cuda(param)))
        if print_values:
            print('\t\tvalues: {}'.format(param.tolist()))
    return


def save_model(model, ack_print=True):
    torch.save(model.state_dict(), model.path)
    if ack_print:
        print('{} saved to {}'.format(model.title, model.path))
    return


def load_model(model, ack_print=True):
    dst_allocation = None if cuda_on() else 'cpu'
    model.load_state_dict(torch.load(model.path, map_location=dst_allocation))
    model.eval()
    if ack_print:
        print('{} loaded from {}'.format(model.title, model.path))
    return model


def get_dims_size(t: torch.Tensor, ignore_first=True):
    # ignore rows
    total = 1
    my_shape = t.shape[1:] if ignore_first else t.shape
    for d in my_shape:
        total *= d

    # t.view(1, -1).size()
    return total


def revive_model(model):
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def freeze_model(model):
    for param in model.parameters():  # freeze p_model
        param.requires_grad_(False)
    return model


def is_model_frozen(model):
    for param in model.parameters():
        if is_trainable(param) is not False:
            return '{} is NOT frozen'.format(model.title)
    return '{} is frozen'.format(model.title)


def is_model_fully_trainable(model):
    for param in model.parameters():
        if is_trainable(param) is not True:
            return '{} is NOT fully trainable'.format(model.title)
    return '{} is fully trainable'.format(model.title)


def copy_models(model_source, model_target, ignore_layers):
    print('Copying model {} to model {} except {}:'.format(model_source.title, model_target.title, ignore_layers))
    for name, param in model_source.named_parameters():
        if name not in ignore_layers:
            # print('Copying {}'.format(name))
            # print('\t\tvalues orig   : {}'.format(param.tolist()))
            # print('\t\tvalues coreset: {}'.format(model_coreset.state_dict()[name].tolist()))
            model_target.state_dict()[name].copy_(param)
            # print('\t\tvalues coreset: {}'.format(model_coreset.state_dict()[name].tolist()))
        else:
            # msg = '\tlayer with name {:15s} was not copied'
            # print(msg.format(name))
            pass
    return model_target


def freeze_layers(model, trainable_layers_list):
    print('Model {}: freezing all except {}:'.format(model.title, trainable_layers_list))
    for name, param in model.named_parameters():
        if name not in trainable_layers_list:
            param.requires_grad_(False)
            # print('frozen {}'.format(name))
        else:
            param.requires_grad_(True)
            # print('alive {}'.format(name))
    return model


def is_trainable(var: torch.Tensor):
    return var.requires_grad


def is_cuda(var: torch.Tensor):
    return var.is_cuda


def get_lr(optimizer: torch.optim):
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer: torch.optim, new_lr: float):
    optimizer.param_groups[0]['lr'] = new_lr
    return optimizer


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

    def __init__(self, optimizer, factor, patience, min_lr):
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
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best = None
        return

    def should_early_stop(self, loss):
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


def numpy_to_torch(var_np: np.array, detach: bool = False):
    if detach:
        var_torch = add_cuda(torch.from_numpy(var_np).float()).detach()
    else:
        var_torch = add_cuda(torch.from_numpy(var_np).float())
    return var_torch


def torch_to_numpy(var_torch: torch.Tensor):
    if is_trainable(var_torch):
        var_np = var_torch.detach().cpu().numpy()
    else:
        var_np = var_torch.cpu().numpy()
    return var_np


def make_trainable_variable(list_tensors: List[torch.Tensor], trainable: bool = True):
    """ changes trainable status for each tensor in the list
    :param list_tensors:
    :param trainable: True or False
    :return:
    """
    for tensor in list_tensors:
        tensor.requires_grad_(trainable)
    return


def get_current_time():
    return time()


def get_time_str(start_time):
    hours, rem = divmod(int(time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def profiler_start():
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiler_end(pr):
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    return s.getvalue()


def random_init(c_size, train_x):
    print('random_init')
    coreset_size = ()
    coreset_size += (c_size,)
    for d in train_x[0].shape:
        coreset_size += (d,)
    coreset_x = torch.rand(size=coreset_size, requires_grad=True).float()
    return coreset_x


def subset_init(c_size, train_x):
    print('subset_init')
    n = train_x.shape[0]
    # print('n {}'.format(n))
    perm = np.random.permutation(n)
    # print('perm {}'.format(perm))
    idx = perm[:c_size]
    # print('idx[0] {}'.format(idx[0]))
    coreset_x = train_x[idx].clone().float()
    coreset_x.requires_grad_(True)
    return coreset_x


def compare_images_sets(set_a, set_b):
    """
    plot set a of images in row 1 and set b in row 2
    set_a and b can be ndarray or torch arrays
    notice images should be in the format:
        gray scale mnist: [number of images, 28, 28]
        RGB  Cifar10    : [number of images, 32, 32, 3]
    example:
        from torchvision import datasets
        # choose data set
        dataset = datasets.MNIST(root='./auto_encoder/data', train=False, download=False)
        dataset = datasets.CIFAR10(root='./auto_encoder/data', train=False, download=False)
        set_a = dataset.data[:3]
        set_b = dataset.data[10:50]
        compare_images_sets(set_a, set_b)
    """
    # print(set_a.shape)
    n_cols = max(set_a.shape[0], set_b.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, sharex='all', sharey='all', figsize=(15, 4))
    for images, row in zip([set_a, set_b], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
    return


def compare_images_multi_sets_squeezed(sets_dict, title=None):
    """
    DOC STRING NEEDS EDITING
    for each set in set_list: plot set a of images in row i
    set_list must have at least 1 set
    set_i can be ndarray or torch arrays
    notice images should be in the format:
        gray scale mnist: [number of images, 1, 28, 28]
        RGB  Cifar10    : [number of images, 3, 32, 32]
    example:
    from torchvision import datasets
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(), ])
    # choose data set
    dataset = datasets.MNIST(root='./auto_encoder/data', train=False, download=False, transform=transform)
    dataset = datasets.CIFAR10(root='./auto_encoder/data', train=False, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    images32, labels = iter(data_loader).next()

    images = images32[:16]  # imagine the first 16 are base images and predicted_images are the model output
    predicted_images = images32[16:32]
    set_list = []
    set_list.append(images)
    set_list.append(predicted_images)
    compare_images_multi_sets_squeezed(set_list)
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


def histogram(values, title):
    plt.hist(values, bins=50, facecolor='green', alpha=0.75, range=(values.min(), values.max()))
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title(title)
    plt.grid(True)
    plt.show()
    return


def plot_x_y_std(data_x, groups, title=None, x_label='Size', y_label='Error', with_std=True):
    """
    :param with_std: if false, regular graph
    :param title:
    :param x_label:
    :param y_label:
    :param data_x: x values
    :param groups: list of groups s.t. each tuple(y values, y std, color, title)
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
    if with_std:
        # in order to see all STDs, move a little on the x axis
        # group[0]: data_x += -int(len(groups) / 2) * data_x_jump. each group will move data_x by data_x_jump
        data_x_jump = 0.5
        data_x_offset = - int(len(groups) / 2) * data_x_jump
        # data_x_shift = g_l/
        line_style = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 2, "elinewidth": 1, "capsize": 4}
        for i, group in enumerate(groups):
            data_y, std_y = group[0], group[1]
            color, label = group[2], group[3]
            dx_shift = [x + i * data_x_jump + data_x_offset for x in data_x]
            plt.errorbar(dx_shift, data_y, std_y, color=color, fmt='.', label=label, **line_style)
    else:
        for i, group in enumerate(groups):
            data_y = group[0]
            color, label = group[2], group[3]
            plt.plot(data_x, data_y, '-{}.'.format(color), label=label)

    plt.grid()
    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
    if title is not None:
        plt.title(title)
    plt.xticks(data_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return


def plot_x_y(data_x, groups, title=None, x_label='Size', y_label='Error'):
    """
    :param title:
    :param x_label:
    :param y_label:
    :param data_x: x values
    :param groups: each group:= tuple(values, color, title)
    :return:
    """
    for group in groups:
        data_y = group[0]
        color, label = group[1], group[2]
        plt.plot(data_x, data_y, '-{}.'.format(color), label=label)

    plt.grid()
    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
    if title is not None:
        plt.title(title)
    plt.xticks(data_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    # if save_full_path is not None:
    #     plt.locator_params(nbins=10)
    #     plt.savefig(save_full_path, dpi=100, bbox_inches='tight')

    # wm = plt.get_current_fig_manager()
    # wm.window.state('zoomed')
    return


def init_logger(logger_path=None):
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


def augment_x_y(X, y, out_torch=True):
    """
    :param out_torch: if true returns tensor, else type(X), type(y)
    :param X: nxd points
    :param y: nx1 labels
    :return: A of size nx(d+1)
    """
    assert X.shape[0] == y.shape[0], 'row count must be the same'
    if isinstance(X, np.ndarray):
        assert isinstance(y, np.ndarray), 'X is ndarray but y is not'
        A = np.column_stack((X, y))
        if out_torch:
            A = add_cuda(numpy_to_torch(A))
    else:  # assuming torch
        if len(y.shape) < len(X.shape):  # change y size()=[n,] to size()=[n,1]
            y = y.view(y.shape[0], 1)
        A = torch.cat((X, y), 1)

    # if out_torch is False:
    #     y = y.reshape(y.shape[0], 1)
    #     A = np.concatenate((X, y), axis=1)
    #
    # else:
    #     A = torch.cat((X, y), 1)
    return A


def de_augment(A, out_torch=True):
    """
    :param out_torch: if true returns tensor, else type(A)
    :param A: nx(d+1) points with labels
    :return: X nxd and y nx1
    """
    if len(A.shape) == 1:
        if isinstance(A, torch.Tensor):
            A = A.view(1, A.shape[0])
        elif isinstance(A, np.ndarray):
            A = A.reshape(1, A.shape[0])

    X, y = A[:, :-1], A[:, -1]
    if isinstance(A, np.ndarray) and out_torch:
        X, y = add_cuda(numpy_to_torch(X)), add_cuda(numpy_to_torch(y))

    if len(y.shape) < len(X.shape):  # change y size()=[n,] to size()=[n,1]
        if isinstance(y, torch.Tensor):
            y = y.view(y.shape[0], 1)
        elif isinstance(y, np.ndarray):
            y = y.reshape(y.shape[0], 1)
    # if A.dim() > 1:
    #     X, y = A[:, :-1], A[:, -1]
    # else:
    #     X, y = A[:-1], A[-1]
    # print(X.shape, y.shape)
    return X, y


def filter_classes(A, keep_classes_list, n):
    X, y = de_augment(A)
    all_classes_indices = None
    # print('Count classes:')
    # cnt = count_keys(y)
    # for i in range(len(cnt)):
    #     print('\tClass {}: {} samples'.format(i, cnt[i]))

    print('keeping only classes 0 and 1 (n={} from each)'.format(n))
    for keep_class in keep_classes_list:
        class_indices = (y == keep_class).nonzero()[:, 0]
        class_indices = class_indices[:n]
        print('\ttaking first {} points from class {}'.format(class_indices.shape[0], keep_class))
        all_classes_indices = class_indices if all_classes_indices is None else torch.cat(
            (all_classes_indices, class_indices), 0)

    A = A[all_classes_indices]
    print('new A.shape {}'.format(A.shape))
    for i in range(all_classes_indices.shape[0]):
        assert A[i][-1] == 0 or A[i][-1] == 1

    # _, y = de_augment(A)
    # cnt = count_keys(y)
    # for i in range(len(cnt)):
    #     print('\tClass {}: {} samples'.format(i, cnt[i]))
    return A


def count_keys(values_array):
    """
    :param values_array: nx1 array (torch, numpy, list)
    :return: counter dict
    example: cnt = count_keys(['red', 'blue', 'red', 'green', 'blue', 'blue'])
             out: Counter({'blue': 3, 'red': 2, 'green': 1})
    """
    from collections import Counter
    cnt = Counter()
    if isinstance(values_array, torch.Tensor):
        for value in values_array:
            cnt[value.item()] += 1
    elif isinstance(values_array, np.ndarray) or isinstance(values_array, list):
        for value in values_array:
            cnt[value] += 1
    return cnt


def list_to_str(list_of_nxd, float_precision=3):
    row_base = '{:.' + str(float_precision) + 'f}'
    all_rows_str = ''
    for i, d in enumerate(list_of_nxd):
        row_str = '('
        for j, dim in enumerate(d):
            row_str += row_base.format(dim)
            row_str += ', ' if (j + 1) < len(d) else ''
        row_str += ')'
        all_rows_str += row_str
    return all_rows_str


def select_coreset_ben(P: np.array, SP: np.array, coreset_size: int, W_in: np.array = None) -> (np.array, np.array):
    """
    based on BEN's code
    P, SP, W_in - all the same type: numpy or torch
    :param P:  your points. if you have labels: from the shape X|y
    :param SP: your points sensitivity.
    :param coreset_size: the size of desired coreset
    :param W_in: weights of P. if not wighted input - ones(n)
    :return:
    """
    # BEN ORIGINAL CODE
    # def compute_coreset
    # (self, points: torch.Tensor, weights: torch.Tensor, sensitivity: torch.Tensor, coreset_size: int):
    #     assert coreset_size <= points.shape[0]
    #     prob = sensitivity.cpu().numpy()
    #
    #     indices = set()
    #     idxs = []
    #
    #     cnt = 0
    #     while len(indices) < coreset_size:
    #         i = np.random.choice(a=points.shape[0], size=1, p=prob).tolist()[0]
    #         idxs.append(i)
    #         indices.add(i)
    #         cnt += 1
    #
    #     hist = np.histogram(idxs, bins=range(points.shape[0] + 1))[0].flatten()
    #     idxs = np.nonzero(hist)[0]
    #     coreset = points[idxs, :]
    #
    #     weights = (weights[idxs].t() * torch.tensor(hist[idxs]).float()).t()
    #     weights = (weights.t() / (torch.tensor(prob[idxs]) * cnt)).t()
    #
    #     return coreset, weights
    n = P.shape[0]
    assert coreset_size <= n
    if W_in is None:
        W_in = np.ones(n)

    prob = SP / SP.sum()
    indices = set()
    idxs = []

    cnt = 0
    while len(indices) < coreset_size:
        i = np.random.choice(a=n, size=1, p=prob).tolist()[0]
        idxs.append(i)
        indices.add(i)
        cnt += 1

    # print('indices', indices)
    # print('cnt', cnt)
    # print('idxs', idxs)

    hist = np.histogram(idxs, bins=range(n + 1))[0].flatten()
    # print('hist', hist)
    idxs = np.nonzero(hist)[0]
    # print('idxs', idxs)
    C = P[idxs, :]
    # print('C', C.tolist())

    # W_out = (W_in[idxs].t() * torch.tensor(hist[idxs]).float()).t()
    W_out = (W_in[idxs].T * hist[idxs]).T
    # print('W_out', W_out.tolist())
    W_out = (W_out.T / (prob[idxs] * cnt)).T
    W_out = add_cuda(numpy_to_torch(W_out))
    # print('W_out', W_out.tolist())
    # print(C.shape, W_out.shape)
    return C, W_out


def select_coreset_old_way(A, SP: np.array, coreset_size: int, out_torch: bool = True):
    """
    :param A: nx(d+1) points. can be torch\numpy
    :param SP: numpy array of nx1 sensitivities for each p in A
    :param coreset_size: size of the coreset
    :param out_torch: if you need W as torch
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
        W = add_cuda(numpy_to_torch(W))  # NOTICE: you might need different data type
    return C, W


def get_uniform_data_A(n, d, bottom, top):
    """ :return: A: torch of size nx(d+1) """
    A = torch_uniform((n, d + 1), bottom, top)
    return A


def get_gaus_data_A(n, d, miu, std):
    """ :return: A: torch of size nx(d+1) """
    A = torch_normal((n, d + 1), miu, std)
    return A


def main():
    # demo_linear()
    return


if __name__ == "__main__":
    # g_s = profiler_start()
    g_start_time = get_current_time()
    print("Python  Version {}".format(sys.version))
    print("PyTorch Version {}".format(torch.__version__))
    main()
    print('Total run time {}'.format(get_time_str(g_start_time)))
    # print(profiler_end(g_s))

# def demo_linear():
#     def gen_points(points_count, b=2, a=1):
#         np.random.seed(42)
#         x = np.random.rand(points_count, 1)
#         y = a + b * x + .1 * np.random.randn(points_count, 1)
#         print('generated {} points around the line y={}x+{}'.format(points_count, b, a))
#         return x, y
#
#     def split_points(x, y, split):
#         split_val = int(split * len(x))
#         idx = np.arange(len(x))
#         np.random.shuffle(idx)
#
#         train_idx = idx[:split_val]
#         val_idx = idx[split_val:]
#
#         x_train, y_train = x[train_idx], y[train_idx]
#         x_val, y_val = x[val_idx], y[val_idx]
#         return x_train, y_train, x_val, y_val
#
#     torch.manual_seed(42)
#     x, y = gen_points(points_count=100)
#     x_train, y_train, x_val, y_val = split_points(x, y, 0.8)
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('nn_linear_reg(device={}):'.format(device))
#     x_train_tensor = torch.from_numpy(x_train).float().to(device)
#     y_train_tensor = torch.from_numpy(y_train).float().to(device)
#
#     a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#     b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#
#     optimizer = torch.optim.Adam([a, b], lr=0.1)
#     optimizer = OptimizerHandler(optimizer, factor=0.5, patience=50, min_lr=0.00005)
#     es = EarlyStopping(patience=30)
#     best = np.inf
#     n_epochs = 1000
#
#     for epoch in range(1, n_epochs + 1):
#         yhat = a + b * x_train_tensor
#         error = y_train_tensor - yhat
#         loss = (error ** 2).mean()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         current_lr = optimizer.lr()
#         optimizer.update_lr()
#         # current_lr = optimizer.param_groups[0]['lr']
#         if loss.item() < best:
#             best = loss.item()
#         print('{:4}/{:4} lr {} loss {:.20f}, best {:.20f}'.format(epoch, n_epochs, current_lr, loss.item(), best))
#         if es.should_early_stop(loss.item()):
#             print('early stop')
#             break
#
#     print('\tnn_linear_reg(way 3) - line found: y={:.2f}x+{:.2f}'.format(b.item(), a.item()))
#     return 0