import torch
import numpy as np
from typing import Callable
import time
import copy


def get_time_str(start_time: float) -> str:
    hours, rem = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def calc_SP(A: torch.tensor, f: Callable, epochs: int, lr: float, x_sources: torch.tensor, verbose: int = 0) -> (
        np.array, np.array):
    """
    FUTURE: support weighted A
    :param A: all points
    :param f: loss_function(data, query, weights for data)
    :param epochs:
    :param lr:
    :param x_sources: list of initializers for x
    :param verbose: 0,1,2. 0 for no print, 1 for minimal, 2 for all
    :return: SP_np for sensitivities and X_np for the maximizing xs
    """
    PROGRESS = int(A.shape[0] / 10)
    start_time = time.time()
    if verbose > 0:
        print('calc_SP:')

    SP, X = [], []
    for i, point in enumerate(A):
        biggest_spi, best_max_x = -1.0, None
        for x_source in x_sources:
            sp_i, maximizing_x = calc_1_sp(A, point, f, epochs, lr, x_source, verbose)
            if verbose > 1:
                print('\t\tsp_{}: {} (x={})'.format(i, sp_i, maximizing_x.tolist()))
            if sp_i > biggest_spi:
                biggest_spi = sp_i
                best_max_x = maximizing_x
        SP.append(biggest_spi)
        X.append(best_max_x)

        if (i + 1) % PROGRESS == 0 or (i + 1) == 10:
            print('Done {}/{} time so far {}'.format(i + 1, len(A), get_time_str(start_time)))
        # break
    SP = np.array(SP)

    if isinstance(x_sources[0], torch.Tensor):
        X = np.array([X[i].cpu().numpy() for i in range(len(X))])
    else:  # probably nn.model
        X = np.zeros(len(SP))  # TODO fix: cant save models the same way as tensor

    if verbose > 0:
        print('\tAll SP(size={}, sum={}):'.format(len(SP), sum(SP)))
        for i, sens in enumerate(SP):
            print('\t\tsp_{}: {}, x={}'.format(i, sens, X[i].tolist()))
    return SP, X


def calc_1_sp(A: torch.tensor, ai: torch.tensor, f: Callable, epochs: int, lr: float, x_source: torch.tensor,
              verbose: int) -> (float, torch.Tensor):
    """
    :param A: all points
    :param ai: the point we are calculating sensitivity for
    :param f: lost function. it must be in the shape func(P, x, Wp=None) and return torch
    :param epochs: number of epochs
    :param lr: learning rate
    :param x_source: initialized x
    :param verbose: if bigger than 1, prints leaning updates
    :return:
    """
    if verbose > 1:
        print('\tcalc_1_sp for point = {}'.format(ai.tolist()))
    maximal_loss, maximizing_x = -float("inf"), None  # init
    x = None
    optimizer = None
    if isinstance(x_source, torch.Tensor):
        x = x_source.clone()  # don't change given x
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)
    elif isinstance(x_source, torch.nn.Module):
        x = copy.deepcopy(x_source)
        for param in x.parameters():
            param.requires_grad_(True)
        optimizer = torch.optim.Adam(x.parameters(), lr=lr)

    for i in range(1, epochs + 1):
        # ============ Forward ============
        # loss_ai = f(ai, x)
        loss_ai = f(ai.view(1, ai.shape[0]), x)  # view ai as an array of size (1,d) and not (d)
        loss_A = f(A, x)
        loss = -loss_ai / loss_A  # minimizing the negative loss <==> maximizing loss
        # ============ Backward ============
        optimizer.zero_grad()
        loss.backward()
        loss_neg = -loss.item()
        if verbose > 1:
            msg = '\t\t[{:3}/{:3}] loss_ai:{:.6f}, loss_A:{:.6f}, loss:{:.6f}, x:{}'
            print(msg.format(i, epochs, loss_ai, loss_A, loss_neg, x.clone().detach().tolist()))

        if loss_neg > maximal_loss:  # "save model"
            maximal_loss = loss_neg
            if isinstance(x, torch.Tensor):
                maximizing_x = x.clone().detach()
            elif isinstance(x, torch.nn.Module):
                maximizing_x = None  # TODO fix saving model
                # maximizing_x = copy.deepcopy(x)
                # from Dependencies.UtilsScript import set_model_status
                # set_model_status(maximizing_x, status=False, status_print=False)
        optimizer.step()
    return maximal_loss, maximizing_x


# def relu_on_mult(P: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.float32:
#     from utils import var_to_string
#     mult = P @ q
#     rel = torch.relu(mult)
#     if w is not None:
#         rel = w * rel
#     res = torch.sum(rel)
#     print(var_to_string(mult, 'mult', with_data=True))
#     print(var_to_string(rel, 'rel', with_data=True))
#     print(var_to_string(res, 'res', with_data=True))
#
#     if res == 0:  # in this problem, res could be zero
#         res += 1e-8  # replace with epsilon
#     return res


def lin_reg_loss(A: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.float32:
    """
    loss_function for linear_regression
    :param A: of size nxd+1. A=X|y where X is nxd and y is 1x1
    :param q: of size d+1x1. q=slopes|bias
        where the first d elements are the slopes (1 for each dim). the last is the bias
    :param w:  of size nx1 weights for each data point if A is weighted
    :return: the Euclidean 2-norm || y - (aX + b) ||^2
    doing:
        y_hat = X @ a + b  # (nx1)
        diff = y - y_hat  # (nx1) -> reshape to size (n)
        diff_squared = diff^2  # (n). element wise ^2
        if w is not None:
            diff_squared = diff_squared * w  # (n). weight added
        loss = sum(diff_squared)  # (1). scalar
    """
    from Dependencies.UtilsScript import de_augment_torch

    X, y = de_augment_torch(A)
    n, d = X.shape

    a, b = q[:-1], q[-1]
    assert y.shape == (n, 1) and a.shape == (d, 1) and b.shape[0] == 1
    y_hat = X @ a + b
    error = (y_hat - y).view(n)  # change size from nx1 to n
    error = error ** 2
    if w is not None:
        error = w * error
    loss = error.sum()
    return loss


def sensitivity_example():
    from Dependencies.UtilsScript import set_cuda_scope_and_seed, make_cuda_invisible, torch_normal, var_to_string, \
        plot_2d_scatter, histogram

    g_start_time = time.time()
    make_cuda_invisible()
    set_cuda_scope_and_seed(42)

    # DATA PARAMS: 10 points in 2d
    n, d, k = 10, 2, 1  # to assign k>1, the loss function must support it.
    A = torch_normal((n, d), miu=0, std=10)
    A[0][0], A[0][1] = 100, 4  # this points should very high sp in regard to the rest

    print(var_to_string(A, 'A', with_data=True))

    # SP PARAMS
    epochs, lr, verbose = 50, 0.1, 2

    q_init_size = 1
    Q_init = torch_normal((q_init_size, d, k), miu=0, std=10)  # group of init
    print(var_to_string(Q_init, 'Q', with_data=True))

    sensitivities, xs = calc_SP(A, f=lin_reg_loss, epochs=epochs, lr=lr, x_sources=Q_init, verbose=verbose)

    print('Total run time {}'.format(get_time_str(g_start_time)))
    group1 = (A, 'g', 'A {}'.format(A.shape))
    plot_2d_scatter([group1])
    histogram(sensitivities, 'SP(A): |A|={}'.format(A.shape))
    return


if __name__ == "__main__":
    sensitivity_example()
