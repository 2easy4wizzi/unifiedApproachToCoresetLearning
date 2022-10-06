import torch
from typing import Callable
import time


def get_time_str(start_time: float) -> str:
    hours, rem = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def MEE(A: torch.tensor, C: torch.tensor, U: torch.tensor, x_source: torch.tensor, f: Callable, epochs: int,
        lr: float, verbose: int = 0) -> (torch.tensor, float):
    """
    # TODO MEE try to max |1 - f(C,U,q)/f(A,q)|. 2 cases:
        # 1 decreasing f(C,U,q) and increasing f(A,q): loss could be maximum 1
        # 2 increasing f(C,U,q) and decreasing f(A,q): loss could be infinity
        # if the optimizer does the wrong starting move, it might get stuck in the wrong result
        possible fix - 2 iters: force him to go in both directions
    FUTURE: support weighted A
    objective - find the x that maximize: |1 - loss(C,x)/loss(P,x)| aka effective epsilon
    beware of local minimum. you should try different range_low and range_high
    :param A: all points
    :param C: the coreset (normally a subset of A, but no mandatory)
    :param U: weights for each coreset point
    :param x_source: a query
    :param f: loss_function(data, query, weights for data)
    :param epochs: number of epochs
    :param lr: learning rate of the optimizer
    :param verbose: 0 or 1 for prints
    :return: maximal_error the highest |1 - loss(C,x)/loss(P,x)| found
    :return: maximizing_x the x from the the highest |1 - loss(C,x)/loss(P,x)| found
    """
    maximal_error, maximizing_x = -float("inf"), None  # init
    x = x_source.clone()  # don't change given x
    if verbose > 0:
        print('MEE:')
        loss_init = abs(1 - f(C, x, U).item() / f(A, x).item())
        print('\tinitial x {}'.format(x.tolist()))
        print('\tinitial epsilon {}'.format(loss_init))
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=lr)
    for epoch in range(1, epochs + 1):
        # ============ Forward ============
        loss_p = f(A, x)
        loss_c = f(C, x, U)
        loss = -torch.abs(1 - loss_c / loss_p)  # minimizing the negative loss <==> maximizing loss
        # ============ Backward ============
        optimizer.zero_grad()
        loss.backward()
        loss_neg = -loss.item()
        if verbose > 0:
            msg = '\t[{:3}/{:3}] loss_c:{:.6f}, loss_p:{:.6f}, loss:{:.6f}'
            print(msg.format(epoch, epochs, loss_c, loss_p, loss_neg))

        if loss_neg > maximal_error:  # "save model"
            maximal_error = loss_neg
            maximizing_x = x.clone().detach()

        optimizer.step()
    if verbose > 0:
        print('maximizing_x {}'.format(maximizing_x.tolist()))
        print('maximal epsilon {}'.format(maximal_error))
    return maximizing_x, maximal_error


def one_point_center_loss(A: torch.Tensor, center: torch.Tensor, W: torch.Tensor = None) -> torch.float32:
    """
    :param A: nxd
    :param center: dx1 (could be expanded)
    :param W: nx1
    :return:
    """
    # from utils import var_to_string
    dst = torch.cdist(A, center).view(A.shape[0])  # (n,1) -> (n). 1 result per point
    # print(var_to_string(dst, 'dst', with_data=True))
    if W is not None:
        dst = W * dst
        # print(var_to_string(dst, 'dst', with_data=True))
    max_dst = torch.max(dst)
    # print(var_to_string(max_dst, 'max_dst', with_data=True))
    return max_dst


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


def mee_example():
    from Dependencies.UtilsScript import set_cuda_scope_and_seed, make_cuda_invisible, torch_normal, var_to_string, \
        plot_2d_scatter

    g_start_time = time.time()
    make_cuda_invisible()
    set_cuda_scope_and_seed(42)

    # example designed for 2d data:
    # A = 10 points 2d, C = 3 points 2d, U = weight for each point (n/c_size=3.33)
    # to assign k>1, the loss function must support it.
    n, c_size, d, k = 10, 5, 2, 1

    A = torch_normal((n, d), miu=100, std=1)
    A[:, 1] = 0  # zero the ys
    C = torch_normal((c_size, d), miu=-100, std=1)
    C[:, 1] = 0  # zero the ys
    U = torch.zeros(c_size) + 2  # weights of C
    x = torch.zeros((k, d))  # origin

    # 1 point center:
    # x (0,0), U =[2,2,2,2,2]
    # loss(A,x) = max(dist(A,x)) = max([~100,...,~100]) = ~100
    # loss(C,U,x) = max(dist(C,U,x)) = max([~200,...,~200]) = ~200
    # loss = abs(1 - loss(C,x)/loss(P,x)) = abs(~200/~100) = ~2
    # we expect MEE will move x towards A:
    # loss(P,x) = max(dist(P,x)) = max([~0,...,~0]) = ~0
    # loss(C,U,x) = max(dist(C,U,x)) = max([~400,...,~400]) = ~400
    # loss = abs(1 - loss(C,x)/loss(P,x)) = abs(~400/~0)

    # MEE params
    epochs, lr, ver = 300, 1.0, 1  # change ver to 1 to see training

    maximizing_x, maximal_error = MEE(A=A, C=C, U=U, x_source=x, f=one_point_center_loss, epochs=epochs, lr=lr,
                                      verbose=ver)

    print(var_to_string(A, 'A1', with_data=True))
    print(var_to_string(C, 'C', with_data=True))
    print(var_to_string(U, 'U', with_data=True))
    print(var_to_string(x, 'x', with_data=True))
    print(var_to_string(maximizing_x, 'maximizing_x', with_data=True))

    la = one_point_center_loss(A, x).item()
    print('loss(A,x) = max(dist(A,x))     = {}'.format(la))
    lc = one_point_center_loss(C, x, U).item()
    print('loss(C,U,x) = max(dist(C,U,x)) = {}'.format(lc))
    loss = abs(1 - lc / la)
    print('loss = |1- loss(C,U,x)/loss(A,x)| = {}'.format(loss))

    laMEE = one_point_center_loss(A, maximizing_x).item()
    print('loss(A,MEE(x)) = max(dist(A,x))     = {}'.format(laMEE))
    lcMEE = one_point_center_loss(C, maximizing_x, U).item()
    print('loss(C,U,MEE(x)) = max(dist(C,U,x)) = {}'.format(lcMEE))
    lossMEE = abs(1 - lcMEE / laMEE)
    print('loss = |1- loss(C,U,,MEE(x))/loss(A,,MEE(x))| = {}'.format(lossMEE))

    group1 = (A, 'g', 'A {}'.format(A.shape))
    group2 = (C, 'r', 'C {}'.format(C.shape))
    group3 = (x, 'blue', 'x {}'.format(x.shape))
    group4 = (maximizing_x, 'black', 'maximizing_x {}'.format(maximizing_x.shape))
    title = '1 point center\n'
    title += '|1- loss(C,U,x)/loss(A,x)| = {}\n'.format(loss)
    title += '|1- loss(C,U,,MEE(x))/loss(A,,MEE(x))| = {}'.format(lossMEE)
    plot_2d_scatter([group1, group2, group3, group4], title)

    print('Total run time {}'.format(get_time_str(g_start_time)))
    return


if __name__ == "__main__":
    mee_example()
