from Dependencies.UtilsScript import *
import torch
import sys
from sklearn import linear_model


def log_loss_sklearn_np(A: np.array, q: np.array, w: np.array = None) -> float:
    """
    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @return: logistic regression loss
    """
    loss_float = -1
    # noinspection PyBroadException
    try:
        X, y = a_to_x_y_np(A)
        n, d = X.shape
        assert y.shape == (n,) and q.shape == (d + 1,)
        # noinspection PyProtectedMember
        loss_float = linear_model.logistic._logistic_loss(q, X, y, alpha=1.0, sample_weight=w)
    except:  # noqa: E722
        print('Something went wrong. Error: {}'.format(sys.exc_info()[0]))
        print('|A| {} type(A)={}'.format(A.shape, type(A)))
        print('|q| {} type(q)={}'.format(q.shape, type(q)))
        if w is not None:
            print('|w| {} type(w)={}'.format(w.shape, type(w)))
    return loss_float


def log_loss_np_opt(A: np.array, q: np.array, w: np.array = None, lambda_: float = 1.0) -> float:
    """
    Alaa and Murad's version to support un even weights. looks identical to the original 1 they sent
    if w is None:
        return -1
    cost = (lambda A, w, q, lambda_: 1 / 2 * np.linalg.norm(q[:-1], 2) ** 2 + lambda_ * np.sum(
        np.multiply(w, np.log1p(np.exp(-np.multiply(A[:, -1], np.dot(A[:, :-1], q[:-1]) + q[-1]))))))
    loss = cost(A, w, q, lambda_)
    return loss

    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @param lambda_: regularization
    @return: logistic regression loss

    original code from murad:
    def loss(P, W, x, LAMBDA=1.0):
    return 1 / 2 * np.linalg.norm(x[:-1], 2) ** 2 + LAMBDA * np.sum(
        np.multiply(W, np.log1p(np.exp(-np.multiply(P[:, -1], np.dot(P[:, :-1], x[:-1]) + x[-1])))))

    all vars should be flattened (q, W)
    e.g.
    p1 = [1, 2]  # x,y of point 1
    p2 = [2, 2]  # x,y of point 2
    P = np.array([p1, p2]) # |P|=(2,2)
    x = np.array([1, 0])  # x is slope|bias. |x|=(2,)
    W = np.ones(P.shape[0])  # weights are 1. |W|=(2,)
    logistic_reg_loss_np_orig(P, x, W)  # 0.6450779389607822
    """
    if w is None:
        loss_float = 1 / 2 * np.linalg.norm(q[:-1], 2) ** 2 + lambda_ * np.sum(
            np.log1p(np.exp(-np.multiply(A[:, -1], np.dot(A[:, :-1], q[:-1]) + q[-1]))))
    else:
        loss_float = 1 / 2 * np.linalg.norm(q[:-1], 2) ** 2 + lambda_ * np.sum(
            np.multiply(w, np.log1p(np.exp(-np.multiply(A[:, -1], np.dot(A[:, :-1], q[:-1]) + q[-1])))))
    return loss_float


def log_loss_np(A: np.array, q: np.array, w: np.array = None, lambda_: float = 1.0) -> float:
    """
    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @param lambda_: regularization
    @return: logistic regression loss
    """
    X, y = a_to_x_y_np(A)
    n, d = X.shape
    a, b = q[:-1], q[-1]
    assert y.shape == (n,) and a.shape == (d,) and b.shape == ()
    slopes_norm = np.linalg.norm(a, 2)
    penalty = 1 / 2 * slopes_norm ** 2
    dot_coefs_slopes_plus_bias = np.dot(X, a) + b
    exponent = -np.multiply(y, dot_coefs_slopes_plus_bias)
    if w is not None:
        lan = np.multiply(w, np.log1p(np.exp(exponent)))
    else:
        lan = np.log1p(np.exp(exponent))
    loss = penalty + lambda_ * np.sum(lan)
    return loss


def log_loss(A: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None, lambda_: float = 1.0) -> torch.float32:
    """
    torch - supports grads
    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @param lambda_: regularization
    @return: logistic regression loss
    """
    # print(var_to_string(A, 'A', with_data=True))
    # print(var_to_string(q, 'q', with_data=True))
    # print(var_to_string(w, 'W', with_data=True))
    X, y = a_to_x_y(A)
    n, d = X.shape
    a, b = q[:-1], q[-1]
    assert y.shape == (n,) and a.shape == (d,) and b.shape == ()
    slopes_norm = torch.norm(a, 2)
    penalty = 1 / 2 * slopes_norm ** 2
    dot_coefs_slopes_plus_bias = torch.matmul(X, a) + b
    # print(var_to_string(dot_coefs_slopes_plus_bias, 'dot_coefs_slopes_plus_bias', with_data=True))
    # print(var_to_string(A[:, -1], 'A[:, -1]', with_data=True))
    # exponent = -np.multiply(A[:, -1].detach(), dot_coefs_slopes_plus_bias.detach())
    # print(var_to_string(exponent, 'exponent', with_data=True))
    exponent = -torch.mul(y, dot_coefs_slopes_plus_bias)
    # exponent = -torch.dot(A[:, -1], dot_coefs_slopes_plus_bias.t())
    # print(var_to_string(exponent, 'exponent', with_data=True))

    # lan = np.multiply(w, np.log1p(np.exp(exponent.detach())))
    # print(var_to_string(lan, 'lan', with_data=True))
    if w is not None:
        lan = torch.mul(w, torch.log1p(torch.exp(exponent)))
    else:
        lan = torch.log1p(torch.exp(exponent))
    # print(var_to_string(lan, 'lan', with_data=True))
    loss = penalty + lambda_ * torch.sum(lan)
    # exit(1)
    return loss


def log_loss_opt(A: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None, lambda_: float = 1.0) -> torch.float32:
    """
    torch - supports grads
    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @param lambda_: regularization
    @return: logistic regression loss
    """
    if w is None:
        loss = (1 / 2 * torch.norm(q[:-1], 2) ** 2) + lambda_ * torch.sum(
            torch.log1p(torch.exp(-torch.mul(A[:, -1], torch.matmul(A[:, :-1], q[:-1]) + q[-1]))))
    else:
        loss = (1 / 2 * torch.norm(q[:-1], 2) ** 2) + lambda_ * torch.sum(
            torch.mul(w, torch.log1p(torch.exp(-torch.mul(A[:, -1], torch.matmul(A[:, :-1], q[:-1]) + q[-1])))))
    return loss


def log_solver_sklearn(A: np.array, w: np.array = None, out: bool = False, to_torch: bool = True) -> np.array:
    """
    A: |A|=nx(d+1). data of shape X|y
    w: |w|=(n,). weights per sample
    out: prints results
    to_torch: output torch or numpy
    returns q: logistic regression coefficients and bias for A
    """
    try:
        if isinstance(A, torch.Tensor):
            A = torch_to_numpy(A)
        if isinstance(w, torch.Tensor):
            w = torch_to_numpy(w)

        if w is not None:
            if len(w.shape) > 1:
                w = w.flatten()
            if (w < 0).any():  # w can't be negative
                raise ValueError

        X, y = a_to_x_y_np(A)
        n, d = X.shape
        import warnings  # billions of convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # logr = linear_model.LogisticRegression(solver='liblinear', random_state=0, C=1.0)
            # by murad: newton-cg better for uneven weights
            logr = linear_model.LogisticRegression(tol=1e-7, solver='newton-cg', random_state=0, C=1.0, max_iter=1e8)
            logr.fit(X, y, sample_weight=w)
        a = logr.coef_.flatten()  # slopes
        b = logr.intercept_  # bias
        q_opt = np.concatenate((a, b), axis=0).astype('float64')

        assert q_opt.shape == (d + 1,)  # slope for each dim and 1 bias
        if out:
            print('log_solver_sklearn:')
            print(var_to_string(A, title='\tA', with_data=False))
            print(var_to_string(np.round(A[:2], 3), title='\tA[:2]', with_data=True))
            print(var_to_string(np.round(q_opt, 3), title='\tq_opt', with_data=True))
            if w is not None:
                print(var_to_string(w, title='\tw', with_data=False))
                print(var_to_string(np.round(w[:5], 3), title='\tw[:5]', with_data=True))
            l1 = log_loss_sklearn_np(A, q_opt, w)
            print('\tlog_loss_sklearn_np(A,q_opt)={:,.12f}'.format(l1))

            l2 = log_loss_np_opt(A, q_opt, w)
            print('\tlog_loss_np_opt(A,q_opt)    ={:,.12f}'.format(l2))

            l3 = log_loss_np(A, q_opt, w)
            print('\tlog_loss_np(A,q_opt)        ={:,.12f}'.format(l3))

            A_t = numpy_to_torch(A)
            q_opt_t = numpy_to_torch(q_opt)
            w_t = None if w is None else numpy_to_torch(w)

            l4 = log_loss_opt(A_t, q_opt_t, w_t)  # torch
            print('\tlog_loss_opt_torch(A,q_opt) ={:,.12f}'.format(l4))

            l5 = log_loss(A_t, q_opt_t, w_t)  # torch
            print('\tlog_loss_torch(A,q_opt)     ={:,.12f}'.format(l5))

        if to_torch:
            q_opt = numpy_to_torch(q_opt)
    except ValueError:
        print('Something went wrong (w_i negative, labels not 0 or 1, only 1 class, ...)')
        print(var_to_string(A, '\tA', with_data=True))
        print(var_to_string(w, '\tw', with_data=True))
        q_opt = None
    return q_opt


def lin_solver_sklearn(A: np.array, w: np.array = None, out: bool = False, to_torch: bool = True) -> np.array:
    """
    A: |A|=nx(d+1). data of shape X|y
    w: |w|=(n,). weights per sample
    out: prints results
    to_torch: output torch or numpy
    returns q: linear regression coefficients and bias for A
    """
    try:
        if isinstance(A, torch.Tensor):
            A = torch_to_numpy(A)
        if isinstance(w, torch.Tensor):
            w = torch_to_numpy(w)

        if w is not None:
            if len(w.shape) > 1:
                w = w.flatten()
            if (w < 0).any():  # w can't be negative
                raise ValueError

        X, y = a_to_x_y_np(A)
        n, d = X.shape
        linr = linear_model.LinearRegression()
        linr.fit(X, y, sample_weight=w)
        a = linr.coef_.flatten()  # slopes
        b = np.array([linr.intercept_])  # bias
        q_opt = np.concatenate((a, b), axis=0)

        assert q_opt.shape == (d + 1,)  # slope for each dim and 1 bias
        if out:
            print('lin_solver_sklearn:')
            print(var_to_string(A, title='\tA', with_data=False))
            print(var_to_string(np.round(A[:2], 3), title='\tA[:2]', with_data=True))
            print(var_to_string(np.round(q_opt, 3), title='\tq_opt', with_data=True))
            if w is not None:
                print(var_to_string(w, title='\tw', with_data=False))
                print(var_to_string(np.round(w[:5], 3), title='\tw[:5]', with_data=True))

            l1 = lin_loss_np(A, q_opt, w=None)
            print('\tlin_loss_np(A,q_opt)  ={:,.12f}'.format(l1))

            l2 = lin_loss_np(A, q_opt, w=w)
            print('\tlin_loss_np(A,q_opt,w)={:,.12f}'.format(l2))

            l3 = log_loss_np(A, q_opt, w=None)
            print('\tlog_loss_np(A,q_opt)  ={:,.12f}'.format(l3))

        if to_torch:
            q_opt = numpy_to_torch(q_opt)
    except ValueError:
        # print('w has negative entry {}'.format(var_to_string(w, 'w', with_data=True)))
        q_opt = None
    return q_opt


def lin_loss_np(A: np.array, q: np.array, w: np.array = None):
    """
    @param A: |A|=nx(d+1). data of shape X|y
    @param q: |q|=(d+1,). coefficients and bias
    @param w: |w|=(n,). weights per sample (optional)
    @return: linear regression loss:= the Euclidean 2-norm || y - (aX + b) ||^2
    doing:
        y_hat = X @ a + b  # |yhat|=(n,)
        diff = y - y_hat  # |diff|=(n,)
        diff_squared = diff^2  # (n,). element wise ^2
        if w is not None:
            diff_squared = w * diff_squared  # (n,). weights added
        loss = sum(diff_squared)  # scalar
    """
    X, y = a_to_x_y_np(A)
    n, d = X.shape

    a, b = np.expand_dims(q[:-1], axis=1), q[-1]  # |a|=(d,) -> (d,1)

    assert X.shape[1] == d and y.shape == (n,) and a.shape == (d, 1) and b.shape == ()
    y_hat = (X @ a + b).flatten()
    # print(var_to_string(y_hat, 'yhat'))
    error = (y_hat - y)
    # print(var_to_string(error, 'e'))
    error = error ** 2
    # print(var_to_string(error, 'e'))
    if w is not None:
        error = w * error
        # print(var_to_string(error, 'e'))
    loss = np.sum(error)
    return loss


def a_to_x_y_np(A: np.array) -> (np.array, np.array):
    X, y = A[:, :-1], A[:, -1]
    return X, y


def x_y_to_a_np(X: np.array, y: np.array) -> np.array:
    A = np.concatenate((X, y.expand_dims(axis=0)), axis=0)
    return A


def a_to_x_y(A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    X, y = A[:, :-1], A[:, -1]
    return X, y


def x_y_to_a(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    A = torch.cat((X, y.view(y.shape[0], 1)), 1)
    return A


def sanity_check(A: torch.Tensor, f_test: bool = False, mes_f: bool = False, solvers: bool = False):
    print('\nLogisticReg.sanity_check:')
    # A = [[0.03, -0.18, -0.16, -0.29, -0.36, -0.47, 0.27, -0.05, -1.0],
    #      [1.02, -0.01, -0.41, -0.36, 1.59, 2.64, -1.68, -1.0, -1.0],
    #      [0.6, 0.22, -0.29, -0.27, -0.38, -0.79, 1.26, 1.47, -1.0],
    #      [0.19, 0.46, -0.34, -0.27, -0.38, -0.55, 0.86, 0.55, -1.0],
    #      [-0.91, 0.0, -0.0, -0.25, -0.33, -0.52, 0.0, -0.18, -1.0],
    #      [0.66, 1.23, -0.38, -0.39, -0.2, 0.25, -0.76, -0.76, -1.0],
    #      [0.56, 0.4, -0.48, -0.31, -0.34, -0.6, 0.13, 0.01, -1.0],
    #      [-0.08, -1.24, 0.21, 0.1, -0.37, -0.67, 0.76, 0.64, -1.0],
    #      [0.14, -0.58, 0.01, -0.27, -0.36, -0.62, 0.39, 0.21, -1.0],
    #      [-0.94, -1.37, 0.06, 0.03, -0.34, -0.47, 0.04, -0.18, -1.0],
    #      [0.73, 0.61, -0.23, -0.29, -0.23, 0.22, -0.59, -0.69, -1.0],
    #      [-0.22, 0.38, -0.09, -0.29, -0.34, -0.61, 0.07, -0.02, -1.0],
    #      [-0.49, -1.02, 0.07, -0.07, -0.28, -0.01, -0.24, -0.47, -1.0],
    #      [0.12, 1.23, -0.38, -0.35, -0.33, -0.59, 0.08, -0.02, -1.0],
    #      [-2.41, -0.53, 2.27, 1.36, 0.91, 1.99, -1.49, -0.97, 1.0],
    #      [-0.64, -1.05, 0.02, -0.07, -0.36, -0.64, 0.66, 0.5, -1.0],
    #      [-0.53, -0.24, -0.2, -0.27, -0.36, -0.61, 0.74, 0.57, -1.0],
    #      [0.06, 0.96, 0.17, -0.28, 1.89, 3.2, -1.68, -1.0, -1.0],
    #      [0.63, -0.17, -0.34, -0.24, -0.21, 0.39, -0.61, -0.7, 1.0],
    #      [-0.02, -0.11, -0.1, -0.18, -0.4, -0.77, 1.73, 1.76, -1.0],
    #      [-0.63, 0.46, 0.14, -0.17, -0.33, -0.39, -0.07, -0.3, -1.0],
    #      [1.2, 0.57, -0.22, -0.34, -0.33, -0.52, -0.02, -0.18, -1.0],
    #      [0.64, 1.59, -0.45, -0.33, -0.41, -0.88, 2.77, 3.87, -1.0],
    #      [0.08, -0.6, -0.2, -0.23, -0.34, -0.42, 0.07, -0.18, -1.0],
    #      [0.31, 0.35, -0.21, -0.29, -0.39, -0.56, 1.15, 0.79, -1.0],
    #      [-1.93, -1.49, 1.68, 1.13, -0.24, 0.17, -0.5, -0.63, 1.0],
    #      [0.21, 2.23, -0.39, -0.4, -0.33, -0.34, 0.07, -0.21, -1.0],
    #      [-0.59, 1.38, 0.43, -0.22, 5.42, 2.43, -2.1, -0.98, 1.0],
    #      [-0.81, -1.5, 0.07, 0.07, -0.35, -0.5, 0.3, 0.03, -1.0],
    #      [0.48, -0.47, -0.25, -0.21, -0.28, -0.12, -0.46, -0.59, -1.0],
    #      [0.06, -0.46, -0.11, -0.22, -0.36, -0.61, 0.44, 0.21, -1.0],
    #      [-0.14, 0.14, -0.22, -0.27, -0.32, -0.37, -0.07, -0.28, -1.0],
    #      [1.09, 1.96, -0.68, -0.4, 0.5, 2.04, -1.37, -0.96, -1.0],
    #      [-1.14, -0.72, 0.17, -0.06, -0.25, 0.1, -0.57, -0.67, -1.0],
    #      [-1.16, -0.96, 0.19, -0.06, -0.36, -0.59, 0.29, 0.09, -1.0],
    #      [0.17, 0.88, -0.16, -0.26, -0.27, 0.09, -0.43, -0.61, -1.0],
    #      [-0.03, 0.83, -0.35, -0.32, -0.35, -0.65, 0.23, 0.16, -1.0],
    #      [0.86, 0.5, -0.5, -0.31, -0.24, 0.18, -0.54, -0.66, -1.0],
    #      [-0.23, -0.06, -0.22, -0.24, -0.33, -0.35, 0.31, -0.0, -1.0],
    #      [0.45, 1.79, -0.44, -0.4, -0.2, 0.3, -0.69, -0.72, -1.0],
    #      [0.25, 0.12, -0.19, -0.24, -0.35, -0.67, 0.23, 0.21, -1.0],
    #      [-0.23, -0.75, -0.14, -0.16, -0.37, -0.51, 0.52, 0.15, -1.0],
    #      [-0.17, 0.82, -0.3, -0.37, -0.34, -0.62, 0.09, 0.01, -1.0],
    #      [0.78, 1.13, -0.52, -0.32, -0.3, -0.16, -0.3, -0.49, -1.0],
    #      [1.11, 1.04, -0.48, -0.29, -0.37, -0.58, 0.49, 0.2, -1.0],
    #      [-3.13, -2.43, 4.51, 5.01, 0.39, 1.26, -1.28, -0.92, 1.0],
    #      [0.55, -0.39, -0.42, -0.23, -0.33, -0.49, -0.05, -0.22, -1.0],
    #      [-0.08, -0.03, -0.19, -0.26, -0.3, -0.08, -0.22, -0.46, -1.0],
    #      [-0.17, -0.79, -0.01, -0.15, -0.35, -0.47, 0.05, -0.21, -1.0],
    #      [0.61, 0.7, -0.58, -0.33, -0.41, -0.85, 2.66, 3.29, -1.0]]
    # A_np = torch_to_numpy(A)
    # qc_np = np.array([2.89113712310791, -1.4303019046783447, 0.3062683939933777,
    # 9.088421821594238, 2.6698670387268066,
    #                   -2.7429726123809814, 2.293642520904541, -12.69900131225586, -6.971704006195068])
    # print(var_to_string(A_np, 'A_np'))
    # print(var_to_string(qc_np, 'qc_np', with_data=True))
    # l1 = log_loss_sklearn_np(A_np, qc_np)
    # print('\tlog_loss_sklearn_np(A,qc)={:,.12f}'.format(l1))
    #
    # l3 = log_loss_np(A_np, qc_np)
    # print('\tlog_loss_np(A,qc)        ={:,.12f}'.format(l3))
    # exit(222222)
    # A = torch.Tensor(A)
    # W = torch.zeros(A.shape[0]) + 50
    # print(var_to_string(A, 'A', with_data=True))
    # print(var_to_string(W, 'W', with_data=True))
    # qa = log_solver_sklearn(A, W)
    # print(var_to_string(qa, 'qa', with_data=True))
    # qa = log_solver_sklearn(A, W)
    # print(var_to_string(qa, 'qa', with_data=True))
    # qa = log_solver_sklearn(A, W)
    # print(var_to_string(qa, 'qa', with_data=True))
    # exit(33)

    if mes_f:
        A_np = torch_to_numpy(A)
        q_opt = log_solver_sklearn(A)
        q_opt_np = torch_to_numpy(q_opt)
        measure_F(
            F=[
                (log_loss_sklearn_np, A_np, q_opt_np),
                (log_loss_np_opt, A_np, q_opt_np),
                (log_loss_np, A_np, q_opt_np),
                (log_loss_opt, A, q_opt),
                (log_loss, A, q_opt)
            ],
            rounds=10)
    if f_test:
        test_loss_function_and_solver()
    if solvers:
        A_np = torch_to_numpy(A)
        _ = log_solver_sklearn(A_np, w=None, out=True, to_torch=False)
        n = A_np.shape[0]
        w = np.ones(n)
        w[int(n / 3):2 * int(n / 3)] = 2
        w[2 * int(n / 3):] = 3
        _ = log_solver_sklearn(A_np, w=w, out=True, to_torch=False)
        _ = lin_solver_sklearn(A_np, w=w, out=True, to_torch=False)
    return


def measure_F(F: list, rounds: int = 10000):
    print('measure_F:')
    for (f_i, data, q) in F:
        print('\tf_i={}'.format(f_i))
        t = get_current_time()
        for i in range(rounds):
            f_i(data, q)
            if i == 0:
                print('\t\tf_i(A,q)={}'.format(f_i(data, q)))
        print('\t\t{}'.format(get_mili_seconds_str(t)))
    return


def test_loss_function_and_solver():
    eps = 0.00001
    print('test_loss_function_and_solver:')
    print('LOSS and SOLVER CHECK:')
    A_np = np.zeros(shape=(4, 2))
    A_np[0] = [0, -1]
    A_np[1] = [1, -1]
    A_np[2] = [3, 1]
    A_np[3] = [4, 1]
    print(var_to_string(A_np, 'A', with_data=True))
    lin_q_opt = lin_solver_sklearn(A_np, to_torch=False)
    print(var_to_string(lin_q_opt, 'lin_q_opt', with_data=True))
    lin_loss = lin_loss_np(A_np, lin_q_opt)
    print('\tlin_loss_np(A,q_opt)={:,.12f}'.format(lin_loss))
    lin_loss2 = log_loss_np(A_np, lin_q_opt)
    print('\tlog_loss_np(A,q_opt)={:,.12f}'.format(lin_loss2))

    log_q_opt = log_solver_sklearn(A_np, to_torch=False)
    print(var_to_string(log_q_opt, 'log_q_opt', with_data=True))

    l1 = log_loss_sklearn_np(A_np, log_q_opt)
    print('\tlog_loss_sklearn_np(A,q_opt)={:,.12f}'.format(l1))

    l2 = log_loss_np_opt(A_np, log_q_opt)
    print('\tlog_loss_np_opt(A,q_opt)    ={:,.12f}'.format(l2))

    l3 = log_loss_np(A_np, log_q_opt)
    print('\tlog_loss_np(A,q_opt)        ={:,.12f}'.format(l3))

    A_t = numpy_to_torch(A_np)
    log_q_opt_t = numpy_to_torch(log_q_opt)

    l4 = log_loss_opt(A_t, log_q_opt_t)  # torch
    print('\tlog_loss_opt_torch(A,q_opt) ={:,.12f}'.format(l4))

    l5 = log_loss(A_t, log_q_opt_t)  # torch
    print('\tlog_loss_torch(A,q_opt)     ={:,.12f}'.format(l5))

    log_losses = [l1, l2, l3, l4.item(), l5.item()]
    for i in range(len(log_losses)):
        for j in range(i + 1, len(log_losses)):
            assert abs(log_losses[i] - log_losses[j]) < eps, 'diff({},{}) > {}'.format(i, j, eps)

    plot_2d_regression(A_np, log_q_opt, lin_q_opt, l1, lin_loss2)
    return


def get_extra_testQ(A: torch.Tensor) -> list:
    extraQ_tuples = []
    subQ = get_sub_opt(A, [2], 20, show_infoQ=True)
    extraQ_tuples.append((subQ, 'subQ', 'sub optimal slice = 2'))

    normQ1 = build_lines_around_data(A, dist='normal', size=20)
    info_Q(normQ1, A, 'normQ1')
    extraQ_tuples.append((normQ1, 'normQ1', 'build_lines_around_data normal'))

    uniQ1 = (build_lines_around_data(A, dist='uniform', size=20))
    info_Q(uniQ1, A, 'uniQ1')
    extraQ_tuples.append((uniQ1, 'uniQ1', 'build_lines_around_data uniform'))

    normQ2 = (torch_normal((10, A.shape[1], 1), 0, 100))
    info_Q(normQ2, A, 'normQ2')
    extraQ_tuples.append((normQ2, 'normQ2', 'torch_normal miu=0, std=100'))

    uniQ2 = (torch_uniform((10, A.shape[1], 1), -1000, 1000))
    info_Q(uniQ2, A, 'uniQ2')
    extraQ_tuples.append((uniQ2, 'uniQ2', 'torch_uniform low -1000 high 1000'))
    return extraQ_tuples


def get_Q_sets(A: torch.tensor, path_Q: str, param_Q: tuple, trainQ_s: int, valQ_s: int, testQ_s: int) -> (
        torch.tensor, torch.tensor, torch.tensor):
    max_size = trainQ_s + valQ_s + testQ_s
    Q = build_Q(A, path_Q, param_Q, max_size=max_size)
    Q = shuffle_tensor(Q)
    trainQ = Q[:trainQ_s]
    valQ = Q[trainQ_s:valQ_s + trainQ_s]
    testQ = Q[valQ_s + trainQ_s:]

    info_Q(trainQ, A, 'trainQ')
    info_Q(valQ, A, 'valQ')
    info_Q(testQ, A, 'testQ')

    return trainQ, valQ, testQ


def build_Q(A: torch.tensor, path_Q: str, param_Q: tuple, max_size: int, title: str = 'Q') -> (
        torch.tensor, torch.tensor, torch.tensor):
    # about middle of the training, you need to get to q_opt
    print('\nGet Q(max_size={}):'.format(max_size))
    if os.path.exists(path_Q):
        Q = load_tensor(path_Q)
    else:
        initQ_s, epochs, bs, lr = param_Q
        means, stds = get_normal_dist_by_dim(torch_to_numpy(A))
        initQ = np_normal(shape=(initQ_s, A.shape[1]), mius=means, stds=stds)
        initQ = numpy_to_torch(initQ)
        info_Q(initQ, A, 'initQ')

        Q = torch.empty((0, A.shape[1]), dtype=torch.double)
        for i, q in enumerate(initQ):
            print('{}/{}'.format(i + 1, initQ.shape[0]))
            Q_i = build_1_Q(A, q, epochs=epochs, bs=bs, lr=lr, progress=0.1)
            Q = torch.cat((Q, Q_i), 0)
        save_tensor(Q, path_Q)

    print(var_to_string(Q, '\tQ_all', with_data=False))
    Q = shuffle_tensor(Q)
    Q = Q[:max_size]
    Q = add_cuda(Q)
    if title is not None:
        info_Q(Q, A, title)
    return Q


def sample_from_list(tensor_list: list, n: int) -> list:
    perm = torch.randperm(len(tensor_list))
    random_indices = perm[:n]
    # print(random_indices)
    sub_set_list = [tensor_list[i] for i in random_indices]
    return sub_set_list


def build_1_Q(A: torch.tensor, q: torch.tensor, epochs: int, bs: int, lr: float,
              es_ratio: float = 0.001, progress: float = 0.1, sample: float = 0.1) -> torch.Tensor:
    """
    overfit to breast cancer data:
    :param sample:
    :param progress:
    :param es_ratio:
    :param A: X|y where |A|=nx(d+1). row i:= Xi|yi where |Xi| = 1xd, |yi| = 1x1
    :param q: initialing q
    :param epochs:
    :param bs:
    :param lr: base lr
    moving lr: each 'pat' epochs, lr*='fac' and min lr is 'min_lr'
    function trying to learn the optimal q (see loss function)
    :return: Q:= some of the states q passed on the way towards q opt.
             |Q|=0.1 *(#batches) * epochs
             Q is a list of tensors (q states)
    """
    print('build_Q:')
    n, d = A.shape[0], A.shape[1] - 1
    total_batches = int(n / bs) + 1
    print('\tepochs {}, bs {}, #batches {}, base lr {}'.format(epochs, bs, total_batches, lr))
    qs_from_each_epoch = int(sample * total_batches) + 1  # save 10% qs from each epoch
    msg = '\tIn each epoch({} total) there are {} batches(different qs). sample 10%({}) from them. expected |Q|={}'
    print(msg.format(epochs, total_batches, qs_from_each_epoch, qs_from_each_epoch * epochs))
    print('\tearly stop if |1-loss_q/loss_q_opt|< {}'.format(es_ratio))
    # TEST optimal loss and q init loss
    opt_loss = log_loss(A, log_solver_sklearn(A))
    opt_loss_avg = opt_loss / n
    print('\tOpt loss {:,.3f}. avg={:,.3f}'.format(opt_loss, opt_loss_avg))
    q_init_loss = log_loss_opt(A, q)
    print('\tOur loss {:,.3f}. avg={:,.3f}'.format(q_init_loss, q_init_loss / n))
    q.requires_grad = True

    print('Training...')
    optimizer = torch.optim.Adam([q], lr=lr)  # q is trainable
    Q = []
    PROGRESS_PRINT = max(int(progress * epochs), 1)  # print progress each PROGRESS_PRINT epochs done

    best_avg_loss = float("inf")
    best_diff = -1
    best_e = -1
    for epoch in range(1, epochs + 1):
        Q_epoch = []
        A = shuffle_tensor(A)

        for i in range(0, n, bs):
            batch_A = A[i:i + bs]
            loss = log_loss_opt(batch_A, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Q_epoch.append(q.detach().clone())  # each batch adds 1 q state - sample 10% from it later
            # print('\tA[{}:{}] avg loss:{:,.3f}'.format(i, min(i + bs, n), loss.item() / batch_A.shape[0]))

        Q += sample_from_list(Q_epoch, qs_from_each_epoch)
        real_avg_loss = log_loss_opt(A, q) / n
        diff = torch.abs(1 - real_avg_loss / opt_loss_avg).item()

        if real_avg_loss < best_avg_loss:
            best_avg_loss = real_avg_loss
            best_e = epoch
            best_diff = diff

        if epoch % PROGRESS_PRINT == 0 or (diff < es_ratio):
            progress_msg = '\tepoch [{}/{}] real avg loss:{:,.8f},diff={:,.8f} lr={}, |Q|={}'
            print(progress_msg.format(epoch, epochs, real_avg_loss, diff, get_lr(optimizer), len(Q)))
            if diff < es_ratio:
                break
        # optimizer.update_lr()

    print('Done training. Results:')
    print('\tOpt avg loss {:,.3f}'.format(opt_loss_avg))
    print('\tOur avg loss {:,.3f} (epoch {})'.format(best_avg_loss, best_e))
    print('\tbest_diff={:,.6f}'.format(best_diff))

    # convert to list of tensors to 3d tensor
    Q_t = torch.empty((len(Q), d + 1), dtype=torch.double)
    for i, q_ in enumerate(Q):
        q_.requires_grad = False
        Q_t[i] = q_
    # info_Q(Q_t, A, 'Q_t', False)
    return Q_t


def info_Q(Q: torch.tensor, A: torch.tensor, title: str):
    loss_A_on_q_opt = log_loss_opt(A, log_solver_sklearn(A)).item()
    losses, diffs = [], []
    for q in Q:
        loss_A_on_q = log_loss_opt(A, q).item()
        diff = abs(1 - loss_A_on_q / loss_A_on_q_opt)
        losses.append(loss_A_on_q)
        diffs.append(diff)

    print('\t|{}|={}:'.format(title, t_size_str(Q)))
    spaces = -5 + len(title) + 4  # -5 for 'q_opt', add len title, 4 for 'avg '
    print('\t\tloss(A,q_opt){}={:20,.2f}'.format(' ' * spaces, loss_A_on_q_opt))
    print('\t\tavg loss(A,{})={:20,.2f}'.format(title, np.mean(losses)))
    print('\t\tavg |1- loss(A,q)/loss(A,q_opt)|={:,.3f} with std {:,.3f}'.format(np.mean(diffs), np.std(diffs)))

    return


def get_sub_opt(A: torch.tensor, slices_size: list, size_per_size: int, title: str = 'subQ',
                show_infoQ: bool = False) -> torch.tensor:
    n, d = A.shape[0], A.shape[1] - 1
    total_size = size_per_size * len(slices_size)
    Q_sub = torch.empty((total_size, d + 1, 1))

    for i, slice_size in enumerate(slices_size):
        for j in range(size_per_size):
            perm = torch.randperm(n)
            random_indices = perm[:slice_size]
            B = A[random_indices]
            sub_opt_q = log_solver_sklearn(B)
            # print(i * size_per_size + j)
            Q_sub[i * size_per_size + j] = sub_opt_q
        # print(slice_size, f(A, Q_sub[i * size_per_size + j]))

    if show_infoQ:
        info_Q(Q_sub, A, title)
    return Q_sub


def build_lines_around_data(A: torch.Tensor, dist: str = 'normal', size: int = 200):
    A_np = torch_to_numpy(A) if isinstance(A, torch.Tensor) else A
    d = A.shape[1]

    if dist == 'normal':
        noise = 20
        means, stds = get_normal_dist_by_dim(A_np)
        # stds *= 1  # add more noise
        # print('A({}): means {}, stds: {}'.format(A_np.shape, list_1d_to_str(means.tolist()),
        #                                          list_1d_to_str(stds.tolist())))
        A_tag_np = np_normal(shape=(500, d), mius=means, stds=stds * noise)  # create new points same dist (add noise)
        # means, stds = get_distribution_by_dim(A_tag_np)  # verify new data dist
        # print('A\'({}): means {}, stds: {}'.format(A_tag_np.shape, list_1d_to_str(means.tolist()),
        #                                            list_1d_to_str(stds.tolist())))
    else:  # uniform
        noise = 10
        lows, highs = get_uniform_dist_by_dim(A_np)
        # lows *= 1  # add more noise
        # highs *= 1  # add more noise
        A_tag_np = np_uniform(shape=(500, d), lows=lows * noise, highs=highs * noise)

    # if d == 2 and plot_if_2d:
    #     group1 = (A_np, A_COLOR_1, 'origin data {}'.format(A.shape))
    #     group2 = (A_tag_np, A_COLOR_2, 'new data {}'.format(A_tag_np.shape))
    #     plot_2d_scatter([group1, group2], title='creation of new points({})'.format(dist))
    Q = get_sub_opt(numpy_to_torch(A_tag_np), [2], size)  # select 2 points and get their line
    return Q
