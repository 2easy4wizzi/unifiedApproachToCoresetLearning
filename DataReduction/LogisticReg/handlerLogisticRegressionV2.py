from Dependencies.UtilsScript import *
import torch
import math
from sklearn import linear_model


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
    # print('penalty {}'.format(penalty.item()))
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
    # print(wu.tt.to_str(lan, 'lan', fp=8))
    # print(wu.tt.to_str(lambda_ * torch.sum(lan), 'lambda_ * torch.sum(lan)'))
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


def get_Q_sets(A: torch.tensor, cfgQ: dict) -> (torch.tensor, torch.tensor, torch.tensor):
    start_time = get_current_time()
    print('get_Q_sets')
    if os.path.exists(cfgQ['path']):
        allQ = load_tensor(cfgQ['path'])
        info_Q(allQ, A, 'allQ', with_error=False)
    else:
        allQ = build_Q(A, cfgQ)
        info_Q(allQ, A, 'allQ', with_error=True)  # once for the log
        exit(99)  # TODO build Q and exit - rerun for experiments

    allQ = shuffle_tensor(allQ)

    max_size = cfgQ['trainQ_s'] + cfgQ['valQ_s'] + cfgQ['testQ_s']
    Q = allQ[:max_size]
    Q = add_cuda(Q)
    split1 = cfgQ['trainQ_s']
    split2 = cfgQ['valQ_s'] + cfgQ['trainQ_s']

    trainQ = Q[:split1]
    valQ = Q[split1:split2]
    testQ = Q[split2:]

    info_Q(trainQ, A, 'trainQ', with_error=cfgQ['infoQ_show_er'])
    info_Q(valQ, A, 'valQ', with_error=cfgQ['infoQ_show_er'])
    info_Q(testQ, A, 'testQ', with_error=cfgQ['infoQ_show_er'])
    print('get_Q_sets time {}'.format(get_time_str(start_time)))
    return trainQ, valQ, testQ


def build_Q(A: torch.tensor, cfgQ: dict) -> (torch.tensor, torch.tensor, torch.tensor):
    print('build_Q:')
    epochs, bs, lr = cfgQ['epochs'], cfgQ['bs'], cfgQ['lr']
    initQ = torch_normal(shape=(cfgQ['init_size'], A.shape[1]), miu=cfgQ['init_miu'], std=cfgQ['init_std'],
                         to_double=True)
    info_Q(initQ, A, 'initQ', with_error=cfgQ['infoQ_show_er'])
    Q = torch.empty((0, A.shape[1]))

    for i, q in enumerate(initQ):
        print('{}/{}'.format(i + 1, initQ.shape[0]))
        Q_i = build_1_Q(A, q, epochs=epochs, bs=bs, lr=lr, mv=cfgQ['mv'], progress=0.1, sample=cfgQ['sample_step'])
        Q = torch.cat((Q, Q_i), 0)

    save_tensor(Q, cfgQ['path'])
    return Q


def sample_from_list(tensor_list: list, n: int) -> list:
    perm = torch.randperm(len(tensor_list))
    random_indices = perm[:n]
    # print(random_indices)
    sub_set_list = [tensor_list[i] for i in random_indices]
    return sub_set_list


def build_1_Q(A: torch.tensor, q: torch.tensor, epochs: int, bs: int, lr: float, mv: dict,
              es_ratio: float = 0.001, progress: float = 0.1, sample: float = 0.1) -> torch.Tensor:
    """
    :param sample:
    :param progress:
    :param es_ratio:
    :param A: X|y where |A|=nx(d+1). row i:= Xi|yi where |Xi| = 1xd, |yi| = 1x1
    :param q: initialing q
    :param epochs:
    :param bs:
    :param lr: base lr
    :param mv: moving lr params
    function trying to learn the optimal q (see loss function)
    :return: Q:= some of the states q passed on the way towards q opt.
             |Q|=0.1 *(#batches) * epochs
             Q is a list of tensors (q states)
    """
    print('build_Q:')
    n, d = A.shape[0], A.shape[1] - 1
    total_batches = math.ceil(n / bs)
    print('\tepochs {}, bs {}, #batches {}, base lr {}'.format(epochs, bs, total_batches, lr))
    qs_from_each_epoch = math.ceil(sample * total_batches)  # save X% qs from each epoch
    msg = '\tIn each epoch({} total) there are {} batches(different qs). sample {}%({}) from them. expected |Q|={}'
    print(msg.format(epochs, total_batches, sample * 100, qs_from_each_epoch, qs_from_each_epoch * epochs))
    print('\tearly stop if |1-loss_q/loss_q_opt|< {}'.format(es_ratio))
    # TEST optimal loss and q init loss
    opt_loss = log_loss(A, log_solver_sklearn(A))
    opt_loss_avg = opt_loss / n
    print(wu.tt.to_str(q, '\tq'))
    print('\tOpt loss {:,.3f}. avg={:,.3f}'.format(opt_loss, opt_loss_avg))
    q_init_loss = log_loss_opt(A, q)
    print('\tOur loss {:,.3f}. avg={:,.3f}'.format(q_init_loss, q_init_loss / n))
    q.requires_grad = True

    print('Training...')
    optimizer = torch.optim.Adam([q], lr=lr)  # q is trainable
    optimizer = OptimizerHandler(optimizer, mv['factor'], mv['patience'], mv['min_lr'])
    Q = []
    PROGRESS_PRINT = max(int(progress * epochs), 1)  # print progress each PROGRESS_PRINT epochs done
    progress_msg = '\tepoch [{}/{}]: avg loss:{:,.3f}, diff={:,.6f}, lr={:,.5f}, |Q|={}'
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
            print(progress_msg.format(epoch, epochs, real_avg_loss, diff, optimizer.lr(), len(Q)))
            if diff < es_ratio:
                break
        optimizer.update_lr()

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


def info_Q(Q: torch.tensor, A: torch.tensor, title: str, with_error: bool):
    print('\t|{}|={}:'.format(title, t_size_str(Q)))
    if with_error:
        n = A.shape[0]
        q_opt = log_solver_sklearn(A)
        print(wu.tt.to_str(q_opt, '\tq_opt'))
        loss_A_on_q_opt = log_loss_opt(A, q_opt).item()
        losses, losses_avg, diffs = [], [], []
        for q in Q:
            loss_A_on_q = log_loss_opt(A, q).item()
            diff = abs(1 - loss_A_on_q / loss_A_on_q_opt)
            losses.append(loss_A_on_q)
            losses_avg.append(loss_A_on_q / n)
            diffs.append(diff)

        msg = '\t\tloss(A,q_opt)      ={:20,.2f}(avg={:,.2f})'
        print(msg.format(loss_A_on_q_opt, loss_A_on_q_opt / n))
        msg = '\t\tavg loss(A,{:7})={:20,.2f}(avg={:,.2f},std={:,.2f},min={:,.2f}, max={:,.2f})'
        print(msg.format(title, np.mean(losses), np.mean(losses_avg), np.std(losses_avg), min(losses_avg),
                         max(losses_avg)))
        msg = '\t\tavg |1- loss(A,q)/loss(A,q_opt)|={:,.3f}(std={:,.2f},min={:,.2f}, max={:,.2f}))'
        print(msg.format(np.mean(diffs), np.std(diffs), min(diffs), max(diffs)))
    return


def test_weighted_loss():
    from Dependencies import GetDataScript
    A: torch.Tensor = GetDataScript.get_data_A_by_name(ds_name='HTRU_2', data_root='../../Datasets')
    A_np = torch_to_numpy(A)
    ones_indices = np.where(A_np[:, -1] == 1.0)
    print(wu.tt.to_str(ones_indices, 'ones'))
    q_opt = log_solver_sklearn(A, w=None, out=False, to_torch=True)
    print(wu.tt.to_str(A, 'A'))
    loss = log_loss(A, q=q_opt, w=None)
    print('{} log_loss(A)'.format(loss))

    samples = torch.zeros(size=(3, A.shape[1])).double()
    samples[0] = A[18]
    samples[1] = A[41]
    samples[2] = A[60]
    losses_truth = torch.tensor([24.743452303627702, 29.77764796890446, 24.650602294952744])
    print(wu.tt.to_str(samples, 'samples', chars=-1))

    eps = 0.00001
    for samp, loss_truth in zip(samples, losses_truth):
        print('*' * 10)
        S = samp.unsqueeze(0)
        print(wu.tt.to_str(S, 'S', chars=-1))
        loss = log_loss(S, q=q_opt, w=None)
        cond = abs(loss - loss_truth) < eps
        msg = '{} log_loss. expected?{}'.format(loss, cond)
        print(wu.add_color(msg, ops='g' if cond else 'r'))

    print('*' * 20)
    loss = log_loss(samples, q=q_opt, w=None)
    penalty_truth = 24.650082238753665
    lan_truth = losses_truth - penalty_truth
    cond = abs(loss - (penalty_truth + torch.sum(lan_truth))) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 0.0, 0.0]).double()
    loss = log_loss(samples, q=q_opt, w=weights)
    cond = abs(loss - penalty_truth) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 1.0, 0.0]).double()
    loss = log_loss(samples, q=q_opt, w=weights)
    cond = abs(loss - losses_truth[1]) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 3.0, 0.0]).double()
    loss = log_loss(samples, q=q_opt, w=weights)
    loss_e = penalty_truth + 3 * lan_truth[1]
    cond = abs(loss - loss_e) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 3.0, 2.0]).double()
    loss = log_loss(samples, q=q_opt, w=weights)
    loss_e = penalty_truth + 3 * lan_truth[1] + 2 * lan_truth[2]
    cond = abs(loss - loss_e) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    return


def main():
    test_weighted_loss()
    return


if __name__ == "__main__":
    wu.main_wrapper(
        main_function=main,
        seed=42,
        ipv4=False,
        cuda_off=True,  # change to False for GPU
        torch_v=True,
        tf_v=False,
        cv2_v=False,
        with_pip_list=False,
        with_profiler=False
    )
