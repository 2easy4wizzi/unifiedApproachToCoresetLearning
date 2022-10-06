from Dependencies.UtilsScript import *
import torch
import math
from sklearn import linear_model


def f(A: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.float32:
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
    X, y = de_augment_torch(A)
    n, d = X.shape

    a, b = q[:-1], q[-1]
    assert y.shape == (n, 1) and a.shape == (d, 1) and b.shape[0] == 1
    y_hat = X @ a + b
    error = (y_hat - y).view(n)  # change size from nx1 to n
    error = error ** 2
    # print(error.tolist())
    if w is not None:
        error = w * error
        # print(error.tolist())
    loss = error.sum()
    # print(loss.item())
    return loss


def f_opt(A: torch.Tensor, q: torch.Tensor, w: torch.Tensor = None) -> torch.float32:
    """
    loss_function for linear_regression
    see function f. this is the same just faster
    """
    # print(var_to_string(A, 'A'))
    # print(var_to_string(q, 'q'))
    # if w is not None:
    #     print(var_to_string(w, 'w'))
    if len(A.shape) == 1:  # if one point
        A = A.view(1, A.shape[0])
    if w is None:
        return (((A[:, :-1] @ q[:-1] + q[-1]).view(A.shape[0]) - A[:, -1]) ** 2).sum()
    else:
        return (w * (((A[:, :-1] @ q[:-1] + q[-1]).view(A.shape[0]) - A[:, -1]) ** 2)).sum()


def solver_sklearn(A: torch.tensor, w: torch.tensor = None, out: bool = False, to_torch: bool = True) -> torch.Tensor:
    try:
        # looking for best line y=ax+b. best line:= minimizes MSE
        msg = 'type(A) was numpy'
        if isinstance(A, torch.Tensor):
            A = torch_to_numpy(A)
            msg = 'type(A) was torch'
        if isinstance(w, torch.Tensor):
            w = torch_to_numpy(w)

        if w is not None and (w < 0).any():
            # w can't be negative
            raise ValueError

        X, y = de_augment_numpy(A)
        n, d = X.shape
        linr = linear_model.LinearRegression()
        if len(y.shape) > 1:
            y = y.reshape(y.shape[0])
        linr.fit(X, y, sample_weight=w)
        a = linr.coef_  # slopes
        b = linr.intercept_  # bias

        a = a.reshape(a.shape[0], 1)
        b = np.zeros((1, 1)) + b
        q = np.row_stack((a, b))

        assert q.shape == (d + 1, 1)  # slope for each dim and 1 bias
        if out:
            a, b = q[:-1, 0], q[-1, 0]
            print('solver_sklearn:y={}x+{:,.2f}'.format(list_1d_to_str(list(a), 2), b))
            print('\t{}'.format(msg))
            print('\tq_opt: {}'.format(q.tolist(), 'q_opt', with_data=True))
        if to_torch:
            q = numpy_to_torch(q, to_double=False)
    except ValueError:
        # print('w has negative entry {}'.format(w.tolist()))
        q = None
    return q


def get_SP_SVD(A: torch.Tensor, folder: str, show_hist: bool = False) -> np.array:
    print('LinearReg.get_SP_SVD {}:'.format(A.shape))

    path = '{}/SVD.npy'.format(folder)

    if os.path.exists(path):
        SP_SVD = load_np(path, tabs=1)
        path_img = None
    else:
        if isinstance(A, torch.Tensor):
            A = torch_to_numpy(A)
        SP_SVD = calc_SP_SVD(A)
        save_np(SP_SVD, path, tabs=1)
        path_img = path.replace('.npy', '')

    msg = '\t|A|={}, |SP_SVD|={}, sum(SP_SVD)={:,.2f} (should be equal to d({}))'
    print(msg.format(A.shape, SP_SVD.shape, sum(SP_SVD), A.shape[1]))
    histogram(SP_SVD, title=path, save_path=path_img, show_hist=show_hist)
    return SP_SVD


def calc_SP_SVD(A: np.array) -> np.array:
    # sensitivity for linear_regression_origin:
    # # A = ai1 ... ain bi
    # # udvT = SVD(A)
    # # SPi = ||Ui||^2
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    SP_SVD = np.array([(np.linalg.norm(u) ** 2) for u in U])
    return SP_SVD


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
    initQ = torch_normal(shape=(cfgQ['init_size'], A.shape[1], 1), miu=cfgQ['init_miu'], std=cfgQ['init_std'])
    info_Q(initQ, A, 'initQ', with_error=cfgQ['infoQ_show_er'])
    Q = torch.empty((0, A.shape[1], 1))

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
    opt_loss = f_opt(A, solver_sklearn(A))
    opt_loss_avg = opt_loss / n
    print('\tOpt loss {:,.3f}. avg={:,.3f}'.format(opt_loss, opt_loss_avg))
    q_init_loss = f_opt(A, q)
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
            loss = f_opt(batch_A, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Q_epoch.append(q.detach().clone())  # each batch adds 1 q state - sample 10% from it later
            # print('\tA[{}:{}] avg loss:{:,.3f}'.format(i, min(i + bs, n), loss.item() / batch_A.shape[0]))

        Q += sample_from_list(Q_epoch, qs_from_each_epoch)
        real_avg_loss = f_opt(A, q) / n
        diff = torch.abs(1 - real_avg_loss / opt_loss_avg).item()

        if real_avg_loss < best_avg_loss:
            best_avg_loss = real_avg_loss
            best_e = epoch
            best_diff = diff

        if epoch % PROGRESS_PRINT == 0 or (diff < es_ratio):
            print(progress_msg.format(epoch, epochs, f_opt(A, q) / n, diff, optimizer.lr(), len(Q)))
            if diff < es_ratio:
                break
        optimizer.update_lr()

    print('Done training. Results:')
    print('\tOpt avg loss {:,.3f}'.format(opt_loss_avg))
    print('\tOur avg loss {:,.3f} (epoch {})'.format(best_avg_loss, best_e))
    print('\tbest_diff={:,.6f}'.format(best_diff))

    # convert to list of tensors to 3d tensor
    Q_t = torch.empty((len(Q), d + 1, 1))
    for i, q_ in enumerate(Q):
        q_.requires_grad = False
        Q_t[i] = q_
    # info_Q(Q_t, A, 'Q_t', False)
    return Q_t


def info_Q(Q: torch.tensor, A: torch.tensor, title: str, with_error: bool):
    print('\t|{}|={}:'.format(title, t_size_str(Q)))
    if with_error:
        n = A.shape[0]
        loss_A_on_q_opt = f_opt(A, solver_sklearn(A)).item()
        losses, losses_avg, diffs = [], [], []
        for q in Q:
            loss_A_on_q = f_opt(A, q).item()
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
