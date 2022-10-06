import wizzi_utils as wu
from Dependencies import UtilsScript as utils
from sklearn import svm
import torch
import math
import os
import numpy as np

DEBUG = False
DOUBLE_TORCH = True


def data_test():
    """ """
    from Dependencies import GetDataScript
    from sklearn.model_selection import train_test_split
    A_train, A_test = get_HTRU_2_MURAD_split(out_torch=False)
    sample = A_train[0]
    # sample = A_test[0]

    ds_name = 'HTRU_2'  # (i) HTRU [19] — 17; 898 radio emissions of the Pulsar star each consisting of 9 features.
    A1 = GetDataScript.get_data_A_htru_v2(ds_name, data_root='../../Datasets', preprocess_data=False)
    X1, y1 = utils.de_augment_numpy(A1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=42, test_size=0.1)
    A1_train, A1_test = utils.augment_x_y_numpy(X_train, y_train), utils.augment_x_y_numpy(X_test, y_test)
    print(wu.to_str(A1_train, 'A1_train'))
    print(wu.to_str(A1_test, 'A1_test'))
    # sample = A1_train[0]

    ans = any((A1_train[:] == sample).all(1))
    print('found sample in A1_train? {}'.format(ans))
    print(wu.to_str(sample, '\tsample'))
    return


def de_augment_A_numpy(data):
    """
    :param data: nx(d+1) points with labels
    :return: X nxd and y nx1
    """
    X, y = data[:, :-1], data[:, -1]
    return X, y


def de_augment_q_numpy(q):
    w, b = q[:-1], q[-1]
    # w = w.T
    return w, b


def get_HTRU_2_MURAD_split(out_torch: bool = True):
    A_train, A_test, SP_train = get_data_from_npz()
    if out_torch:
        A_train = wu.tt.numpy_to_torch(A_train, to_double=DOUBLE_TORCH)
    return A_train, A_test


def get_SP_MURAD():
    A_train, A_test, SP_MURAD = get_data_from_npz()
    # SP_MURAD = SP_MURAD / SP_MURAD.sum()
    msg = '|A|={}, |SP_MURAD|={}, sum(SP_MURAD)={:,.2f}'
    print(msg.format(A_train.shape, SP_MURAD.shape, sum(SP_MURAD)))
    hist_fp = './HTRU_2_16107_9/MURAD_NPZ/hist.png'
    if not os.path.exists(hist_fp):
        utils.histogram(SP_MURAD, title=hist_fp, save_path=hist_fp, show_hist=True)
    return SP_MURAD


def get_data_from_npz():
    """
    data origin https://archive.ics.uci.edu/ml/datasets/HTRU2
    loading Murad's file
    it contains a dictionary with the following keys
    s.keys() = ['P_test', 'SP', 'P_train']
    """
    path = './HTRU_2_16107_9/MURAD_NPZ/gilad.npz'

    file_obj = np.load(path)
    A_train = file_obj['P_train']
    A_test = file_obj['P_test']
    SP_train = file_obj['SP']
    print(wu.to_str(A_train, 'A_train'))
    print(wu.to_str(A_test, 'A_test'))
    print(wu.to_str(SP_train, 'SP_train'))
    return A_train, A_test, SP_train


def run_svm(A_train, A_test_x, w=None):
    clf = svm.SVC(kernel='linear')  # sklearn.svm

    X, y = de_augment_A_numpy(A_train)
    clf.fit(X, y, sample_weight=w)

    test_X, test_y = utils.de_augment_numpy(A_test_x)
    score = clf.score(test_X, test_y)
    return score * 100


def choose_and_run(A_train, A_test, SP, c_size, c_reps, title, verbose=0):
    if verbose > 0:
        print('{}:'.format(title))
    cum_score = 0.0
    for i in range(c_reps):
        C, W = utils.select_coreset(A_train, SP, c_size, out_torch=False)
        # TODO add if Cy is only one class - repeat selection
        # C, W = select_coreset_ben(A_train, SP, c_size)
        score_data = run_svm(C, A_test, W)
        if verbose > 0:
            print('\t{}: svm({} in size {:5}) = {:.4f}%'.format(i, title, c_size, score_data))
        cum_score += score_data
    cum_score /= c_reps
    return cum_score


def murad_test():
    start_t_all = wu.get_timer()
    c_sizes = [900]
    c_reps = 50
    A_train, A_test, SP_svm = get_data_from_npz()
    n = A_train.shape[0]
    score_full_data = run_svm(A_train, A_test, w=None)

    for c_size in c_sizes:
        start_t = wu.get_timer()
        score_C_murad = choose_and_run(A_train, A_test, SP_svm, c_size, c_reps, 'C_murad')

        SP_uniform = np.ones(shape=n)
        score_U = choose_and_run(A_train, A_test, SP_uniform, c_size, c_reps, 'Uniform')

        print('Summary(size {}, rep {}):'.format(c_size, c_reps))
        print('\tsvm(P in size {:5}) = {:.4f}%'.format(n, score_full_data))
        print('\tsvm(C in size {:5}) = {:.4f}% (Murad)'.format(c_size, score_C_murad))
        print('\tsvm(U in size {:5}) = {:.4f}% (Uniform)'.format(c_size, score_U))
        print('\ttime {}'.format(wu.get_timer_delta(start_t)))
    print('\ttime all exp {}'.format(wu.get_timer_delta(start_t_all)))
    return


def solver_sklearn(A: torch.tensor, weights: torch.tensor = None, out: bool = False, to_torch: bool = True):
    """
    @param A: nx(d+1) A=X|y |X|=nxd, |y|=nx1
    @param weights: nx1
    @param out:
    @param to_torch:
    @return: q_opt (d+1,1)
    """
    q_opt = None
    try:
        # looking for best line y=ax+b. best line:= minimizes MSE
        msg = 'type(A) was numpy'
        if isinstance(A, torch.Tensor):
            A = utils.torch_to_numpy(A)
            msg = 'type(A) was torch'
        if isinstance(weights, torch.Tensor):
            weights = utils.torch_to_numpy(weights)

        if weights is not None and (weights < 0).any():
            # w can't be negative
            raise ValueError

        X, y = de_augment_A_numpy(A)
        n, d = X.shape

        # sklearn.svm
        clf = svm.SVC(kernel='linear', C=1.0, verbose=DEBUG)  # newton
        clf.fit(X, y, sample_weight=weights)

        a = clf.coef_  # slopes
        b = clf.intercept_  # bias

        q_opt = np.column_stack((a, b)).T
        if DEBUG:
            print(wu.to_str(X, title='X'))
            print(wu.to_str(y, title='y'))
            if weights is not None:
                print(wu.to_str(weights, title='w'))
            print(wu.to_str(a, title='a'))
            print(wu.to_str(b, title='b'))
            print(wu.to_str(q_opt, title='q_opt'))
            # c = clf.support_
            # d = clf.n_support_
            # print(wu.to_str(c, title='c'))
            # print(wu.to_str(d, title='d'))

        assert q_opt.shape == (d + 1, 1)  # slope for each dim and 1 bias
        if out:
            # w, b = q[:-1], q[-1]
            w, b = de_augment_q_numpy(q_opt)
            # a, b = q[:, :-1], q[:, -1]
            list_w = utils.list_1d_to_str(list_of_floats=list(w.flatten()), float_precision=2)
            print('solver_sklearn:y={}x+{:,.2f}'.format(list_w, b[0]))
            print('\t{}'.format(msg))
            print(wu.to_str(q_opt, 'q_opt'))
            loss_q_opt = svm_soft_margin_np(A, q_opt, weights=None)
            print('\t{:.6f} svm_soft_margin_numpy'.format(loss_q_opt))
            # on A_train:
            #  885.381381 svm_soft_margin_np
            # -885.380038 sklearn
            # sklearn full:
            # [LibSVM].......
            # Warning: using -h 0 may be faster
            # *
            # optimization finished, #iter = 7979
            # obj = -885.380038, rho = -1.534607
            # nSV = 895, nBSV = 884
            # Total nSV = 895

        if to_torch:
            q_opt = utils.numpy_to_torch(q_opt, to_double=DOUBLE_TORCH)
    except ValueError as e:
        # print('w has negative entry {}'.format(w.tolist()))
        print('ValueError failed {}'.format(e))
    except Exception as e:
        print('Exception failed {}'.format(e))
    return q_opt


def f_opt(A, q, weights=None):
    # return svm_soft_margin_np(A, q, weights)
    return svm_soft_margin_torch(A, q, weights)


def svm_soft_margin_torch(A: torch.tensor, q: torch.tensor, weights: torch.tensor = None) -> float:
    """
    :param A: of size nx(d+1). A=X|y where X is nxd and y is 1x1
    :param q: of size 1x(d+1). q=w|b where w is 1xd and b is 1x1
    :param weights:  weights for each X if A is weighted
    https://www.baeldung.com/cs/svm-hard-margin-vs-soft-margin#:~:text=SVM%20with%20a%20Soft%20Margin,deal%20with%20one%20more%20constraint.
    slack = minimize(0.5||w||^2 + C*sigma(i=1 to n)c_i s.t. 1<=i<=n and c_i>=0
    hinge = max{0,1-y_i(w.T+b)}
    in HTRU_2 (murad data - split to train test of HTRU_2 ds which is (17897, 9))
    |A|=(16107, 9)  # train
    |A|=(1790, 9)  # test
    |X|=torch.Size([16107, 8])
    |y|=torch.Size([16107]
    |q|=torch.Size([9, 1])
    |w|=torch.Size([8, 1])
    |b|=torch.Size([1])
    slack: float
    |hinge|=torch.Size([16107])
    loss=slack+sum(hinge): float
    validation:
    q_opt = solver_sklearn(A, w=None, out=True, to_torch=True)
    # q_opt(torch.Tensor,s=torch.Size([9, 1]),dtype=torch.float64,trainable=False,is_cuda=False):
            [[0.48], [-0.01], [2.64], [-0.75], [-0.28], [0.3], [-0.03], [-0.08], [-1.53]]
    loss = svm_soft_margin_torch(A, q=q_opt, weights=None)
    # slack 3.9746391701750343
    # loss 885.3813806378993

    old comments:
    # 1
    # if weights is not None:  # numpy code - place in the begining
    #     A_n = 1 if A.ndim == 1 else A.shape[0]
    #     assert w.shape[0] == A_n
    #     A = (A.T * np.sqrt(w)).T

    # 2
    # reg = 1 / n * 0.5 * np.linalg.norm(w) ** 2 * np.sum(weights)  # 0.5 * ||w||^2
    # [sum(max(0,1-yi(w*xi - b)]
    # hinge = np.multiply(np.maximum(0, 1 - np.multiply(y, X.dot(w.T).flatten() + b)),weights)
    # loss = reg + np.sum(hinge)

    # 3(BEN):
    # MAYBE reg: reg *= np.sum(weights) / |A|
    # loss = np.sum(weights * hinge) + reg
    """

    X, y = de_augment_A_numpy(A)
    n, d = X.shape
    if DEBUG:
        print(wu.tt.to_str(X, 'X'))
        print(wu.tt.to_str(y, 'y'))
        print(wu.tt.to_str(q, 'q'))

    err = '{}, {}, {}'.format(X.shape, y.shape, q.shape)

    assert y.ndim == 1 and y.shape[0] == n and q.shape == (d + 1, 1), err

    w, b = de_augment_q_numpy(q)
    if DEBUG:
        print(wu.tt.to_str(w, 'w'))
        print(wu.tt.to_str(b, 'b'))
    err = '{}, {}'.format(w.shape, b.shape)
    assert w.shape == (d, 1) and b.ndim == 1 and b.shape[0] == 1, err

    slack = 0.5 * torch.norm(w, 2) ** 2  # 0.5 * ||w||^2
    a = 1 - torch.mul(y, torch.matmul(X, w).flatten() + b)
    hinge = torch.maximum(torch.zeros(size=a.size()), a)  # [sum(max(0,1-yi(w*xi - b)]
    if weights is not None:
        if DEBUG:
            print(wu.tt.to_str(weights, 'weights'))
        # print(wu.tt.to_str(hinge, 'hinge'))
        hinge = torch.mul(weights, hinge)
        # print(wu.tt.to_str(hinge, 'hinge'))
    loss = slack + torch.sum(hinge)

    if DEBUG:
        print('slack {}'.format(slack))
        print(wu.tt.to_str(a, 'a'))
        print(wu.tt.to_str(hinge, 'hinge'))
        print('loss {}'.format(loss))
    return loss


def svm_soft_margin_np(A: np.array, q: np.array, weights: np.array = None) -> float:
    """
    :param A: of size nx(d+1). A=X|y where X is nxd and y is 1x1
    :param q: of size 1x(d+1). q=w|b where w is 1xd and b is 1x1
    :param weights:  weights for each X if A is weighted
    https://www.baeldung.com/cs/svm-hard-margin-vs-soft-margin#:~:text=SVM%20with%20a%20Soft%20Margin,deal%20with%20one%20more%20constraint.
    slack = minimize(0.5||w||^2 + C*sigma(i=1 to n)c_i s.t. 1<=i<=n and c_i>=0
    hinge = max{0,1-y_i(w.T+b)}
    in HTRU_2 (murad data - split to train test of HTRU_2 ds which is (17897, 9))
    |A|=(16107, 9)  # train
    |A|=(1790, 9)  # test
    |X|=(16107, 8)
    |y|=(16107,)
    |q|=(9, 1)
    |w|=(1, 8)
    |b|=(1,)
    slack: float
    |hinge|=(16107,)
    loss=slack+sum(hinge): float
    validation:
    q_opt = solver_sklearn(A, w=None, out=True, to_torch=False)
    # q_opt(numpy.ndarray,s=(1, 9),dtype=float64): [[0.48, -0.01, 2.64, -0.75, -0.28, 0.3, -0.03, -0.08, -1.53]]
    loss = svm_soft_margin(A, q=q_opt, weights=None)
    # slack 3.9746391701750343
    # loss 885.381380637899
    """

    X, y = de_augment_A_numpy(A)
    n, d = X.shape
    if DEBUG:
        print(wu.to_str(X, 'X'))
        print(wu.to_str(y, 'y'))
        print(wu.to_str(q, 'q'))

    err = '{}, {}, {}'.format(X.shape, y.shape, q.shape)

    assert y.ndim == 1 and y.shape[0] == n and q.shape == (d + 1, 1), err

    w, b = de_augment_q_numpy(q)
    if DEBUG:
        print(wu.to_str(w, 'w'))
        print(wu.to_str(b, 'b'))
    err = '{}, {}'.format(w.shape, b.shape)
    assert w.shape == (d, 1) and b.ndim == 1 and b.shape[0] == 1, err
    slack = 0.5 * np.linalg.norm(w) ** 2  # 0.5 * ||w||^2
    hinge = np.maximum(0, 1 - np.multiply(y, X.dot(w).flatten() + b))  # [sum(max(0,1-yi(w*xi - b)]
    if weights is not None:
        if DEBUG:
            print(wu.to_str(weights, 'weights'))
        # print(wu.to_str(hinge, 'hinge'))
        # hinge = torch.mul(weights, hinge)
        hinge = np.multiply(weights, hinge)
        # print(wu.to_str(hinge, 'hinge'))
    loss = slack + np.sum(hinge)

    if DEBUG:
        print('slack {}'.format(slack))
        print(wu.to_str(hinge, 'hinge'))
        # a = np.nonzero(hinge)
        # print(a)
        print('loss {}'.format(loss))
    return loss


def info_Q(Q: torch.tensor, A: torch.tensor, title: str, with_error: bool):
    print('\t|{}|={}:'.format(title, utils.t_size_str(Q)))
    if with_error:
        n = A.shape[0]
        loss_A_on_q_opt = f_opt(A, solver_sklearn(A)).item()
        losses, losses_avg, diffs = [], [], []
        for q in Q:
            # noinspection PyUnresolvedReferences
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


def get_Q_sets(A: torch.tensor, cfgQ: dict) -> (torch.tensor, torch.tensor, torch.tensor, bool):
    start_time = utils.get_current_time()
    print('get_Q_sets')
    if os.path.exists(cfgQ['path']):
        existed = True
        allQ = utils.load_tensor(cfgQ['path'])
        info_Q(allQ, A, 'allQ', with_error=False)
    else:
        existed = False  # build Q and exit - rerun for experiments
        allQ = build_Q(A, cfgQ)
        info_Q(allQ, A, 'allQ', with_error=True)  # once for the log

    allQ = utils.shuffle_tensor(allQ)

    max_size = cfgQ['trainQ_s'] + cfgQ['valQ_s'] + cfgQ['testQ_s']
    Q = allQ[:max_size]
    Q = utils.add_cuda(Q)
    split1 = cfgQ['trainQ_s']
    split2 = cfgQ['valQ_s'] + cfgQ['trainQ_s']

    trainQ = Q[:split1]
    valQ = Q[split1:split2]
    testQ = Q[split2:]

    info_Q(trainQ, A, 'trainQ', with_error=cfgQ['infoQ_show_er'])
    info_Q(valQ, A, 'valQ', with_error=cfgQ['infoQ_show_er'])
    info_Q(testQ, A, 'testQ', with_error=cfgQ['infoQ_show_er'])
    if cfgQ['infoQ_show_er']:
        # from some reason info_Q with error changes the selection of C_init
        utils.set_cuda_scope_and_seed(seed=42)
    print('get_Q_sets time {}'.format(utils.get_time_str(start_time)))
    return trainQ, valQ, testQ, existed


def build_Q(A: torch.tensor, cfgQ: dict) -> (torch.tensor, torch.tensor, torch.tensor):
    print('build_Q:')
    epochs, bs, lr = cfgQ['epochs'], cfgQ['bs'], cfgQ['lr']
    initQ = utils.torch_normal(shape=(cfgQ['init_size'], A.shape[1], 1), miu=cfgQ['init_miu'], std=cfgQ['init_std'],
                               to_double=DOUBLE_TORCH)
    info_Q(initQ, A, 'initQ', with_error=cfgQ['infoQ_show_er'])
    Q = torch.empty((0, A.shape[1], 1))
    if DOUBLE_TORCH:
        Q = Q.double()

    for i, q in enumerate(initQ):
        print('{}/{}'.format(i + 1, initQ.shape[0]))
        Q_i = build_1_Q(A, q, epochs=epochs, bs=bs, lr=lr, mv=cfgQ['mv'], progress=0.1, sample=cfgQ['sample_step'])
        Q = torch.cat((Q, Q_i), 0)

    utils.save_tensor(Q, cfgQ['path'])
    print(wu.tt.to_str(Q, '\tQ'))
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
    optimizer = utils.OptimizerHandler(optimizer, mv['factor'], mv['patience'], mv['min_lr'])
    Q = []
    PROGRESS_PRINT = max(int(progress * epochs), 1)  # print progress each PROGRESS_PRINT epochs done
    progress_msg = '\tepoch [{}/{}]: avg loss:{:,.3f}, diff={:,.6f}, lr={:,.5f}, |Q|={}'
    best_avg_loss = float("inf")
    best_diff = -1
    best_e = -1
    for epoch in range(1, epochs + 1):
        Q_epoch = []
        A = utils.shuffle_tensor(A)

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


def test_weighted_loss():
    # TODO
    A, _ = get_HTRU_2_MURAD_split(out_torch=True)
    q_opt = solver_sklearn(A, weights=None, out=False, to_torch=True)
    # print(wu.to_str(A, 'A', chars=-1))
    # loss = svm_soft_margin_np(A, q=q_opt, weights=None)
    # print('{} svm_soft_margin_numpy'.format(loss))
    # exit(123)

    samples = torch.zeros(size=(3, A.shape[1])).double()
    samples[0] = A[33]
    samples[1] = A[52]
    samples[2] = A[64]
    losses_truth = torch.tensor([5.6254696157968604, 4.554592302869937, 3.9955330557641653])
    print(wu.tt.to_str(samples, 'samples', chars=-1))

    eps = 0.00001
    for samp, loss_truth in zip(samples, losses_truth):
        print('*' * 10)
        S = samp.unsqueeze(0)
        print(wu.to_str(S, 'S', chars=-1))
        loss = svm_soft_margin_torch(S, q=q_opt, weights=None)
        cond = abs(loss - loss_truth) < eps
        msg = '{} svm_soft_margin_numpy. expected?{}'.format(loss, cond)
        print(wu.add_color(msg, ops='g' if cond else 'r'))

    print('*' * 20)
    loss = svm_soft_margin_torch(samples, q=q_opt, weights=None)
    slack_truth = 3.9746391701750343
    hinge_truth = losses_truth - slack_truth
    cond = abs(loss - (slack_truth + torch.sum(hinge_truth))) < eps
    msg = '{} svm_soft_margin_numpy. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 0.0, 0.0]).double()
    loss = svm_soft_margin_torch(samples, q=q_opt, weights=weights)
    cond = abs(loss - slack_truth) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 1.0, 0.0]).double()
    loss = svm_soft_margin_torch(samples, q=q_opt, weights=weights)
    cond = abs(loss - losses_truth[1]) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 3.0, 0.0]).double()
    loss = svm_soft_margin_torch(samples, q=q_opt, weights=weights)
    loss_e = slack_truth + 3 * hinge_truth[1]
    cond = abs(loss - loss_e) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    weights = torch.tensor([0.0, 3.0, 2.0]).double()
    loss = svm_soft_margin_torch(samples, q=q_opt, weights=weights)
    loss_e = slack_truth + 3 * hinge_truth[1] + 2 * hinge_truth[2]
    cond = abs(loss - loss_e) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    samples_np = utils.torch_to_numpy(samples)
    weights_np = utils.torch_to_numpy(weights)
    q_opt_np = utils.torch_to_numpy(q_opt)
    loss = svm_soft_margin_np(samples_np, q=q_opt_np, weights=weights_np)
    loss_e = slack_truth + 3 * hinge_truth[1] + 2 * hinge_truth[2]
    cond = abs(loss - loss_e) < eps
    msg = '{} log_loss. expected?{}'.format(loss, cond)
    print(wu.add_color(msg, ops='g' if cond else 'r'))

    return


def main():
    # murad_test()
    # data_test()
    # A, _ = get_HTRU_2_MURAD_split(out_torch=True)
    # q_opt = solver_sklearn(A, weights=None, out=True, to_torch=True)
    # loss = svm_soft_margin_np(utils.torch_to_numpy(A), q=utils.torch_to_numpy(q_opt), weights=None)
    # print('{:.5f} svm_soft_margin_numpy'.format(loss))
    # loss = svm_soft_margin_torch(A, q=q_opt, weights=None)
    # print('{:.5f} svm_soft_margin_torch'.format(loss))
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
