import wizzi_utils as wu
from Dependencies import UtilsScript as utils
from Dependencies import GetDataScript
from DataReduction.SVM import handlerSVM as handler
from collections import defaultdict
from typing import Callable
import math
import torch
import os
import numpy as np

# name of group e.g. SGD
ERR_KEY = '{}_err_list'
STD_KEY = '{}_std_list'

WEAK_TITLE = '1/reps * sum(|1 - f(A,q1) / f(A,q2)|) where q1=solver(C,U), q2=solver(A)'
Q_OPT_TITLE = '1/reps * sum(|1 - f(C,U,q_opt) / f(A,q_opt))|) where q_opt=solver(A)'
STRONG_TITLE_MAX_MEAN = 'save avg of: for (Ci,Ui) in coresets: save max of: for q in Q: |1 - f(Ci,Ui,q) / f(A,q))|'
STRONG_TITLE_MEAN_MEAN = 'save avg of: for (Ci,Ui) in coresets: save avg of: for q in Q: |1 - f(Ci,Ui,q) / f(A,q))|'


def get_data(ds_name: str, data_limit: int = 0) -> torch.Tensor:
    A = GetDataScript.get_data_A_by_name(
        ds_name=ds_name, data_root='../../Datasets', data_limit=data_limit, out_torch_double=False)
    return A


def test_coresets(A: torch.Tensor, argv: dict, name_color_list, coresets_dict, testQ,
                  with_weak: bool = True, with_strong: bool = True):
    """
    :param argv:
    :param A: nxd+1 tensor
    :param name_color_list: item:= (sp name, sp color)
    :param coresets_dict: key=c_size, value=coresets_dict_per_size
        coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
        coreset_list: [(C,U), ... repetitions ...]
    :param testQ: Q set to check
        e.g. linear regression |q|=(d+1,1) so |testQ|=(#test,d+1,1)
    :param with_weak:
    :param with_strong:
    :return:
    """
    start_time = utils.get_current_time()
    print('test_coresets:')
    print('\tname_color_list:{}'.format(name_color_list))
    folder = argv['Bal']['folder']
    f, solver = argv['f'], argv['solver']
    c_sizes, reps = argv['c_sizes'], argv['reps']
    save_plots, show_plots = argv['save_plots'], argv['show_plots']
    with_std = True
    # prepare title base
    title_base = 'linear regression, ds {}({})\n'.format(argv['ds_name'], utils.t_size_str(A))
    names = ','.join([sp_name for (sp_name, sp_color) in name_color_list])
    title_base += '{} '  # test type (strong, weak, q_opt)
    title_base += 'on {} (other reps={}, bal reps={})\n'.format(names, argv['reps'], argv['Bal']['reps'])
    title_base += '{} '  # test definition

    print('\t|A|={}, c_sizes {}, reps={}, bal reps={}:'.format(utils.t_size_str(A), c_sizes, reps, argv['Bal']['reps']))
    error_A_on_qa = f(A, solver(A)).item()  # it's always the same
    print('\t\terror(A, q_opt)={:,.2f}'.format(error_A_on_qa / A.shape[0]))

    if with_weak:
        print('starting weak...')
        weak_time = utils.get_current_time()
        errors_dict = test_weak_many_c_sizes(f, solver, A, coresets_dict)
        title = title_base.format('weak test', WEAK_TITLE)
        save_path = '{}/{}_to_{}_weak'.format(folder, c_sizes[0], c_sizes[-1]) if save_plots else None
        out_results(c_sizes, errors_dict, name_color_list, title, save_path, show_plots, with_std)
        print('weak_time {}'.format(utils.get_time_str(weak_time)))

    if with_strong:
        print('starting strong...')
        st_mean_time = utils.get_current_time()
        title = title_base.format('strong test(|testQ|={})'.format(utils.t_size_str(testQ)), STRONG_TITLE_MEAN_MEAN)
        save_path2 = '{}/{}_to_{}_strong_mean_mean'.format(folder, c_sizes[0], c_sizes[-1]) if save_plots else None
        errors_dict = test_many_c_sizes_by_c(f, testQ, A, coresets_dict, use_max=False)
        out_results(c_sizes, errors_dict, name_color_list, title, save_path2, show_plots, with_std)
        print('st_mean_time {}'.format(utils.get_time_str(st_mean_time)))

    print('test_coresets time {}'.format(utils.get_time_str(start_time)))
    return


def out_results(c_sizes: list, errors_dict: dict, name_color_list: list, title: str, save_path: str = None,
                show_plot: bool = True, with_std: bool = False, float_pre: bool = 5):
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    :returns error dict:
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ value for c_size[0], ... , value for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ value for c_size[0], ... , value for c_size[-1] ]
    """
    print(title.split('\n')[-2])  # prints second last row
    groups = []
    for name, color in name_color_list:
        ers = errors_dict[ERR_KEY.format(name)]  # must exist
        stds = errors_dict[STD_KEY.format(name)] if with_std else None
        group = (ers, stds, color, name)
        msg = '\t{} errors: {}'.format(name, utils.list_1d_to_str(ers, float_pre))
        if stds is not None:
            msg += ', stds: {}'.format(utils.list_1d_to_str(stds, float_pre))
        print(msg)
        groups.append(group)

    if len(groups) > 0:
        utils.plot_x_y_std(c_sizes, groups, title, save_path=save_path, show_plot=show_plot, with_shift=True)
    return


# weak test
def test_weak_many_c_sizes(f: Callable, f_solver: Callable, A: torch.Tensor, coresets_dict: dict) -> dict:
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    :returns error dict:
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ np.mean(ers) for c_size[0], ... , np.mean(ers) for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ np.std(ers) for c_size[0], ... , np.mean(ers) for c_size[-1] ]
    ers for c_size[0]:= [f(A,q_a,q_cu_0), ... , f(A,q_a,q_cu_'reps')]
    """

    errors_dict = defaultdict(list)
    with torch.no_grad():
        for c_size, coresets_dict_per_size in coresets_dict.items():
            qa = f_solver(A)
            error_A_on_qa = f(A, qa).item()  # it's always the same
            for name, coreset_list in coresets_dict_per_size.items():
                errors_weak, errors_A_on_qc, errors_A_on_qa = test_weak_many_coresets(
                    f, f_solver, A, coreset_list, error_A_on_qa)
                print('{}: {}'.format(name, utils.list_1d_to_str(errors_weak, float_precision=4)))  # TODO DEBUG only
                errors_dict[ERR_KEY.format(name)].append(np.mean(errors_weak))
                errors_dict[STD_KEY.format(name)].append(np.std(errors_weak))

    return errors_dict


def test_weak_many_coresets(f: Callable, f_solver: Callable, A: torch.Tensor, coreset_list: list,
                            error_A_on_qa: float = None) -> (list, list, list):
    errors_weak, errors_A_on_qc, _ = [], [], []
    with torch.no_grad():
        if error_A_on_qa is None:
            qa = f_solver(A)
            error_A_on_qa = f(A, qa).item()  # it's always the same

        for (C, U) in coreset_list:
            error, error_A_on_qc, _ = test_weak_1_coreset(f, f_solver, A, C, U, error_A_on_qa)
            errors_weak.append(error)
            errors_A_on_qc.append(error_A_on_qc)
    return errors_weak, errors_A_on_qc, [error_A_on_qa] * len(coreset_list)


def test_weak_1_coreset(f: Callable, f_solver: Callable, A: torch.Tensor, C: torch.Tensor, U: torch.tensor = None,
                        er_A_on_qa: float = None) -> (float, float, float):
    with torch.no_grad():
        if er_A_on_qa is None:
            qa = f_solver(A)
            er_A_on_qa = f(A, qa).item()

        qc = f_solver(C, U)
        if qc is None:
            return -1.0, -1.0, -1.0
        er_A_on_qc = f(A, qc).item()
        if er_A_on_qa > er_A_on_qc:
            print('er_A_on_qa should be optimal')
            print('found er_A_on_qc lower than er_A_on_qa')
            print('{} > {}'.format(er_A_on_qa, er_A_on_qc))
            print('bug on solver or on loss function')
            exit(99)

        er = abs(1 - er_A_on_qc / er_A_on_qa)
    return er, er_A_on_qc, er_A_on_qa


# q_opt test
def test_q_opt_many_c_sizes(f: Callable, f_solver: Callable, A: torch.Tensor, coresets_dict: dict) -> dict:
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    :returns error dict:
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ np.mean(ers) for c_size[0], ... , np.mean(ers) for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ np.std(ers) for c_size[0], ... , np.mean(ers) for c_size[-1] ]
    ers for c_size[0]:= [f(A,q_opt,C_0,U_0), ... , f(A,q_opt,C_'reps',U_'reps')]
    """
    errors_dict = defaultdict(list)
    with torch.no_grad():
        for c_size, coresets_dict_per_size in coresets_dict.items():
            q_opt = f_solver(A)
            for name, coreset_list in coresets_dict_per_size.items():
                ers_q_opt_coresets, lcs_q_opt_coresets, las_q_opt_coresets = test_q_many_coreset(f, q_opt, A,
                                                                                                 coreset_list)
                errors_dict[ERR_KEY.format(name)].append(np.mean(ers_q_opt_coresets))
                errors_dict[STD_KEY.format(name)].append(np.std(ers_q_opt_coresets))
    return errors_dict


# balance test
def test_Q_1_coreset_mean(f: Callable, Q: torch.Tensor, A: torch.Tensor, C: torch.Tensor, U: torch.tensor = None,
                          title=None) -> float:
    """ used for balancing """
    ers, lcs, las = test_Q_1_coreset(f, Q, A, C, U)

    if title is not None:
        n = A.shape[0]
        msg = '{}|Q|={}, loss_c={:,.3f}, loss_p={:,.3f}, error={:,.5f}(max={:,.5f}) (std={:,.3f})'
        print(msg.format(title, len(Q), np.mean(lcs) / n, np.mean(las) / n, np.mean(ers), np.max(ers), np.std(ers)))
    return np.mean(ers)


# strong test: std by q
def test_many_c_sizes_by_q(f: Callable, Q: torch.Tensor, A: torch.Tensor, coresets_dict: dict,
                           use_max: bool = True) -> dict:
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]

    by q: error is the same as by CU, but STD different and reflect the difference of errors by q
    :returns error dict:
    if use_max True(if not - just debugging):
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ np.mean(ers_Q) for c_size[0], ... , np.mean(ers_Q) for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ np.std(ers_Q) for c_size[0], ... , np.mean(ers_Q) for c_size[-1] ]
    ers_Q for c_size[0]:= [max(er_q_0), ..., max(er_q_|Q|)] list in size |Q|
    er_q_0:= [f(A,q_0,C_0,U_0), ... , f(A,q_0,C_'reps',U_'reps')] list in size reps
    """
    errors_dict = defaultdict(list)
    with torch.no_grad():
        for c_size, coresets_dict_per_size in coresets_dict.items():
            for name, coreset_list in coresets_dict_per_size.items():
                ers_Q, lcs_Q, las_Q = test_Q_many_coreset(f, Q, A, coreset_list, use_max)
                errors_dict[ERR_KEY.format(name)].append(np.mean(ers_Q))
                errors_dict[STD_KEY.format(name)].append(np.std(ers_Q))
    return errors_dict


def test_Q_many_coreset(f: Callable, Q: torch.Tensor, A: torch.Tensor, coreset_list: list, use_max: bool = True) -> (
        list, list, list):
    """
    coreset_list: [(C,U), ... repetitions ...]
    :returns
        let q_i be the ith selection q in Q. for q_i in Q get list of losses on every (C,U) in coresets -> ers_q_i
        ers_Q: list of max loss of q_i: [max(ers_q_0), ... , max(ers_q_|Q|)]
    """
    ers_Q, lcs_Q, las_Q = [], [], []
    with torch.no_grad():
        for q in Q:
            ers_q_i, lcs_q_i, las_q_i = test_q_many_coreset(f, q, A, coreset_list)
            if use_max:
                ers_Q.append(max(ers_q_i))
                lcs_Q.append(max(lcs_q_i))
                las_Q.append(max(las_q_i))
            else:
                ers_Q.append(np.mean(ers_q_i))
                lcs_Q.append(np.mean(lcs_q_i))
                las_Q.append(np.mean(las_q_i))
    return ers_Q, lcs_Q, las_Q


def test_q_many_coreset(f: Callable, q: torch.Tensor, A: torch.Tensor, coreset_list: list) -> (list, list, list):
    """
    coreset_list: [(C,U), ... repetitions ...]
    :returns list of losses of q over coresets = {C_0, ... , C_reps}
    the list: [f(A,C_0,U_0,q), ... , f(A,C_reps,U_reps,q)]
    """
    ers_q_i, lcs_q_i, las_q_i = [], [], []
    with torch.no_grad():
        for (C, U) in coreset_list:
            er, lc, la = test_q_1_coreset(f, q, A, C, U)
            ers_q_i.append(er)
            lcs_q_i.append(lc)
            las_q_i.append(la)
    return ers_q_i, lcs_q_i, las_q_i


def test_q_1_coreset(f: Callable, q: torch.Tensor, A: torch.Tensor, C: torch.Tensor, U: torch.tensor = None) -> (
        float, float, float):
    with torch.no_grad():
        la = f(A, q).item()
        lc = f(C, q, U).item()
        er = abs(1 - lc / la)
    return er, lc, la


# strong test: std by c
def test_many_c_sizes_by_c(f: Callable, Q: torch.Tensor, A: torch.Tensor, coresets_dict: dict,
                           use_max: bool = True) -> dict:
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]

    by c: error is the same as by q, but STD different and reflect the difference of errors by CU
    :returns error dict:
    if use_max True(if not - just debugging):
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ np.mean(ers_C) for c_size[0], ... , np.mean(ers_C) for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ np.std(ers_C) for c_size[0], ... , np.mean(ers_C) for c_size[-1] ]
    ers_C for c_size[0]:= [max(er_c_0), ..., max(er_c_reps)] list in size reps
    er_c_0:= [f(A,q_0,C_0,U_0), ... , f(A,q_len(Q),C_0,U_0)] list in size |Q|
    """
    errors_dict = defaultdict(list)
    with torch.no_grad():
        for c_size, coresets_dict_per_size in coresets_dict.items():
            for name, coreset_list in coresets_dict_per_size.items():
                ers_C, lcs_C, las_C = test_many_coreset_on_Q(f, Q, A, coreset_list, use_max)
                errors_dict[ERR_KEY.format(name)].append(np.mean(ers_C))
                errors_dict[STD_KEY.format(name)].append(np.std(ers_C))
    return errors_dict


def test_many_coreset_on_Q(f: Callable, Q: torch.Tensor, A: torch.Tensor, coreset_list: list,
                           use_max: bool = True) -> (list, list, list):
    """
    coreset_list: [(C,U), ... repetitions ...]
    :returns
        let C_i be the ith selection (C,U). for C_i in coresets get list of losses on every q in Q -> ers_C_i
        ers_C: list of max loss of C_i: [max(ers_C_0), ... , max(ers_C_reps)]
    """
    ers_C, lcs_C, las_C = [], [], []
    with torch.no_grad():
        for (C, U) in coreset_list:
            ers_C_i, lcs_C_i, las_C_i = test_Q_1_coreset(f, Q, A, C, U)
            if use_max:
                ers_C.append(max(ers_C_i))
                lcs_C.append(max(lcs_C_i))
                las_C.append(max(las_C_i))
            else:
                ers_C.append(np.mean(ers_C_i))
                lcs_C.append(np.mean(lcs_C_i))
                las_C.append(np.mean(las_C_i))
    return ers_C, lcs_C, las_C


def test_Q_1_coreset(f: Callable, Q: torch.Tensor, A: torch.Tensor, C: torch.Tensor, U: torch.Tensor) -> (
        list, list, list):
    """
    :returns list: [f(A,C,U,q_0), ... , f(A,C,U,q_|Q|)]
    """
    ers_C_i, lcs_C_i, las_C_i = [], [], []
    with torch.no_grad():
        for q in Q:
            er, lc, la = test_q_1_coreset(f, q, A, C, U)
            ers_C_i.append(er)
            lcs_C_i.append(lc)
            las_C_i.append(la)
    return ers_C_i, lcs_C_i, las_C_i


# balance part
def build_1_bal_SGD_coreset(argv: dict, A: torch.Tensor,
                            C: torch.Tensor, U: torch.Tensor,
                            trainQ: torch.Tensor, valQ: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    print('build_1_bal_SGD_coreset(size={}):'.format(C.shape[0]))
    valQ4train = trainQ if valQ is None else valQ
    f, solver = argv['f'], argv['solver']
    lambda_loss = argv['Bal']['lambda']
    C_best, U_best = None, None  # faster than saving each improvment
    best_val_err, val_err, best_epoch = float("inf"), float("inf"), -1
    best_weak_er = float("inf")
    n = A.shape[0]
    trainable_vars = [C, U] if utils.is_trainable(C) and utils.is_trainable(U) else [C]
    opt = torch.optim.Adam(trainable_vars, lr=argv['Bal']['lr'])
    print('\toptimizer = {}, |trainable_vars| = {}'.format(utils.opt_to_str(opt), len(trainable_vars)))

    opt = utils.OptimizerHandler(opt, argv['Bal']['mv']['factor'], argv['Bal']['mv']['patience'],
                                 argv['Bal']['mv']['min_lr'])
    valid_values = torch.Tensor([-1., 1.]).double()
    new_rounding = True
    bal_time = utils.get_current_time()
    for epoch in range(1, argv['Bal']['epochs'] + 1):
        trainQ = utils.shuffle_tensor(trainQ)
        error_cum_grad = 0.0
        loss_p_ers, loss_c_ers, train_ers = [], [], []  # per epoch - for debugging
        count_b = 0

        for i, q in enumerate(trainQ):
            count_b += 1
            loss_p = f(A, q)
            loss_c = f(C, q, U)
            # error = torch.abs(1 - loss_c / loss_p)
            error = torch.abs(1 - loss_c / loss_p) + lambda_loss * torch.abs(n - torch.sum(U))  # sum(W) = n
            error_cum_grad += error

            loss_p_ers.append(loss_p.item())
            loss_c_ers.append(loss_c.item())
            train_ers.append(error.item())  # used to 'save' model if found new best

            # ============ Backward ============
            last_batch = (i == (len(trainQ) - 1))
            if (i + 1) % argv['Bal']['bs'] == 0 or last_batch:
                opt.zero_grad()
                error_cum_grad.backward()
                opt.step()
                with torch.no_grad():  # apply constraints:
                    # t = wu.get_timer()
                    if new_rounding:  # new rounding all labels at once
                        C_y = torch.round(C[..., -1])
                        C[..., -1] = C_y

                        C_counts = C_y.unique(return_counts=False)
                        # C_counts = C_y.unique(return_counts=True)
                        # print(C_counts)

                        if not torch.equal(valid_values, C_counts):
                            print('labels must be -1 or 1')
                            print(np.round(C.tolist(), 2))
                            print(var_to_string(C, 'C', with_data=True))
                            print(var_to_string(U, 'U', with_data=True))
                            exit(77)
                    else:  # old rounding - 1 by 1
                        for c_i in range(C.shape[0]):  # keep label entry as 1 or -1
                            C[c_i][-1] = torch.round(C[c_i][-1])
                            if C[c_i][-1] != -1 and C[c_i][-1] != 1:
                                print('labels must be -1 or 1')
                                print(np.round(C.tolist(), 2))
                                print(var_to_string(C, 'C', with_data=True))
                                print(var_to_string(U, 'U', with_data=True))
                                exit(77)
                    # print('time {}'.format(wu.get_timer_delta(s_timer=t, with_ms=True)))
                    # print('{} c==c2'.format(i), torch.equal(C, C2))
                    # noinspection PyArgumentList
                    U.clamp_(min=0)  # all weights >=0

                error_cum_grad = 0.0
                if argv['Bal']['batches_info']:
                    if i + 1 == argv['Bal']['bs']:
                        print('\tEpoch[{}/{}]'.format(epoch, argv['Bal']['epochs']))
                    msg = '\t\tbatch [{}/{}]'.format(i + 1, len(trainQ))
                    msg += ': error:{:,.5f}'.format(np.mean(train_ers[-count_b:]))
                    msg += ', loss_c:{:,.0f}'.format(np.mean(loss_c_ers[-count_b:]))
                    msg += ', loss_p:{:,.0f}'.format(np.mean(loss_p_ers[-count_b:]))
                    print(msg)
                count_b = 0

        found_new_best = False
        val_err = test_Q_1_coreset_mean(f, valQ4train, A, C, U)
        if val_err <= best_val_err:
            C_best = C.detach().clone()
            U_best = U.detach().clone()
            best_val_err, best_epoch = val_err, epoch
            found_new_best = True

        if argv['Bal']['epochs_info']:
            weak_er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C, U)
            if weak_er < best_weak_er:
                best_weak_er = weak_er
            msg = '\tEpoch[{}/{}]:'.format(epoch, argv['Bal']['epochs'])
            if argv['Bal']['epochs_info_stds']:
                msg += ' loss_c:{:,.0f}({:,.0f})'.format(np.mean(loss_c_ers) / n, np.std(loss_c_ers) / n)
                msg += ', loss_p:{:,.0f}({:,.0f})'.format(np.mean(loss_p_ers) / n, np.std(loss_p_ers) / n)
            else:
                msg += 'loss_c:{:,.5f}'.format(np.mean(loss_c_ers) / n)
                msg += ', loss_p:{:,.5f}'.format(np.mean(loss_p_ers) / n)
            msg += ', erTrain:{:,.5f}, erVal:{:,.5f}'.format(np.mean(train_ers), val_err)
            msg += ', erWeak:{:,.5f}, erAqc:{:,.3f}, erAqa:{:,.3f}'.format(weak_er, er_A_on_qc / n, er_A_on_qa / n)
            msg += ', lr {:,.5f}'.format(opt.lr())
            msg += ', time {}'.format(utils.get_time_str(bal_time))
            if found_new_best:
                msg += ', V'
            print(msg)
        opt.update_lr()  # <- this counts the epochs and tries to change the lr
    msg = 'Done balancing: epoch {}, best_val_err={:,.5f}, best_weak_er={:,.5f}, run time {}'
    print(msg.format(best_epoch, best_val_err, best_weak_er, utils.get_time_str(bal_time)))
    return C_best, U_best


# logistic regression based
def build_bal_SGD_coresets(A: torch.Tensor, argv: dict, trainQ: torch.Tensor, valQ: torch.Tensor = None,
                           testQ: torch.Tensor = None):
    start_time = utils.get_current_time()
    print('build_bal_SGD_coresets for |A|={} (DONE ONLY ONCE):'.format(A.shape))
    if valQ is None:
        print('\tc_sizes={}, bal_reps={}, |trainQ|={}'.format(argv['c_sizes'], argv['Bal']['reps'], len(trainQ)))
    else:
        print('\tc_sizes={}, bal_reps={}, |trainQ|={}, |valQ|={}'.format(argv['c_sizes'], argv['Bal']['reps'],
                                                                         len(trainQ), len(valQ)))

    bal_path_base = argv['Bal']['folder'] + '/bal_coreset_{}_rep_{}.pt'
    for c_size in argv['c_sizes']:
        c_size_start_time = utils.get_current_time()
        print('c_size {}:'.format(c_size))
        for rep_i in range(argv['Bal']['reps']):
            rep_i_start_time = utils.get_current_time()
            print('-' * 80, 'rep {}'.format(rep_i))
            bal_coreset_path = bal_path_base.format(c_size, rep_i)
            if os.path.exists(bal_coreset_path):
                print('\t{} already exists'.format(bal_coreset_path))
            else:
                f, solver = argv['f'], argv['solver']
                epochs, bs, lr = argv['Bal']['epochs'], argv['Bal']['bs'], argv['Bal']['lr']
                n, d = A.shape[0], A.shape[1] - 1

                C_init = utils.subset_init(c_size, A, trainable=False)
                U_init = utils.add_cuda(torch.zeros(C_init.shape[0]) + n / C_init.shape[0])  # init U
                if handler.DOUBLE_TORCH:
                    U_init = U_init.double()
                    C_init = C_init.double()
                print('\tstarting sum weights:')
                print('\t\tsum(W)={}'.format(n))
                print('\t\tsum(U)={}'.format(torch.sum(U_init).item()))

                C_init.requires_grad = True
                # U_init.requires_grad = False  # just like logistic
                U_init.requires_grad = True  # just like logistic

                total_batches = math.ceil(len(trainQ) / bs)
                msg = 'Balance params: epochs {}, batch_size {}, total batches {}, base_lr {}'
                print(msg.format(epochs, bs, total_batches, lr))

                # print('Tensors:')
                # print(var_to_string(C_init, '\tC', with_data=False))
                # print(var_to_string(U_init, '\tU', with_data=False))

                if argv['Bal']['loss_stats']:
                    print('Loss before training:')
                    print(wu.tt.to_str(trainQ, '\ttrainQ'))
                    print(wu.tt.to_str(A, '\tA'))
                    print(wu.tt.to_str(C_init, '\tC'))
                    print(wu.tt.to_str(U_init, '\tU'))
                    test_Q_1_coreset_mean(f, trainQ, A, C_init, U_init, title='\tloss(A,C,U,Q_train):')
                    if valQ is not None:
                        print(wu.tt.to_str(valQ, '\tvalQ'))
                        test_Q_1_coreset_mean(f, valQ, A, C_init, U_init, title='\tloss(A,C,U,Q_val  ):')
                    if testQ is not None:
                        print(wu.tt.to_str(testQ, '\ttestQ'))
                        test_Q_1_coreset_mean(f, testQ, A, C_init, U_init, title='\tloss(A,C,U,testQ  ):')
                    er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C_init, U_init)
                    msg = '\tWeak test(pre): f(A,lc)={:,.5f}, f(A,la)={:,.5f}, error={:,.5f}'
                    print(msg.format(er_A_on_qc / n, er_A_on_qa / n, er))

                C, U = build_1_bal_SGD_coreset(
                    argv=argv,
                    A=A,
                    C=C_init,
                    U=U_init,
                    trainQ=trainQ,
                    valQ=valQ)

                utils.save_tensor({'C': C, 'U': U}, bal_coreset_path)

                # print('\tTensors:')
                # print(var_to_string(C, '\tC', with_data=False))
                # print(var_to_string(U, '\tU', with_data=False))

                if argv['Bal']['loss_stats']:
                    print('Loss post training:')
                    print(wu.tt.to_str(C, '\tC'))
                    print(wu.tt.to_str(U, '\tU'))
                    test_Q_1_coreset_mean(f, trainQ, A, C, U, title='\tloss(A,C,U,Q_train):')
                    if valQ is not None:
                        test_Q_1_coreset_mean(f, valQ, A, C, U, title='\tloss(A,C,U,Q_val  ):')
                    if testQ is not None:
                        test_Q_1_coreset_mean(f, testQ, A, C, U, title='\tloss(A,C,U,testQ  ):')
                    er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C, U)
                    msg = '\tWeak test(post): f(A,lc)={:,.5f}, f(A,la)={:,.5f}, error={:,.5f}'
                    print(msg.format(er_A_on_qc / n, er_A_on_qa / n, er))

                print('\tending sum weights:')
                print('\t\tsum(W)={}'.format(n))
                print('\t\tsum(U)={}'.format(torch.sum(U).item()))

            print('-' * 80, 'rep {} time {}'.format(rep_i, utils.get_time_str(rep_i_start_time)))
        print('c_size = {}: time {}'.format(c_size, utils.get_time_str(c_size_start_time)))
    print('build_bal_SGD_coresets time {}'.format(utils.get_time_str(start_time)))
    return


def get_sp_list(A: torch.Tensor, argv: dict, group_names: list, trainQ: torch.Tensor, valQ: torch.Tensor,
                testQ: torch.Tensor) -> (list, list):
    """ :returns list of tuples. tuple (SP vector, sp name, sp color) """
    start_time = utils.get_current_time()
    print('get_sp_list:')
    sp_list = []
    for group_name in group_names:
        SP = None
        if 'SGD' in group_name:
            # done once per A for every c_size - needs trainQ and valQ that are not in testQ
            build_bal_SGD_coresets(A, argv, trainQ, valQ, testQ)
        elif 'MURAD' in group_name:
            SP = handler.get_SP_MURAD()  # SP 1
        elif 'UNI' in group_name:
            SP = np.ones(A.shape[0])
        sp_list.append((SP, group_name, argv['color_dict'][group_name]))

    names = ','.join([sp_name for (SP, sp_name, sp_color) in sp_list])
    print('sp_list: {}'.format(names))
    name_color_list = [(sp_tuple[1], sp_tuple[2]) for sp_tuple in sp_list]  # SP is not needed anymore
    print('get_sp_list time {}'.format(utils.get_time_str(start_time)))
    return sp_list, name_color_list


def get_bal_coresets(bal_folder: str, c_size: int, bal_reps: int) -> list:
    coreset_list = []
    for i in range(bal_reps):
        C_U_tuple = get_bal_coreset(bal_folder, c_size, bal_rep=i)  # loading SGD bal coreset for each size
        coreset_list.append(C_U_tuple)
    return coreset_list


def get_bal_coreset(bal_folder: str, c_size: int, bal_rep: int) -> (torch.Tensor, torch.Tensor):
    path = bal_folder + '/bal_coreset_{}_rep_{}.pt'.format(c_size, bal_rep)
    assert os.path.exists(path), '{} doesn\'t exist. call build_bal_SGD_coresets()'.format(path)
    save_dict = utils.load_tensor(path, tabs=1)
    C, U = save_dict['C'], save_dict['U']
    if handler.DOUBLE_TORCH:
        U = U.double()
        C = C.double()
    C = utils.add_cuda(C)
    U = utils.add_cuda(U)

    # print(var_to_string(C, '\tC', with_data=True))
    # print(var_to_string(U, '\tU', with_data=True))
    return C, U


def get_list_of_coresets_for_each_size(A: torch.Tensor, sp_list: list, argv: dict) -> dict:
    """
    if SP is none and name == 'SGD', loading pre made balance coresets
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    """
    start_time = utils.get_current_time()
    print('get_list_of_coresets_for_each_size:')
    c_sizes, reps, bal_reps, bal_folder = argv['c_sizes'], argv['reps'], argv['Bal']['reps'], argv['Bal']['folder']

    names = ','.join([sp_name for (SP, sp_name, sp_color) in sp_list])
    print('\t|A|={}, reps={}, c_sizes={}, sp_names={}'.format(utils.t_size_str(A), reps, c_sizes, names))
    coresets_dict = {}

    for c_size in c_sizes:  # coresets: 'reps' x (C,U) in size c_size
        coresets_dict_per_size = {}
        for i, (SP, sp_name, sp_color) in enumerate(sp_list):
            if sp_name == 'SGD':  # no SP for SGD
                coreset_list = get_bal_coresets(bal_folder, c_size, bal_reps)
            else:
                coreset_list = get_x_coresets(reps, c_size, SP, A, regular=True, out_torch=True)
            coresets_dict_per_size[sp_name] = coreset_list
        msg = ['{} {}x{} coresets'.format(sp_name, len(coreset_list), utils.t_size_str(coreset_list[0][0]))
               for sp_name, coreset_list in coresets_dict_per_size.items()]
        print('\tfor size {}: created {}'.format(c_size, ', '.join(msg)))
        coresets_dict[c_size] = coresets_dict_per_size
    print('get_list_of_coresets_for_each_size time {}'.format(utils.get_time_str(start_time)))
    return coresets_dict


def get_x_coresets(c_reps: int, c_size: int, SP: np.array, A: torch.Tensor, regular: bool = True,
                   out_torch: bool = True) -> list:
    """  :return: list of coresets. coreset:= (C,U) """
    coresets_list = []
    for i in range(c_reps):
        if regular:
            C, W = utils.select_coreset(A, SP, c_size, out_torch, out_torch_double=False)
        else:
            C, W = utils.select_coreset_frequency(A, SP, c_size, out_torch_double=False)
        coresets_list.append((C, W))
    return coresets_list


def get_argv(A: torch.Tensor, ds_name: str) -> dict:
    # noinspection PyDictCreation
    argv = {}
    n, d = A.shape[0], A.shape[1]
    argv['ds_name'] = ds_name
    argv['base_folder'] = './{}_{}_{}'.format(ds_name, n, d)  # e.g. diabetes_442_11/
    utils.create_dir_if_not_exists(data_root=argv['base_folder'], ack=True, tabs=0)

    argv['f'] = handler.f_opt  # loss function
    argv['solver'] = handler.solver_sklearn  # solver

    argv['Q'] = {
        'path': '{}/Q1.pt'.format(argv['base_folder']),
        'init_size': 10,  # different init. each build_1_Q get a different init
        'init_miu': 0,
        'init_std': 1,
        'sample_step': 0.1,
        'epochs': 1000,  # epochs for build_1_Q
        'bs': 1000,  # step each 'bs' samples from data set
        'lr': 0.001,  # lr for build_1_Q
        'mv': {  # moving lr
            'factor': 0.1,
            'patience': 100000,  # cancels moving lr
            'min_lr': 0.0001,
        },
        'infoQ_show_er': True,  # prints stats on Q sets (takes time)  # TODO set True on final run
        # 'trainQ_s': 200,  # TODO set
        # 'valQ_s': 200,  # TODO set
        # 'testQ_s': 2000  # TODO set
        'trainQ_s': 10000,
        'valQ_s': 2000,
        'testQ_s': 1000
    }

    argv['color_dict'] = {'SGD': 'b', 'MURAD': 'g', 'UNI': 'r'}
    # argv['MURAD_sp_folder'] = '{}'.format(argv['base_folder'])  # 2 arrays (values, Q)

    argv['Bal'] = {
        'folder': '{}/BalQ1_6_6_all'.format(argv['base_folder']),
        # BALANCE METHOD PARAMS - DONE ONCE
        'reps': 5,
        'epochs': 200,
        'bs': 100,
        'lr': 0.001,
        'mv': {  # moving lr
            'factor': 0.1,
            'patience': 10000000,  # cancels moving lr
            'min_lr': 0.0001,
        },
        'lambda': 1,
        'epochs_info': True,
        'epochs_info_stds': False,
        'batches_info': False,
        'loss_stats': True,  # runs tests before and after training - optional
    }
    argv['show_plots'] = False
    argv['save_plots'] = True
    argv['with_weak'] = True
    argv['with_strong'] = True

    # argv['c_sizes'] = [i for i in range(50, 141, 10)]  # TODO c_size
    # argv['c_sizes'] = [800]  # TODO c_size
    # argv['c_sizes'] = [1000]  # TODO c_size
    # argv['c_sizes'] = [1200]  # TODO c_size
    # argv['c_sizes'] = [1400]  # TODO c_size
    # argv['c_sizes'] = [1600]  # TODO c_size
    # argv['c_sizes'] = [1800]  # TODO c_size
    argv['c_sizes'] = [800, 1000, 1200, 1400, 1600, 1800]  # TODO c_size
    # argv['c_sizes'] = [800, 1000, 1200, 1400]  # TODO c_size
    # argv['reps'] = 1  # TODO reps
    argv['reps'] = 100  # TODO reps

    print('argv:')
    for k, v in argv.items():
        print('\t{}: {}'.format(k, v))

    utils.create_dir_if_not_exists(data_root=argv['Bal']['folder'], ack=True, tabs=0)
    log_fp = '{}/{}.txt'.format(argv['Bal']['folder'], os.path.basename(argv['Bal']['folder']))
    if not os.path.exists(log_fp):
        # open a log file to copy and paste the output after the run ended
        log_file = open(file=log_fp, mode='w', encoding='utf-8')
        log_file.close()
    return argv


def main_balance(argv: dict, A: torch.Tensor, group_names: list):
    # print('Balancing:')
    trainQ, valQ, testQ, Q_existed = handler.get_Q_sets(A, argv['Q'])
    if Q_existed:  # if new Q - save log and re run for compression
        sp_list, name_color_list = get_sp_list(A, argv, group_names, trainQ, valQ, testQ)
        coresets_dict = get_list_of_coresets_for_each_size(A, sp_list, argv)
        test_coresets(A, argv, name_color_list, coresets_dict, testQ, with_weak=argv['with_weak'],
                      with_strong=argv['with_strong'])
    return


def main():
    utils.set_cuda_scope_and_seed(seed=42)
    ds_name = 'HTRU_2'  # (i) HTRU [19] — 17; 898 radio emissions of the Pulsar star each consisting of 9 features.
    # A = get_data(ds_name=ds_name, data_limit=0)
    A, _ = handler.get_HTRU_2_MURAD_split()
    argv = get_argv(A, ds_name)
    # main_balance(argv, A, group_names=['MURAD', 'UNI'])
    main_balance(argv, A, group_names=['MURAD', 'UNI', 'SGD'])
    return


def read_scores():
    fp = r'D:\workspace\2019SGD\DataReduction\SVM\HTRU_2_16107_9\BalQ1_6_6_2_800_1000\BalQ1_6_6_2_800_1000.txt'
    print(os.path.exists(fp))
    file = open(fp, 'r')
    lines = file.readlines()
    errs = []
    for line in lines:
        line = line.strip()
        if line.startswith('Weak test: f(A,lc)=0, f(A,la)=0'):
            # print(line)
            err = line.split(' ')[-1]
            err = err.split('=')[-1]
            err = float(err)
            errs.append(err)
            # print(err)
    file.close()
    print('errs avg = {} | all = {}'.format(np.mean(errs), errs))
    return


def _plot_3d_data(P: np.array, C: np.array):
    pad_p = 0.1
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    # X, Y, Z = C[:, 0], C[:, 1], C[:, 2]
    min_x, max_x = np.min(X), np.max(X)
    # print(min_x, max_x)
    x_pad = (max_x - min_x) * pad_p
    min_x -= x_pad
    max_x += x_pad
    # print(min_x, max_x)

    min_y, max_y = np.min(Y), np.max(Y)
    # print(min_y, max_y)
    y_pad = (max_y - min_y) * pad_p
    min_y -= y_pad
    max_y += y_pad
    # print(min_y, max_y)

    min_z, max_z = np.min(Z), np.max(Z)
    # print(min_z, max_z)
    z_pad = (max_z - min_z) * pad_p
    min_z -= z_pad
    max_z += z_pad
    # print(min_z, max_z)

    data_plot_0 = {
        'legend': wu.pyplt.LEGEND_DEFAULT,
        'fig_lims': [min_x, max_x, min_y, max_y, min_z, max_z],
        'color_axes': 'black',
        'view': {'azim': -60, 'elev': 30},
        'face_color': 'white',
        'background_color': 'white',
        'xticks': None,
        'yticks': None,
        'zticks': None,
        'datum': [
            {
                'data': P,
                # 'data': P[:1000],
                'c': wu.pyplt.get_rgba_color('green', opacity=0.3),
                'label': 'P({}, {})'.format(P.shape[0], P.shape[1]),
                'marker': '.',
                'marker_s': 10,
            },
            {
                'data': C,
                'c': wu.pyplt.get_rgba_color('red', opacity=1.0),
                'label': 'C({}, {})'.format(C.shape[0], C.shape[1]),
                'marker': 'x',
                'marker_s': 100,
            }
        ]
    }
    title = 'HTRU_2 experiment'
    wu.pyplt.plot_3d_one_dashboard(
        fig_title=title,
        data_plot_i=data_plot_0,
        win_d={
            'title': title,
            'location': 'tl',
            'resize': None,
            'zoomed': True,
        },
        center_d=None,
        max_time=None
    )
    return


def plot_3d_data():
    from sklearn.decomposition import PCA
    ds_name = 'HTRU_2'
    A = GetDataScript.get_data_A_by_name(data_root='../../Datasets', ds_name=ds_name, out_torch=False)
    # print(wu.to_str(A, 'A', chars=0))
    # A = A[:1000]
    Xa, Ya = utils.de_augment_numpy(A)
    pca_Xa = PCA(n_components=2)
    Xa_2d = pca_Xa.fit_transform(Xa)
    A_3d = utils.augment_x_y_numpy(Xa_2d, Ya)
    print(wu.to_str(A_3d, 'A_3d', chars=0))

    c_size, bal_rep = 1800, 0
    path = 'HTRU_2_16107_9/BalQ1_6_6_all/bal_coreset_{}_rep_{}.pt'.format(c_size, bal_rep)
    assert os.path.exists(path), '{} doesn\'t exist. call build_bal_SGD_coresets()'.format(path)
    save_dict = utils.load_tensor(path, tabs=1)
    C, U = utils.torch_to_numpy(save_dict['C']), utils.torch_to_numpy(save_dict['U'])
    print(wu.to_str(C, 'C', chars=0))
    print(wu.to_str(U, 'U', chars=0))
    Xc, Yc = utils.de_augment_numpy(C)
    pca_Xc = PCA(n_components=2)
    Xc_2d = pca_Xc.fit_transform(Xc)
    C_3d = utils.augment_x_y_numpy(Xc_2d, Yc)
    print(wu.to_str(C_3d, 'C_3d', chars=0))
    # U = np.expand_dims(U, axis=0)
    # UC = U @ C
    # print(wu.to_str(UC, 'UC', chars=0))

    # wu.pyplt.test.plot_3d_one_dashboard()
    _plot_3d_data(P=A_3d, C=C_3d)
    return


if __name__ == "__main__":
    wu.main_wrapper(
        main_function=plot_3d_data,
        seed=-1,
        ipv4=False,
        cuda_off=True,  # change to False for GPU
        torch_v=True,
        tf_v=False,
        cv2_v=False,
        with_pip_list=False,
        with_profiler=False
    )
