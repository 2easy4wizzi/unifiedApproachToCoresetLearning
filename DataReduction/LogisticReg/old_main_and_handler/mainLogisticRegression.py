from collections import defaultdict
from typing import Callable
from Dependencies.UtilsScript import *
import sys
from Dependencies import GetDataScript
from DataReduction.LogisticReg.old_main_and_handler import handlerLogisticRegression as Log
import math

# name of group e.g. SGD
ERR_KEY = '{}_err_list'
STD_KEY = '{}_std_list'

WEAK_TITLE = 'weak test: 1/reps * sum(|1 - f(A,q1) / f(A,q2)|) where q1=solver(C,U), q2=solver(A)'
Q_OPT_TITLE = 'q_opt test: 1/reps * sum(|1 - f(C,U,q_opt) / f(A,q_opt))|) where q_opt=solver(A)'
STRONG_TITLE_MAX_MEAN = 'save avg of: for (Ci,Ui) in coresets: save max of: for q in Q: |1 - f(Ci,Ui,q) / f(A,q))|'
STRONG_TITLE_MEAN_MEAN = 'save avg of: for (Ci,Ui) in coresets: save avg of: for q in Q: |1 - f(Ci,Ui,q) / f(A,q))|'


def load_tmf_sp():
    tmf_path = 'HTRU_2_17897_9/HTRU_2_tmf.npz'
    npz = load_np(tmf_path)
    print(type(npz))
    for k in npz.files:
        print(k)
    sp_tmf = npz['sensitivity']
    print(type(sp_tmf), len(sp_tmf), sp_tmf)
    return sp_tmf


def get_data(ds_name: str, data_limit: int = 0, two_d: bool = False) -> torch.Tensor:
    print('get_data(ds_name={}):'.format(ds_name))
    A = GetDataScript.get_data_A_by_name(ds_name, '../../Datasets', data_limit=data_limit, two_d=two_d)
    return A


def get_x_coresets(c_reps: int, c_size: int, SP: np.array, A: torch.Tensor, regular: bool = True,
                   out_torch: bool = True) -> list:
    """  :return: list of coresets. coreset:= (C,U) """
    coresets_list = []
    for i in range(c_reps):
        valid = False
        C, W = None, None
        while not valid:  # select from class 0 and class 1 - invalid otherwise
            if regular:
                C, W = select_coreset(A, SP, c_size, out_torch)
            else:
                C, W = select_coreset_frequency(A, SP, c_size)
            _, C_y = Log.a_to_x_y(C)

            s = torch.sum(C_y).item()
            valid = -c_size < s < c_size  # can't allow only -ones or only ones
        coresets_list.append((C, W))
    return coresets_list


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
    print(title.split('\n')[-1])  # prints last row
    groups = []
    for name, color in name_color_list:
        ers = errors_dict[ERR_KEY.format(name)]  # must exist
        stds = errors_dict[STD_KEY.format(name)] if with_std else None
        group = (ers, stds, color, name)
        msg = '\t{} errors: {}'.format(name, list_1d_to_str(ers, float_pre))
        if stds is not None:
            msg += ', stds: {}'.format(list_1d_to_str(stds, float_pre))
        print(msg)
        groups.append(group)

    if len(groups) > 0:
        plot_x_y_std(c_sizes, groups, title, save_path=save_path, show_plot=show_plot, with_shift=True)
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
                errors_weak, errors_A_on_qc, errors_A_on_qa = test_weak_many_coresets(f, f_solver, A, coreset_list,
                                                                                      error_A_on_qa)
                errors_dict[ERR_KEY.format(name)].append(np.mean(errors_weak))
                errors_dict[STD_KEY.format(name)].append(np.std(errors_weak))

    # if A.shape[1] == 2:
    #     first_c_size = list(coresets_dict.keys())[0]
    #     plotCU_groups(A, first_c_size, coresets_dict[first_c_size])
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

        er = abs(1 - er_A_on_qc / er_A_on_qa)

        if er >= 10 ** 10:
            print('\nError is bigger than 10^10')
            print(np.round(C.tolist(), 2))
            print(var_to_string(C, 'C', with_data=True))
            print(var_to_string(U, 'U', with_data=True))
            print(var_to_string(qc, 'qc', with_data=True))
            print(er, er_A_on_qc, er_A_on_qa)

            # l0 = Log.log_loss_opt(A, qc)
            # print('\tlog_loss_opt(A,qc)       ={:,.12f}'.format(l0))

            # l1 = Log.log_loss_sklearn_np(torch_to_numpy(A), torch_to_numpy(qc))
            # print('\tlog_loss_sklearn_np(A,qc)={:,.12f}'.format(l1))

            # l2 = Log.log_loss_np_opt(torch_to_numpy(A), torch_to_numpy(qc))
            # print('\tlog_loss_np_opt(A,qc)    ={:,.12f}'.format(l2))

            # A_np = torch_to_numpy(A)
            # qc_np = torch_to_numpy(qc)
            # qc_np = qc_np.astype('float64')
            # print(var_to_string(A_np, 'A_np'))
            # print(var_to_string(qc_np, 'qc_np', with_data=True))
            # l3 = Log.log_loss_np(A_np, qc_np)
            # print('\tlog_loss_np(A,qc)        ={:,.12f}'.format(l3))
            exit(22)

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
        print(title)
        msg = '\t\t|Q|={}, loss_c={:,.3f}, loss_p={:,.3f}, error={:,.8f} (std={:,.3f})'
        print(msg.format(Q.shape[0], np.mean(lcs), np.mean(las), np.mean(ers), np.std(ers)))
    return float(np.mean(ers))


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
    coreset_list: [(C,U), ... repetitions ...]
    :returns list of losses of (C,U) over Q = {q_0, ... , q_|Q|}
    the list: [f(A,C,U,q_0), ... , f(A,C,U,q_|Q|)]
    """
    ers_C_i, lcs_C_i, las_C_i = [], [], []
    with torch.no_grad():
        for q in Q:
            er, lc, la = test_q_1_coreset(f, q, A, C, U)
            ers_C_i.append(er)
            lcs_C_i.append(lc)
            las_C_i.append(la)
    return ers_C_i, lcs_C_i, las_C_i


def get_sp_list(A: torch.Tensor, group_names: list, color_dict: dict) -> (list, list):
    """ :returns list of tuples. tuple (SP vector, sp name, sp color) """
    print('\nget_sp_list:')
    sp_list = []
    for group_name in group_names:
        SP = None
        if 'SGD' in group_name:  # SP stays none
            pass
        elif 'UNI' in group_name:
            SP = np.ones(A.shape[0])
        elif 'TMF' in group_name:
            SP = load_tmf_sp()
        sp_list.append((SP, group_name, color_dict[group_name]))

    names = ','.join([sp_name for (SP, sp_name, sp_color) in sp_list])
    print('sp_list: {}'.format(names))
    name_color_list = [(sp_tuple[1], sp_tuple[2]) for sp_tuple in sp_list]  # SP is not needed anymore
    return sp_list, name_color_list


def get_list_of_coresets_for_each_size(A: torch.Tensor, sp_list: list, c_sizes: list, reps: int,
                                       bal_folder: str = None) -> dict:
    """
    if SP is none and name == 'SGD', loading pre made balance coresets
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    """
    print('\nget_list_of_coresets_for_each_size:')
    names = ','.join([sp_name for (SP, sp_name, sp_color) in sp_list])
    print('\t|A|={}, reps={}, c_sizes={}, sp_names={}'.format(t_size_str(A), reps, c_sizes, names))
    coresets_dict = {}

    for c_size in c_sizes:  # coresets: 'reps' x (C,U) in size c_size
        coresets_dict_per_size = {}
        for i, (SP, sp_name, sp_color) in enumerate(sp_list):
            if SP is None and sp_name == 'SGD':
                C, U = get_bal_coreset(bal_folder, c_size)  # loading SGD bal coreset for each size
                # LinearRegHandler.weak_test(A, C, U, title='Weak: post balance', plot=True)
                coreset_list = [(C, U)]
            else:
                coreset_list = get_x_coresets(reps, c_size, SP, A, regular=True, out_torch=True)
            coresets_dict_per_size[sp_name] = coreset_list
        msg = ['{} {}x{} coresets'.format(sp_name, len(coreset_list), t_size_str(coreset_list[0][0]))
               for sp_name, coreset_list in coresets_dict_per_size.items()]
        print('\tfor size {}: created {}'.format(c_size, ', '.join(msg)))
        coresets_dict[c_size] = coresets_dict_per_size
    return coresets_dict


def test_coresets(A, f, name_color_list, coresets_dict, testQ_list, c_sizes, reps, title_base, folder, save_plots,
                  show_plots, with_std=False, solver_f=None):
    """
    :param A: nxd+1 tensor
    :param f: loss function
    :param name_color_list: item:= (sp name, sp color)
    :param coresets_dict: key=c_size, value=coresets_dict_per_size
        coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
        coreset_list: [(C,U), ... repetitions ...]
    :param testQ_list: item:= (testQ:=tensor, titleQ:= 1 word title. used also for save prefix, descQ:= description)
        e.g. linear regression |q|=(d+1,1) so |testQ|=(#test,d+1,1)
    :param c_sizes: coreset sizes
    :param reps: repetitions for each c_size
    :param title_base: title base for plot (ds name problem ....)
    :param folder: were to load SP from and where to save plots
    :param save_plots:
    :param show_plots: can save without showing
    :param with_std:
    :param solver_f: if solver exists, also run weak test and strong test
    :return:
    """
    print('\nname_color_list:{}'.format(name_color_list))
    print('|A|={}, c_sizes {}, reps={}:'.format(t_size_str(A), c_sizes, reps))

    if solver_f is not None:
        errors_dict = test_weak_many_c_sizes(f, solver_f, A, coresets_dict)
        title = title_base.format(WEAK_TITLE, 'weak test')
        save_path = '{}/{}_{}'.format(folder, c_sizes, 'weak') if save_plots else None
        out_results(c_sizes, errors_dict, name_color_list, title, save_path, show_plots, with_std)

        errors_dict = test_q_opt_many_c_sizes(f, solver_f, A, coresets_dict)
        title = title_base.format(Q_OPT_TITLE, 'q opt test')
        save_path = '{}/{}_{}'.format(folder, c_sizes, 'q_opt') if save_plots else None
        out_results(c_sizes, errors_dict, name_color_list, title, save_path, show_plots, with_std)

    for testQ_tuple in testQ_list:
        testQ, titleQ, descQ = testQ_tuple
        # # TODO ADDING q opt to testQ
        # print(var_to_string(testQ, 'testQ', with_data=False))
        # q_opt = solver_f(A).unsqueeze(0)
        # print(var_to_string(q_opt, 'qopt', with_data=False))
        #
        # testQ = torch.cat((testQ, q_opt), 0)
        # print(var_to_string(testQ, 'testQ', with_data=False))

        if isinstance(testQ, torch.Tensor):
            descQ_full = '|{}|={}, description: {}'.format(titleQ, t_size_str(testQ), descQ)
        else:
            descQ_full = '{} {}'.format(titleQ, descQ)

        title = title_base.format(STRONG_TITLE_MAX_MEAN, descQ_full)
        save_path = '{}/{}_{}'.format(folder, c_sizes, titleQ) if save_plots else None
        errors_dict = test_many_c_sizes_by_c(f, testQ, A, coresets_dict, use_max=True)
        out_results(c_sizes, errors_dict, name_color_list, title, save_path, show_plots, with_std)

        title = title_base.format(STRONG_TITLE_MEAN_MEAN, descQ_full)
        save_path2 = '{}/{}_{}_mean_mean'.format(folder, c_sizes, titleQ) if save_plots else None
        errors_dict = test_many_c_sizes_by_c(f, testQ, A, coresets_dict, use_max=False)
        out_results(c_sizes, errors_dict, name_color_list, title, save_path2, show_plots, with_std)
    return


# balance part
def build_1_bal_SGD_coreset(A: torch.Tensor, C: torch.Tensor, U: torch.Tensor, f: Callable, trainQ: torch.Tensor,
                            epochs: int, bs: int, lr: float, valQ: torch.Tensor = None, solver: Callable = None) -> (
        torch.Tensor, torch.Tensor):
    print_epochs_info, print_batch_info = True, False  # print epochs\batches progress
    print('-' * 80)
    print('build_1_bal_SGD_coreset:')

    C_best, U_best = C.detach().clone(), U.detach().clone()  # faster than saving each improvment
    best_error, real_error, best_epoch = float("inf"), float("inf"), -1
    if is_trainable(C) and is_trainable(U):
        opt = torch.optim.Adam([C, U], lr=lr)
    else:
        opt = torch.optim.Adam([C], lr=lr)
    # print('\t\tearly stop: factor {}, patience {}, min lr {}'.format(fac, pat, min_lr))
    # opt = OptimizerHandler(optimizer, factor=fac, patience=pat, min_lr=min_lr)

    print('balancing:')
    bal_time = get_current_time()
    for e in range(1, epochs + 1):
        # print(var_to_string(C, '\t\tC', with_data=True))
        trainQ = shuffle_tensor(trainQ)
        error_cum_grad = 0.0
        ls_p_e, ls_c_e, errors_e = [], [], []  # per epoch - for debugging
        count_b = 0
        for i, q in enumerate(trainQ):
            count_b += 1
            loss_p = f(A, q)
            loss_c = f(C, q, U)
            error = torch.abs(1 - loss_c / loss_p)
            error_cum_grad += error

            ls_p_e.append(loss_p.item())
            ls_c_e.append(loss_c.item())
            errors_e.append(error.item())
            # ============ Backward ============
            last_batch = (i == (trainQ.shape[0] - 1))
            if (i + 1) % bs == 0 or last_batch:
                opt.zero_grad()
                error_cum_grad.backward()
                opt.step()
                with torch.no_grad():  # apply constraints:
                    for c_i in range(C.shape[0]):  # keep label entry as 1 or -1
                        C[c_i][-1] = torch.round(C[c_i][-1])
                        if C[c_i][-1] != -1 and C[c_i][-1] != 1:
                            print('labels must be -1 or 1')
                            print(np.round(C.tolist(), 2))
                            print(var_to_string(C, 'C', with_data=True))
                            print(var_to_string(U, 'U', with_data=True))
                            exit(77)
                    U.clamp_(min=0)  # all weights >=0
                error_cum_grad = 0
                if print_batch_info:
                    if i + 1 == bs:
                        print('\tepoch {}:'.format(e))
                    msg = '\t\tbatch [{}/{}] loss_c:{:,.8f}, loss_p:{:,.8f}, error:{:,.8f}'
                    print(msg.format(i + 1, trainQ.shape[0], np.mean(ls_c_e[-count_b:]), np.mean(ls_p_e[-count_b:]),
                                     np.mean(errors_e[-count_b:])))
                count_b = 0
        suffix = ''
        real_error = test_Q_1_coreset_mean(f, valQ, A, C, U)
        if real_error < best_error:
            C_best = C.detach().clone()
            U_best = U.detach().clone()
            best_error, best_epoch = real_error, e
            suffix = ' New best found'

        if print_epochs_info:
            msg = '\tepoch [{}/{}] loss_c:{:,.4f}({:,.3f}), loss_p:{:,.4f}({:,.3f})'
            msg += ', avg error:{:,.6f}({:,.3f}), valQ error:{:,.6f}'
            if solver:
                er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C, U)
                msg += ', weak error:{:,.6f}, erAqc:{:,.6f}, erAqa:{:,.6f}'.format(er, er_A_on_qc, er_A_on_qa)
            msg += ', lr:{:,.5f}' + suffix + ', time so far {}'.format(get_time_str(bal_time))
            print(msg.format(e, epochs, np.mean(ls_c_e), np.std(ls_c_e), np.mean(ls_p_e), np.std(ls_p_e),
                             np.mean(errors_e), np.std(errors_e), real_error, get_lr(opt)))

    print('Done balancing: epoch {},error={:,.15f}, run time {}'.format(best_epoch, best_error, get_time_str(bal_time)))
    print('-' * 80)
    return C_best, U_best


def build_bal_SGD_coresets(A: torch.Tensor, trainQ: torch.Tensor, f: callable, c_sizes: list, bal_folder: str,
                           epochs: int, bs: int, lr: float, valQ: torch.Tensor = None, solver: Callable = None):
    print('\nbuild_bal_SGD_coresets {} (DONE ONLY ONCE):'.format(A.shape))
    bal_path_base = bal_folder + '/bal_coreset_{}.pt'

    if valQ is None:
        valQ = trainQ
    print('|A|={}, |trainQ|={}, |valQ|={}, c_sizes={}'.format(t_size_str(A), trainQ.shape[0], valQ.shape[0], c_sizes))
    for c_size in c_sizes:
        bal_coreset_path = bal_path_base.format(c_size)
        if os.path.exists(bal_coreset_path):
            print('\t{} already exists'.format(bal_coreset_path))
        else:
            # C_init = subset_init(c_size, A, trainable=False)
            # U_init = add_cuda(torch.zeros(C_init.shape[0]) + A.shape[0] / C_init.shape[0])  # init U

            valid = False
            C_init, U_init = None, None
            random_sp = np.ones(A.shape[0])
            while not valid:  # select from class 0 and class 1 - invalid otherwise
                C_init, U_init = select_coreset(A, random_sp, c_size, out_torch=True)
                _, C_y = Log.a_to_x_y(C_init)
                s = torch.sum(C_y).item()
                valid = -c_size < s < c_size  # can't allow only -ones or only ones
                if not valid:
                    print('bad selection')

            C_init.requires_grad = True
            U_init.requires_grad = False
            total_batches = math.ceil(trainQ.shape[0] / bs)
            print('\nBalance params:')
            print('\tepochs {}, batch_size {}, total batches {}, base_lr {}'.format(epochs, bs, total_batches, lr))

            print('Tensors:')
            print(var_to_string(C_init, '\tC', with_data=True))
            print(var_to_string(U_init, '\tU', with_data=True))

            test_Q_1_coreset_mean(f, trainQ, A, C_init, U_init, title='test loss(A,C,U,Q_train):')
            test_Q_1_coreset_mean(f, valQ, A, C_init, U_init, title='test loss(A,C,U,Q_val):')
            if solver is not None:
                er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C_init, U_init)
                msg = '{}:f(A,lc):{:,.8f}, f(A,la):{:,.8f}, error:{:,.8f}'
                print(msg.format('Weak: pre balance', er_A_on_qc, er_A_on_qa, er))

            C, U = build_1_bal_SGD_coreset(A, C_init, U_init, f, trainQ, epochs, bs, lr, valQ=valQ, solver=solver)
            save_tensor({'C': C, 'U': U}, bal_coreset_path)
            print('\tTensors:')
            print(var_to_string(C, '\tC', with_data=False))
            print(var_to_string(U, '\tU', with_data=True))
            test_Q_1_coreset_mean(f, trainQ, A, C, U, title='test loss(A,C,U,Q_train):')
            test_Q_1_coreset_mean(f, valQ, A, C, U, title='test loss(A,C,U,Q_val):')
            if solver is not None:
                er, er_A_on_qc, er_A_on_qa = test_weak_1_coreset(f, solver, A, C, U)
                msg = '{}:f(A,lc):{:,.8f}, f(A,la):{:,.8f}, error:{:,.8f}'
                print(msg.format('Weak: post balance', er_A_on_qc, er_A_on_qa, er))
    return


def get_bal_coreset(bal_folder: str, c_size: int) -> (torch.Tensor, torch.Tensor):
    print('get_bal_coreset:')
    path = bal_folder + '/bal_coreset_{}.pt'.format(c_size)
    assert os.path.exists(path), '{} doesn\'t exist. call build_bal_SGD_coresets()'.format(path)
    save_dict = load_tensor(path, tabs=1)
    C, U = save_dict['C'], save_dict['U']
    C = add_cuda(C)
    U = add_cuda(U)

    # print(var_to_string(C, '\tC', with_data=True))
    # print(var_to_string(U, '\tU', with_data=True))
    return C, U


def main_balance(argv: dict, A: torch.Tensor):
    print('\nmain_balance:')

    ds, f, solver_f = argv['ds'], argv['f'], argv['solver_f']
    reps, c_sizes = argv['bal_reps'], argv['bal_c_sizes']
    bal_folder = argv['bal_coreset_folder']
    show_plots, save_plots = argv['bal_show_plots'], argv['bal_save_plots']
    epochs, bs, lr = argv['bal_params']
    pathQ, paramsQ = argv['Q_path'], argv['Q_params']
    trainQ_s, valQ_s, testQ_s = argv['bal_train_size'], argv['bal_val_size'], argv['bal_test_size']
    color_dict = argv['color_dict']

    sp_list, name_color_list = get_sp_list(A, group_names=['SGD', 'UNI', 'TMF'], color_dict=color_dict)
    trainQ, valQ, testQ = Log.get_Q_sets(A, pathQ, paramsQ, trainQ_s, valQ_s, testQ_s)

    title_testQ, desc_testQ = 'testQ', 'build Q'
    testQ_list = [(testQ, title_testQ, desc_testQ)]

    # testQ_list += handlerLogisticRegression.get_extra_testQ(A, plot_if_2d)

    # done once per A for every c_size - needs trainQ and valQ that are not in testQ
    build_bal_SGD_coresets(A, trainQ, f, c_sizes, bal_folder, epochs, bs, lr, valQ=valQ, solver=solver_f)

    coresets_dict = get_list_of_coresets_for_each_size(A, sp_list, c_sizes, reps, bal_folder=bal_folder)
    # no need for SP anymore

    # prepare title base
    names = ','.join([sp_name for (SP, sp_name, sp_color) in sp_list])
    title_base = 'problem: Logistic Regression, ds {}({})\n'.format(ds, t_size_str(A))
    title_base += '{} on {} (reps={})\n'.format('bal_test', names, reps)  # test type (sp test, balance test)
    title_base += '{}\n'  # test definition (strong, weak, q_opt)
    title_base += '{}'  # full description of testQ (how testQ was created)

    test_coresets(A, f, name_color_list, coresets_dict, testQ_list, c_sizes, reps, title_base, bal_folder, save_plots,
                  show_plots, with_std=True, solver_f=solver_f)
    return


def simple_sp(A: torch.Tensor, f: Callable, reps: int, c_sizes: list, argv, solver_f: Callable = None):
    print('\nsimple_sp:')
    uni = (np.ones(A.shape[0]), 'UNI', 'r')

    tmf = (load_tmf_sp(), 'TMF', 'g')
    sp_list = [uni, tmf]

    coresets_dict = get_list_of_coresets_for_each_size(A, sp_list, c_sizes, reps, bal_folder=None)
    name_color_list = [(sp_tuple[1], sp_tuple[2]) for sp_tuple in sp_list]  # SP vector is not needed anymore

    if solver_f is not None:
        errors_dict = test_weak_many_c_sizes(f, solver_f, A, coresets_dict)
        out_results(c_sizes, errors_dict, name_color_list, 'weak', save_path=None, show_plot=True, with_std=True)

        errors_dict = test_q_opt_many_c_sizes(f, solver_f, A, coresets_dict)
        out_results(c_sizes, errors_dict, name_color_list, 'q_opt', save_path=None, show_plot=True, with_std=True)

    pathQ, paramsQ = argv['Q_path'], argv['Q_params']
    trainQ_s, valQ_s, testQ_s = argv['bal_train_size'], argv['bal_val_size'], argv['bal_test_size']
    trainQ, valQ, testQ = Log.get_Q_sets(A, pathQ, paramsQ, trainQ_s, valQ_s, testQ_s)
    title_testQ, desc_testQ = 'testQ', 'build Q'
    testQ_list = [(testQ, title_testQ, desc_testQ)]

    for testQ_tuple in testQ_list:
        testQ, titleQ, descQ = testQ_tuple
        if isinstance(testQ, torch.Tensor):
            descQ_full = '|{}|={}, description: {}'.format(titleQ, t_size_str(testQ), descQ)
        else:
            descQ_full = '{} {}'.format(titleQ, descQ)

        title = '{}-{}'.format(STRONG_TITLE_MAX_MEAN, descQ_full)
        errors_dict = test_many_c_sizes_by_c(f, testQ, A, coresets_dict, use_max=True)
        out_results(c_sizes, errors_dict, name_color_list, title, save_path=None, show_plot=True, with_std=True)

    return


def temp_graph():
    """
    Murad and Alaa's results on the paper "Coresets for Near-Convex Functions"
    ds HTRU_2 weak logistic regression coreset
    sample_sizes = [50, 73, 97, 121, 144,
                    168, 192, 215, 239, 263,
                    286, 310, 334, 357, 381,
                    405, 428, 452, 476, 500]
    tmf_ers = [3.15228018, 2.13985204, 1.45572799, 1.00974639, 0.72948824,
               0.558439, 0.45415051, 0.38640284, 0.33536618, 0.28976275,
               0.24502862, 0.20147566, 0.16245338, 0.13251081, 0.11555839,
               0.11302986, 0.12204413, 0.13356717, 0.13057387, 0.08620995]
    tmf_std = [2.04898211, 1.33543832, 0.8722622, 0.580906, 0.40293832,
               0.29615751, 0.23124564, 0.18890384, 0.15741513, 0.13058596,
               0.10602236, 0.08370088, 0.06479819, 0.05074721, 0.04249023,
               0.03990318, 0.04136738, 0.04346779, 0.04079912, 0.02586298]
    @return:
    """

    c_sizes = [100, 200, 300, 400, 500]
    sgd_ers = [0.59185, 0.17396, 0.09078, 0.04582, 0.00629]
    sgd_std = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
    uni_ers = [1.91241, 0.78121, 0.43232, 0.28149, 0.24710]
    uni_std = [1.39308, 0.53928, 0.39579, 0.20123, 0.14548]

    full_sample_sizes = [50, 73, 97, 121, 144,
                         168, 192, 215, 239, 263,
                         286, 310, 334, 357, 381,
                         405, 428, 452, 476, 500]
    full_tmf_ers = [3.15228018, 2.13985204, 1.45572799, 1.00974639, 0.72948824,
                    0.558439, 0.45415051, 0.38640284, 0.33536618, 0.28976275,
                    0.24502862, 0.20147566, 0.16245338, 0.13251081, 0.11555839,
                    0.11302986, 0.12204413, 0.13356717, 0.13057387, 0.08620995]
    full_tmf_std = [2.04898211, 1.33543832, 0.8722622, 0.580906, 0.40293832,
                    0.29615751, 0.23124564, 0.18890384, 0.15741513, 0.13058596,
                    0.10602236, 0.08370088, 0.06479819, 0.05074721, 0.04249023,
                    0.03990318, 0.04136738, 0.04346779, 0.04079912, 0.02586298]
    index_of_97 = full_sample_sizes.index(97)
    index_of_192 = full_sample_sizes.index(192)
    index_of_310 = full_sample_sizes.index(310)
    index_of_405 = full_sample_sizes.index(405)
    index_of_500 = full_sample_sizes.index(500)

    tmf_ers = [
        full_tmf_ers[index_of_97],
        full_tmf_ers[index_of_192],
        full_tmf_ers[index_of_310],
        full_tmf_ers[index_of_405],
        full_tmf_ers[index_of_500]
    ]
    tmf_std = [
        full_tmf_std[index_of_97],
        full_tmf_std[index_of_192],
        full_tmf_std[index_of_310],
        full_tmf_std[index_of_405],
        full_tmf_std[index_of_500]
    ]

    name_color_list = [
        ('SGD', 'b'),
        ('UNI', 'r'),
        ('TMF', 'g')
    ]

    errors_dict = defaultdict(list)
    errors_dict[ERR_KEY.format('SGD')] = sgd_ers
    errors_dict[STD_KEY.format('SGD')] = sgd_std
    errors_dict[ERR_KEY.format('UNI')] = uni_ers
    errors_dict[STD_KEY.format('UNI')] = uni_std
    errors_dict[ERR_KEY.format('TMF')] = tmf_ers
    errors_dict[STD_KEY.format('TMF')] = tmf_std

    names = ','.join([sp_name for (sp_name, sp_color) in name_color_list])
    title = 'problem: Logistic Regression, ds {}({})\n'.format('HTRU_2', [17897, 9])
    title += '{} on {}\n'.format('bal_test', names)
    title += '{}\n'.format(WEAK_TITLE)

    out_results(c_sizes, errors_dict, name_color_list, title=title, save_path='./weak_TMF_UNI_SGD', show_plot=True,
                with_std=True)
    return


def get_argv(ds: str, n: int, d: int) -> dict:
    # noinspection PyDictCreation
    argv = {}

    argv['ds'] = ds
    base_f = argv['base_folder'] = './{}_{}_{}'.format(ds, n, d)  # e.g. diabetes_442_11/

    argv['f'] = Log.log_loss_opt  # loss function
    argv['solver_f'] = Log.log_solver_sklearn  # optional

    argv['Q_path'] = '{}/Q.pt'.format(base_f)
    argv['Q_params'] = (10, 1000, 1000, 0.001)  # learn Q |Q_init|, epochs, bs, lr
    argv['color_dict'] = {'SGD': 'b', 'TMF': 'g', 'UNI': 'r'}

    # BALANCE:
    argv['bal_coreset_folder'] = '{}/balance'.format(base_f)
    argv['bal_show_plots'], argv['bal_save_plots'] = False, True
    argv['bal_train_size'] = 8000
    argv['bal_val_size'] = 1600
    argv['bal_test_size'] = 200
    argv['bal_params'] = (1000, 100, 0.001)  # epochs, batch size, initial lr
    # argv['bal_c_sizes'], argv['bal_reps'] = [i for i in range(50, 451, 50)], 50
    argv['bal_c_sizes'], argv['bal_reps'] = [100, 200, 300, 400, 500], 100

    # argv['bal_moving_lr'] = (1, 10000, 0.000001)  # moving lr: (factor, patience, min_lr)

    for k, v in argv.items():
        print('\t{}: {}'.format(k, v))

    return argv


def main():
    """
    convention: y, w, q: all flattened
    """
    set_cuda_scope_and_seed(seed=42)
    ds = 'HTRU_2'  # (i) HTRU [19] — 17; 898 radio emissions of the Pulsar star each consisting of 9 features.
    A = get_data(ds, data_limit=0, two_d=False)
    Log.sanity_check(A, f_test=False, mes_f=False, solvers=True)

    argv = get_argv(ds, A.shape[0], A.shape[1])
    main_balance(argv, A)

    # c_sizes = [i for i in range(100, 1001, 100)]
    # simple_sp(A, f=argv['f'], reps=50, c_sizes=[100, 200, 300, 400, 500], argv=argv, solver_f=argv['solver_f'])
    return


if __name__ == "__main__":
    make_cuda_invisible()
    g_start_time = get_current_time()
    print("Python  Version {}".format(sys.version))
    print("PyTorch Version {}".format(torch.__version__))
    main()
    print('Total run time {}'.format(get_time_str(g_start_time)))
