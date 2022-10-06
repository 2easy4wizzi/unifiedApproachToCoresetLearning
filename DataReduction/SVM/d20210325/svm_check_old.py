import numpy as np
import sys
from sklearn.svm import SVC
import warnings

from DataReduction.SVM.d20210325.utils20200215 import get_current_time
from DataReduction.SVM.d20210325.utils20200215 import get_time_str
from DataReduction.SVM.d20210325.utils20200215 import set_cuda_scope_and_seed
from DataReduction.SVM.d20210325.utils20200215 import init_logger
from DataReduction.SVM.d20210325.utils20200215 import del_logger
from DataReduction.SVM.d20210325.utils20200215 import log_print
from DataReduction.SVM.d20210325.utils20200215 import log_print_dict

warnings.simplefilter(action='ignore', category=FutureWarning)


# self.clf.score(self.P_test[:, :-1], self.P_test[:, -1])) * 100.0
def split_data_to_x_y(data, column=8):
    data_x = data[:, :column]
    data_y = data[:, column]
    return data_x, data_y


def get_data(path):
    # s.keys() = ['P_test', 'SP', 'P_train']
    file_obj = np.load(path)
    P_train = file_obj['P_train']
    P_test = file_obj['P_test']
    SP_train = file_obj['SP']

    P_train_x, P_train_y = split_data_to_x_y(P_train)
    P_test_x, P_test_y = split_data_to_x_y(P_test)
    # print(P_train[0])
    # print(P_train_x[0])
    # print(P_train_y[0])

    log_print('trainset: x shape {}, y shape {}'.format(P_train_x.shape, P_train_y.shape))
    log_print('SP train    shape {}'.format(SP_train.shape))
    log_print('testset : x shape {}, y shape {}'.format(P_test_x.shape, P_test_y.shape))

    return P_train_x, P_train_y, P_test_x, P_test_y, SP_train


def run_svm(train_x, train_y, test_x, test_y, w=None):
    clf = SVC(kernel='linear')
    if w is not None:
        clf.fit(train_x, train_y, sample_weight=w)
    else:
        clf.fit(train_x, train_y)
    score = clf.score(test_x, test_y)
    return score * 100


def choose_and_run(SP, c_size, P_train_x, P_train_y, P_test_x, P_test_y, k, title):
    log_print('{}:'.format(title))
    cum_score = 0.0
    for i in range(k):
        c_ind, data_x, data_y = select_coreset(P_train_x, P_train_y, SP, c_size)
        w = get_weights(SP, c_ind)
        score_data = run_svm(data_x, data_y, P_test_x, P_test_y, w)
        log_print('\t{}: svm({} in size {:5}) = {:.4f}%'.format(i, title, c_size, score_data))
        cum_score += score_data
    cum_score /= k
    return cum_score


def get_weights(SP, C_ind):
    sp_sum = np.sum(SP)
    len_c = len(C_ind)
    weights_C = np.ones(len_c)
    for i, c_ind in enumerate(C_ind):
        weights_C[i] = sp_sum / (len_c * SP[c_ind])
    return weights_C


def select_coreset(data_x, data_y, SP, coreset_size):
    sp_normed = SP / SP.sum()
    # replace=False -> unique values, True with reps
    C_ind = np.random.choice(a=range(0, len(data_x)), size=coreset_size, p=sp_normed, replace=True)
    C_x = data_x[C_ind]
    C_y = data_y[C_ind]
    return C_ind, C_x, C_y


def get_argv():
    # noinspection PyDictCreation
    argv = {}
    argv['seed'] = 42
    argv['file'] = './Gilad.npz'
    argv['c_size'] = 500
    argv['k'] = 50

    return argv


def main():
    g_start_time = get_current_time()
    argv = get_argv()
    set_cuda_scope_and_seed(argv['seed'])
    init_logger('svm_check_old.txt')  # logger
    log_print("Python  Version {}".format(sys.version))
    log_print_dict(argv)

    P_train_x, P_train_y, P_test_x, P_test_y, SP_train = get_data(argv['file'])
    c_size, k = argv['c_size'], argv['k']
    n, d = P_train_x.shape

    score_full_data = run_svm(P_train_x, P_train_y, P_test_x, P_test_y)
    score_C_murad = choose_and_run(SP_train, c_size, P_train_x, P_train_y, P_test_x, P_test_y, k, 'C_murad')

    SP_uniform = np.ones(shape=n)
    score_U = choose_and_run(SP_uniform, c_size, P_train_x, P_train_y, P_test_x, P_test_y, k, 'Uniform')

    log_print('Summary:')
    log_print('\tsvm(P in size {:5}) = {:.4f}%'.format(n, score_full_data))
    log_print('\tsvm(C in size {:5}) = {:.4f}%'.format(c_size, score_C_murad))
    log_print('\tsvm(U in size {:5}) = {:.4f}%'.format(c_size, score_U))

    # TODO add different sizes
    log_print('Total run time {}'.format(get_time_str(g_start_time)))
    del_logger()
    return


if __name__ == "__main__":
    main()

# def run_svm(train_x_svm_ready, train_y_svm_ready, test_x_svm_ready, test_y_svm_ready, should_print=True):
#     if should_print:
#         print("Running SVM...")
#     clf = SVC(cache_size=7000)
#     clf.fit(train_x_svm_ready, train_y_svm_ready)
#     # specs = clf.fit(train_x_svm_ready, train_y_svm_ready)
#     # print("SVM info:")
#     # print("  {}".format(specs))
#     y_pred = clf.predict(test_x_svm_ready)
#     # score = clf.score(test_x_svm_ready, test_y_svm_ready)
#     score, prec, recall, fscore_w = print_info_of_predict(test_y_svm_ready, y_pred, should_print)
#     return score, prec, recall, fscore_w
