import numpy as np
import sys
from sklearn.svm import SVC
import warnings

from DataReduction.SVM.d20210325.utils20200614 import get_current_time
from DataReduction.SVM.d20210325.utils20200614 import get_time_str
from DataReduction.SVM.d20210325.utils20200614 import select_coreset
from DataReduction.SVM.d20210325.utils20200614 import select_coreset_frequency

warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIM = 8


def de_augment_numpy(A):
    """
    :param A: nx(d+1) points with labels
    :return: X nxd and y nx1
    """
    X, y = A[:, :-1], A[:, -1]
    return X, y


def get_data(path):
    """
    data origin https://archive.ics.uci.edu/ml/datasets/HTRU2
    loading Murad's file
    it contains a dictionary with the following keys
    s.keys() = ['P_test', 'SP', 'P_train']
    """
    file_obj = np.load(path)
    A_train = file_obj['P_train']
    A_test = file_obj['P_test']
    SP_train = file_obj['SP']

    print('A_train  shape {}'.format(A_train.shape))
    print('A_test   shape {}'.format(A_test.shape))
    print('SP train shape {}'.format(SP_train.shape))
    return A_train, A_test, SP_train


def run_svm(A_train, A_test_x, w=None):
    clf = SVC(kernel='linear')

    X, y = de_augment_numpy(A_train)
    clf.fit(X, y, sample_weight=w)

    test_X, test_y = de_augment_numpy(A_test_x)
    score = clf.score(test_X, test_y)
    return score * 100


def choose_and_run(A_train, A_test, SP, c_size, c_reps, title, verbose=0):
    if verbose > 0:
        print('{}:'.format(title))
    cum_score = 0.0
    for i in range(c_reps):
        C, W = select_coreset(A_train, SP, c_size, out_torch=False)
        # TODO add if Cy is only one class - repeat selection
        # C, W = select_coreset_ben(A_train, SP, c_size)
        score_data = run_svm(C, A_test, W)
        if verbose > 0:
            print('\t{}: svm({} in size {:5}) = {:.4f}%'.format(i, title, c_size, score_data))
        cum_score += score_data
    cum_score /= c_reps
    return cum_score


def svm_soft_margin(A, q, weights=None):
    """
    :param A: of size nx(d+1). A=X|y where X is nxd and y is 1x1
    :param q: of size 1x(d+1). q=w|b where w is 1xd and b is 1x1
    :param w:  weights for each X if A is weighted
    :return: the Euclidean 2-norm || y - X q ||^2
    """
    # if w is not None:
    #     A_n = 1 if len(A.shape) == 1 else A.shape[0]
    #     assert w.shape[0] == A_n
    #     A = (A.T * np.sqrt(w)).T

    X, y = de_augment_numpy(A)
    n, d = X.shape
    assert d == DATA_DIM and y.shape[0] == n and q.shape == (1, DATA_DIM + 1)

    w, b = de_augment_numpy(q)
    assert w.shape == (1, DATA_DIM) and len(b) == 1
    import wizzi_utils as wu
    print(wu.to_str(X, 'X'))
    print(wu.to_str(y, 'y'))
    print(wu.to_str(q, 'q'))
    print(wu.to_str(w, 'w'))
    print(wu.to_str(b, 'b'))
    reg = 0.5 * np.linalg.norm(w) ** 2  # 0.5 * ||w||^2
    hinge = np.maximum(0, 1 - np.multiply(y, X.dot(w.T).flatten() + b))  # [sum(max(0,1-yi(w*xi - b)]
    loss = reg + np.sum(hinge)
    # reg = 1 / n * 0.5 * np.linalg.norm(w) ** 2 * np.sum(weights)  # 0.5 * ||w||^2
    # [sum(max(0,1-yi(w*xi - b)]
    # hinge = np.multiply(np.maximum(0, 1 - np.multiply(y, X.dot(w.T).flatten() + b)),weights)
    # loss = reg + np.sum(hinge)

    # BEN:
    # MAYBE reg: reg *= np.sum(weights) / |A|
    # loss = np.sum(weights * hinge) + reg

    return loss


# def svm_soft_margin(A, q, w=None):
#     """
#     :param A: of size nxd+1. A=X|y where X is nxd and y is 1x1
#     :param q: of size dx1
#     :param w:  weights for each X if A is weighted
#     :return: the Euclidean 2-norm || y - X q ||^2
#     """
#     if w is not None:
#         A_n = 1 if len(A.shape) == 1 else A.shape[0]
#         assert w.shape[0] == A_n
#         A = (A.t() * w.sqrt()).t()
#
#     X, y = de_augment(A)
#     n, d = X.shape
#     assert d == DATA_DIM and y.shape[1] == 1 and q.shape == (DATA_DIM, K)
#
#     loss = (X @ q - y).norm() ** 2
#     return loss

def temp(A_train, A_test_x, w=None):
    clf = SVC(kernel='linear')

    X, y = de_augment_numpy(A_train)
    clf.fit(X, y, sample_weight=w)

    test_X, test_y = de_augment_numpy(A_test_x)
    score = clf.score(test_X, test_y)
    print('\tsvm(P in size {:5}) = {:.4f}%'.format(A_train.shape[0], score))

    w = clf.coef_
    b = clf.intercept_
    print('X', type(X), X.shape)
    print('y', type(y), y.shape)
    print('w', type(w), w.shape, w.tolist())
    print('b', type(b), b.shape, b)

    reg = 0.5 * np.linalg.norm(w) ** 2  # ||w||_2^2 /2
    hinge = np.maximum(0, 1 - np.multiply(y, X.dot(w.T).flatten() + b))  #
    loss = reg + np.sum(hinge)
    print(loss)
    print(svm_soft_margin(A_train, np.column_stack((w, b))))
    exit(11)


def get_argv():
    # noinspection PyDictCreation
    argv = {}
    argv['seed'] = 42
    argv['file'] = './Gilad.npz'
    # argv['c_sizes'] = [500 * i for i in range(1, int(16107/500), 1)]
    argv['c_sizes'] = [75 * i for i in range(1, 11, 1)]
    argv['c_sizes'] = [70]
    argv['c_reps'] = 50
    # Summary(50 reps):
    # svm(P in size 16107) = 98.4916%
    # svm(C in size   500) = 98.2849%
    # svm(U in size   500) = 98.1966%
    # argv['c_sizes'] = [75 * i for i in range(1, 11, 1)] - Murad always beats uniform

    argv['sens_load_or_save_path_sp'] = './sens_442_cuda_iris_lr1.npy'
    argv['sens_load_or_save_path_X'] = './x_442_cuda_iris_lr1.npy'
    argv['sen_lr'] = 1.0
    argv['sen_epochs'] = 200
    argv['sen_x_shape'] = (8, 1)  # q shape is (d, k), x is q
    argv['sen_verbose'] = 0
    argv['show_SP_hist'] = False
    return argv


def main():
    g_start_time = get_current_time()
    from DataReduction.SVM.d20210325.utils20200614 import make_cuda_invisible
    make_cuda_invisible()
    argv = get_argv()
    np.random.seed(argv['seed'])
    print("Python  Version {}".format(sys.version))
    for k, v in argv.items():
        print('\t{}: {}'.format(k, v))

    A_train, A_test, SP_train = get_data(argv['file'])
    c_sizes, c_reps = argv['c_sizes'], argv['c_reps']
    n = A_train.shape[0]

    # temp(A_train, A_test)

    score_full_data = run_svm(A_train, A_test)

    for c_size in c_sizes:
        score_C_murad = choose_and_run(A_train, A_test, SP_train, c_size, c_reps, 'C_murad')

        SP_uniform = np.ones(shape=n)
        score_U = choose_and_run(A_train, A_test, SP_uniform, c_size, c_reps, 'Uniform')

        print('Summary:')
        print('\tsvm(P in size {:5}) = {:.4f}%'.format(n, score_full_data))
        print('\tsvm(C in size {:5}) = {:.4f}%'.format(c_size, score_C_murad))
        print('\tsvm(U in size {:5}) = {:.4f}%'.format(c_size, score_U))
        print('Total run time {}'.format(get_time_str(g_start_time)))
    return


if __name__ == "__main__":
    main()
