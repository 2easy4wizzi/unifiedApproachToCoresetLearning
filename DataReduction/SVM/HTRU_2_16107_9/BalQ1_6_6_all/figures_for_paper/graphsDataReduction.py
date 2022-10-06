from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

ERR_KEY = '{}_err_list'
STD_KEY = '{}_std_list'


def plot_x_y_std(data_x: np.array, groups: list, title: str = None, x_label: str = 'Coreset size',
                 y_label: str = 'Error', save_path: str = None, show_plot: bool = True, with_shift: bool = False,
                 zoom_group: list = None):
    """
    data_x: x values
    groups: list of groups s.t. each tuple(y values, y std, color, title)  y std could be None
    example:
        data_x = [10, 20, 30]
        C_errors = [5, 7, 1]
        C_errors_stds = [2, 1, 0.5]
        group_c = (C_errors, C_errors_stds, 'g', 'C')
        U_errors = [10, 8, 3]
        U_errors_vars = [4, 3, 1.5]
        group_u = (U_errors, U_errors_vars, 'r', 'U')
        groups = [group_c, group_u]
        title = 'bla'
        plot_x_y_std(data_x, groups, title)
    :return:
    """
    data_x_last = data_x  # in order to see all STDs, move a little on the x axis
    data_x_jump = 0.5
    data_x_offset = - int(len(groups) / 2) * data_x_jump
    line_style = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 2, "elinewidth": 1, "capsize": 4}
    for i, group in enumerate(groups):
        data_y, std_y = group[0], group[1]  # std_y could be None
        color, label = group[2], group[3]
        if with_shift:  # move x data for each set a bit so you can see it clearly
            dx_shift = [x + i * data_x_jump + data_x_offset for x in data_x]
            data_x_last = dx_shift
        plt.errorbar(data_x_last, data_y, std_y, color=color, fmt='.', label=label, **line_style)

    plt.grid()

    plt.legend(loc='upper right', ncol=1, fancybox=True, framealpha=0.5)
    if title is not None:
        plt.title(title, y=-0.18)
    plt.xticks(data_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylabel(r'$Err_{avg}$')
    # plt.ylabel(r'$Err_{opt}$')

    if zoom_group is not None:
        # plt.legend(loc='upper left', ncol=1, fancybox=True, framealpha=0.5)
        data_y, std_y = zoom_group[0], zoom_group[1]  # std_y could be None
        color, label = zoom_group[2], zoom_group[3]
        # plot_in = {  # top right
        #     # plot size is A on B (e.g. 200 x 100)
        #     'start_x': 0.58,  # inner plot x axis start from 'start_x' * A (e.g. 200 * 'start_x')
        #     'start_y': 0.55,  # inner plot y axis start from 'start_y' * B (e.g. 100 * 'start_y')
        #     'x_size': 0.3,  # inner plot x size will be 'x_size' * A (e.g. 200 * 'x_size')
        #     'y_size': 0.3  # inner plot y size will be 'y_size' * B (e.g. 100 * 'y_size')
        # }
        plot_in = {  # on the left
            # plot size is A on B (e.g. 200 x 100)
            'start_x': -0.4,  # inner plot x axis start from 'start_x' * A (e.g. 200 * 'start_x')
            'start_y': 0.4,  # inner plot y axis start from 'start_y' * B (e.g. 100 * 'start_y')
            'x_size': 0.4,  # inner plot x size will be 'x_size' * A (e.g. 200 * 'x_size')
            'y_size': 0.4  # inner plot y size will be 'y_size' * B (e.g. 100 * 'y_size')
        }

        plt.axes(list(plot_in.values()), facecolor='lightgrey')
        plt.errorbar(data_x_last, data_y, std_y, color=color, fmt='.', label=label, **line_style)
        # plt.title('{} zoom in'.format(label))
        # plt.ylim(0, max(data_y) * 1.1)

        xticks_inner = [data_x[i] for i in range(0, len(data_x), 2)]
        plt.xticks(xticks_inner)

        # y_ticks = list(plt.yticks()[0])
        # yticks_inner = [y_ticks[i] for i in range(0, len(y_ticks), 2)]
        # plt.yticks(yticks_inner)  # 3 values from the original xticks
        # plt.yticks([y_ticks[1], 0.0, y_ticks[-2]])  # 3 values from the original yticks (start from 0)
        # plt.yticks([])

    # dt = 0.001
    # t = np.arange(0.0, 10.0, dt)
    # r = np.exp(-t[:1000] / 0.05)  # impulse response
    # x = np.random.randn(len(t))
    # s = np.convolve(x, r)[:len(x)] * dt  # colored noise
    # plt.axes([.65, .6, .2, .2])
    # plt.hist(s, 400)
    # plt.title('Zoom')
    # plt.xticks([])
    # plt.yticks([])

    # plt.axes([0.2, 0.6, .2, .2], facecolor='y')
    # plt.plot(t[:len(r)], r)
    # plt.title('Impulse response')
    # plt.xlim(0, 0.2)
    # plt.xticks([])
    # plt.yticks([])

    # plt.ylim((0, 1))
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print('\tsaved to {}.png'.format(save_path))
    plt.pause(0.0001)
    if show_plot:
        plt.show(block=True)
    plt.cla()
    plt.close()
    return


def out_results(c_sizes: list, errors_dict: dict, name_color_list: list, title: str, save_path: str = None,
                show_plot: bool = True, with_std: bool = False, with_zoomed_name: str = None):
    """
    coresets_dict: key=c_size, value=coresets_dict_per_size
    coresets_dict_per_size: key=name(e.g. SGD, UNI), value=coreset_list
    coreset_list: [(C,U), ... repetitions ...]
    :returns error dict:
    assume 'SGD' is the only group
    key1 = ERR_KEY.format('SGD'), value=[ value for c_size[0], ... , value for c_size[-1] ]
    key1 = STD_KEY.format('SGD'), value=[ value for c_size[0], ... , value for c_size[-1] ]
    """
    print('{}:'.format(save_path))
    groups = []
    zoom_group = None
    for name, color in name_color_list:
        ers = errors_dict[ERR_KEY.format(name)]  # must exist
        stds = errors_dict[STD_KEY.format(name)] if with_std else None
        group = (ers, stds, color, name)
        msg = '\t{} errors: {}'.format(name, ers)
        if stds is not None:
            msg += ', stds: {}'.format(stds)
        print(msg)
        groups.append(group)
        if with_zoomed_name is not None and name == with_zoomed_name:
            zoom_group = group

    if len(groups) > 0:
        plot_x_y_std(c_sizes, groups, title, save_path=save_path, show_plot=show_plot, with_shift=True,
                     zoom_group=zoom_group)
    return


def plot_SVM_weak(name_color_dict: dict, title: str = '', show_plot: bool = True, solo: bool = False,
                  with_zoomed: bool = False):
    """
    weak test on SGD,SVD,UNI (other reps=100, bal reps=10)
    SGD errors: [0.00769,0.01968,0.00606,0.01288,0.00951,0.00492,0.00521,0.00613,0.00763,0.00464],
        stds: [0.00692,0.02746,0.00735,0.01310,0.01036,0.00548,0.00741,0.00570,0.00830,0.00502]
    SVD errors: [0.04731,0.04279,0.03557,0.03239,0.02368,0.02013,0.01898,0.01576,0.01428,0.01691],
        stds: [0.05054,0.03530,0.03684,0.02750,0.02041,0.01689,0.01595,0.01262,0.01126,0.01605]
    UNI errors: [0.05304,0.04265,0.04006,0.03749,0.03121,0.02947,0.02635,0.02554,0.02183,0.02143],
        stds: [0.05611,0.03457,0.03410,0.03669,0.02793,0.02437,0.02140,0.01890,0.01960,0.01759]
    saved to ./3DRoad_434874_3/bal2/50_to_140_weak.png
    weak_time 00:00:06
    """
    c_sizes = [800, 1000, 1200, 1400, 1600, 1800]

    sgd_ers = [0.04340, 0.03092, 0.02688, 0.02526, 0.02171, 0.01906]
    sgd_std = [0.02508, 0.00751, 0.01245, 0.01002, 0.00434, 0.00778]

    murad_ers = [0.04407, 0.03729, 0.03055, 0.02392, 0.02226, 0.01992]
    murad_std = [0.02292, 0.02211, 0.01676, 0.01398, 0.01253, 0.01211]

    uni_ers = [0.12157, 0.08968, 0.07521, 0.05462, 0.05047, 0.04520]
    uni_std = [0.09520, 0.05498, 0.04834, 0.04051, 0.03364, 0.02581]

    if solo:
        name_color_list = [(name_color_dict['OurCoreset']['name'], name_color_dict['OurCoreset']['color'])]
        save_path = 'SVM_weak_solo'
    else:
        name_color_list = [
            (name_color_dict['OurCoreset']['name'], name_color_dict['OurCoreset']['color']),
            (name_color_dict['Uniform']['name'], name_color_dict['Uniform']['color']),
            (name_color_dict['Competitor_SVM']['name'], name_color_dict['Competitor_SVM']['color']),
        ]
        save_path = 'SVM_weak'

    errors_dict = defaultdict(list)
    errors_dict[ERR_KEY.format(name_color_dict['OurCoreset']['name'])] = sgd_ers
    errors_dict[STD_KEY.format(name_color_dict['OurCoreset']['name'])] = sgd_std
    errors_dict[ERR_KEY.format(name_color_dict['Uniform']['name'])] = uni_ers
    errors_dict[STD_KEY.format(name_color_dict['Uniform']['name'])] = uni_std
    errors_dict[ERR_KEY.format(name_color_dict['Competitor_SVM']['name'])] = murad_ers
    errors_dict[STD_KEY.format(name_color_dict['Competitor_SVM']['name'])] = murad_std

    with_zoomed_name = name_color_dict['OurCoreset']['name'] if with_zoomed else None

    out_results(c_sizes,
                errors_dict,
                name_color_list,
                title=title,
                save_path=save_path,
                show_plot=show_plot,
                with_std=True,
                with_zoomed_name=with_zoomed_name)
    return


def plot_SVM_strong_mean_mean(name_color_dict: dict, title: str = '', show_plot: bool = True, solo: bool = False,
                              with_zoomed: bool = False):
    """
    strong test(|testQ|=[2000, 3, 1]) on SGD,SVD,UNI (other reps=100, bal reps=10)
    SGD errors: [0.00004,0.00013,0.00004,0.00012,0.00007,0.00006,0.00004,0.00006,0.00006,0.00006],
        stds: [0.00001,0.00024,0.00002,0.00014,0.00002,0.00002,0.00002,0.00002,0.00002,0.00001]
    SVD errors: [0.09303,0.08500,0.08710,0.07821,0.07511,0.06983,0.06798,0.07090,0.05495,0.06200],
        stds: [0.04542,0.04292,0.04883,0.03880,0.03714,0.03615,0.03344,0.03434,0.02577,0.03622]
    UNI errors: [0.15692,0.14224,0.13588,0.12738,0.11501,0.11691,0.11024,0.11149,0.11293,0.10120],
        stds: [0.11449,0.08501,0.08841,0.09121,0.08008,0.07111,0.08016,0.07506,0.07079,0.06286]
    saved to ./3DRoad_434874_3/bal2/50_to_140_strong_mean_mean.png
    st_mean_time 02:03:58
    """
    c_sizes = [800, 1000, 1200, 1400, 1600, 1800]

    sgd_ers = [0.00033, 0.00029, 0.00026, 0.00023, 0.00022, 0.00022]
    sgd_std = [0.00006, 0.00005, 0.00005, 0.00002, 0.00002, 0.00003]

    msf_ers = [0.08350, 0.08295, 0.07790, 0.06852, 0.06396, 0.06458]
    msf_std = [0.05779, 0.05274, 0.05280, 0.05328, 0.04480, 0.04540]

    uni_ers = [0.14688, 0.14891, 0.11316, 0.10264, 0.10424, 0.09867]
    uni_std = [0.08687, 0.10653, 0.07824, 0.06550, 0.06916, 0.06556]

    if solo:
        name_color_list = [(name_color_dict['OurCoreset']['name'], name_color_dict['OurCoreset']['color'])]
        save_path = 'SVM_strong_mean_mean_solo'
    else:
        name_color_list = [
            (name_color_dict['OurCoreset']['name'], name_color_dict['OurCoreset']['color']),
            (name_color_dict['Uniform']['name'], name_color_dict['Uniform']['color']),
            (name_color_dict['Competitor_SVM']['name'], name_color_dict['Competitor_SVM']['color']),
        ]
        save_path = 'SVM_strong_mean_mean'

    errors_dict = defaultdict(list)
    errors_dict[ERR_KEY.format(name_color_dict['OurCoreset']['name'])] = sgd_ers
    errors_dict[STD_KEY.format(name_color_dict['OurCoreset']['name'])] = sgd_std
    errors_dict[ERR_KEY.format(name_color_dict['Uniform']['name'])] = uni_ers
    errors_dict[STD_KEY.format(name_color_dict['Uniform']['name'])] = uni_std
    errors_dict[ERR_KEY.format(name_color_dict['Competitor_SVM']['name'])] = msf_ers
    errors_dict[STD_KEY.format(name_color_dict['Competitor_SVM']['name'])] = msf_std

    with_zoomed_name = name_color_dict['OurCoreset']['name'] if with_zoomed else None

    out_results(c_sizes,
                errors_dict,
                name_color_list,
                title=title,
                save_path=save_path,
                show_plot=show_plot,
                with_std=True,
                with_zoomed_name=with_zoomed_name)
    return


def main():
    name_color = {
        'OurCoreset': {
            'name': 'Learnable Coreset',
            'color': 'blue'
        },
        'Uniform': {
            'name': 'Uniform Sampling',
            'color': 'red'
        },
        'Competitor_SVM': {
            # 'name': 'LMS-Coreset',
            'name': 'tmf',
            'color': 'green'
        },
    }
    show_plot = False
    plot_SVM_weak(name_color, title='a', show_plot=show_plot, solo=False, with_zoomed=False)
    plot_SVM_strong_mean_mean(name_color, title='b', show_plot=show_plot, solo=False, with_zoomed=True)

    return


if __name__ == "__main__":
    main()
