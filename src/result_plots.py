"""
author: Florian Krach
"""

import numpy as np
import settings
import pandas as pd
import matplotlib.transforms as tr


if settings.euler:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def plot_JLSWGN_toy():
    path = '/Users/Flo/Desktop/Training results/JLSWGN_toy/2/'
    param_dict = settings.JLSWGN_toy_param_dict3_0
    input_dim = 128
    batch_size = 50
    n_features_first = 256
    learning_rate = 1e-2
    init_method = 'He'
    BN = True
    JL_dim = 360
    JL_error = 0.32
    target_number_samples = 10000
    architecture1 = 'SWGN'
    architecture2 = 'JLSWGN'


    # f = plt.figure()
    # f.subplots(nrows=8, ncols=2)
    # todo: put into subplot

    for N in param_dict['target_dim']:
        for sigma in param_dict['target_sigma']:
            for n_proj in param_dict['n_projections']:
                file_name1 = path + str(input_dim) + '_' + str(batch_size) + '_' + str(n_features_first) + '_' + \
                             str(learning_rate) + '_' + \
                             str(architecture1) + '_' + str(init_method) + '_' + str(BN) + '_' + str(None) + '_' + \
                             str(None) + '_' + \
                             str(n_proj) + '_' + str(N) + '_' + str(target_number_samples) + '_' + \
                             str(sigma) + '/' + 'training_progress.csv'
                file_name2 = path + str(input_dim) + '_' + str(batch_size) + '_' + str(n_features_first) + '_' + \
                             str(learning_rate) + '_' + \
                             str(architecture2) + '_' + str(init_method) + '_' + str(BN) + '_' + str(JL_dim) + '_' + \
                             str(JL_error) + '_' + \
                             str(n_proj) + '_' + str(N) + '_' + str(target_number_samples) + '_' + \
                             str(sigma) + '/' + 'training_progress.csv'

                table1 = pd.read_csv(filepath_or_buffer=file_name1, index_col=0, header=0)
                table2 = pd.read_csv(filepath_or_buffer=file_name2, index_col=0, header=0)

                SWD = False
                ylab = 'W2 approximation'
                if 'SWD' in table1.columns:
                    SWD = True
                    ylab += ', SWD'

                f = plt.figure()
                plt.semilogy(np.cumsum(table1['time_for_epochs']), table1['W2_loss_approximation'], 'ro-', label='SWGN W2 appr')
                plt.semilogy(np.cumsum(table2['time_for_epochs']), table2['W2_loss_approximation'], 'bo-', label='JLSWGN W2 appr')
                if SWD:
                    plt.semilogy(np.cumsum(table1['time_for_epochs']), table1['SWD'], 'mo-', label='SWGN SWD')
                    plt.semilogy(np.cumsum(table2['time_for_epochs']), table2['SWD'], 'co-', label='JLSWGN SWD')
                plt.legend()
                plt.title('N={}, sigma={}, numb_proj={}'.format(N, sigma, n_proj))
                plt.xlabel('time (s)')
                plt.ylabel(ylab)
                f.savefig(path+'plot_{}_{}_{}.png'.format(N, n_proj, sigma), bbox_inches="tight")
                plt.close(f)


def plot_JLSWGN_toy2(param_dict=settings.JLSWGN_toy_param_dict3_0,
                     path='/Users/Flo/Desktop/Training results/JLSWGN_toy/',
                     plot_style='o-'):
    input_dim = param_dict['input_dim'][0]
    n_features_first = param_dict['n_features_first'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    batch_size = param_dict['batch_size'][0]
    BN = param_dict['BN'][0]
    target_number_samples = param_dict['target_number_samples'][0]
    k = 1

    for target_dim in param_dict['target_dim']:
        for target_sigma in param_dict['target_sigma']:
            for n_projections in param_dict['n_projections']:
                data = []
                for i, JL_dim in enumerate([None, 360]):
                    for run in param_dict['run']:
                        if run is None:
                            dir1 = path
                        else:
                            dir1 = path + '{}/'.format(run)
                        data.append([])
                        if JL_dim is None:
                            JL_error = None
                            architecture = 'SWGN'
                        else:
                            JL_error = 0.32
                            architecture = 'JLSWGN'

                        file =dir1 + str(input_dim) + '_' + str(batch_size) + '_' + str(n_features_first) + '_' + \
                              str(learning_rate) + '_' + \
                              str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(JL_dim) + '_' + \
                              str(JL_error) + '_' + \
                              str(n_projections) + '_' + str(target_dim) + '_' + str(target_number_samples) + '_' + \
                              str(target_sigma) + '/' + 'training_progress.csv'
                        table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0)
                        try:
                            data[-1].append(table['time_for_epochs'].values)
                        except KeyError:
                            data[-1].append(table['time_for_iterations'].values)
                        data[-1].append(table['SWD'].values)
                        name = architecture
                        if run is not None:
                            name += '_{}'.format(run)
                        data[-1].append(name)

                f = plt.figure()
                for i in range(len(data)):
                    run = int(data[i][2][-1])
                    if data[i][2][:4] == 'SWGN':
                        color = (1-0.15*(run-1), 0, 0, 0.5)
                    else:
                        color = (0, 1-0.15*(run-1), 0, 0.5)
                    plt.semilogy(np.cumsum(data[i][0][:]), data[i][1][:], plot_style, label=data[i][2], color=color)
                    print 'total trainings time {}: {}'.format(data[i][2], np.cumsum(data[i][0][:])[-1])
                plt.legend()
                plt.title('n={}, L={}'.format(target_dim, n_projections))
                plt.xlabel('time (s)')
                plt.ylabel('SWD')
                f.savefig(path + 'plot_{}_{}_{}.png'.format(target_dim, n_projections, target_sigma),
                          bbox_inches="tight")
                f.savefig(path + 'plot_toy_{}.png'.format(k), bbox_inches="tight")
                k += 1
                plt.close(f)


def plot_JLSWGN_mnist(param_dict=settings.JLSWGN_MNIST_param_dict1_0,
                      path='/Users/Flo/Desktop/Training results/JLSWGN_mnist/',
                      plot_style='o-', last1=100, last2=100):
    input_dim = 128
    batch_size = 250
    n_features_first = 256
    n_features_reduction_factor = 2
    min_features = 64
    learning_rate = 5e-4
    init_method = 'He'
    BN = False
    image_enlarge_method = 'zoom'
    order = 1
    n_projections = 10000

    for power in param_dict['power']:
        size = 2**power
        data = []
        for i, JL_dim in enumerate(param_dict['JL_dim']):
            data.append([])
            if JL_dim is None:
                JL_error = None
                architecture = 'SWGN'
            else:
                JL_error = np.round(np.sqrt(8 * np.log(2 * batch_size) / JL_dim), decimals=4)
                architecture = 'JLSWGN'

            file = path + str(size) + '_' + str(image_enlarge_method) + '_' + str(order) + '_' + \
                str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                str(n_features_first) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + \
                str(JL_dim) + '_' + str(JL_error) + '_' + \
                str(n_projections) + '/' + 'training_progress.csv'
            table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0)
            data[i].append(table['time_for_iterations'].values)
            data[i].append(table['SWD_approximation'].values)
            name = architecture
            if name == 'JLSWGN':
                name += str(JL_dim)
            data[i].append(name)

        f = plt.figure()
        for i in range(len(data)):
            if data[i][2] == 'SWGN':
                last = last1
            else:
                last = last2
            plt.semilogy(np.cumsum(data[i][0][:last]), data[i][1][:last], plot_style, label=data[i][2])
        plt.legend()
        plt.title(r'image size = ${} \times {}$'.format(size, size))
        plt.xlabel('time (s)')
        plt.ylabel('SWD')
        f.savefig(path + 'plot_MNIST_{}_{}_{}.png'.format(size, last1, last2),
                  bbox_inches=tr.Bbox.from_bounds(-0.3,0,6.5,4.7))
        plt.close(f)


def plot_JLSWGN(param_dict=settings.JLSWGN_param_dict1_0, path='/Users/Flo/Desktop/Training results/JLSWGN/cifar10/',
                last1=50, last2=50, plot_style='o-', BN=True):
    input_dim = param_dict['input_dim'][0]
    n_features_first = param_dict['n_features_first'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    dataset = param_dict['data'][0]

    for batch_size in param_dict['batch_size']:
        for n_projections in param_dict['n_projections']:
            data = []
            for i, JL_dim in enumerate(param_dict['JL_dim']):
                data.append([])
                if JL_dim is None:
                    JL_error = None
                    architecture = 'SWGN'
                else:
                    JL_error = np.round(np.sqrt(8 * np.log(2 * batch_size) / JL_dim), decimals=4)
                    architecture = 'JLSWGN'

                file = path + str(dataset)+'_'+str(input_dim)+'_'+str(batch_size)+'_'+str(n_features_first)+'_'+\
                    str(learning_rate)+'_'+str(n_features_reduction_factor)+'_'+\
                    str(architecture)+'_'+str(init_method)+'_'+str(BN)+'_'+str(JL_dim)+'_'+str(JL_error)+'_'+\
                    str(n_projections)+ '/' + 'training_progress.csv'
                table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0)
                try:
                    data[-1].append(table['time_for_epochs'].values)
                except KeyError:
                    data[-1].append(table['time_for_iterations'].values)
                data[-1].append(table['SWD_approximation'].values)
                name = architecture
                if name == 'JLSWGN':
                    name += str(JL_dim)
                data[-1].append(name)

            f = plt.figure()
            for i in range(len(data)):
                if data[i][2] == 'SWGN':
                    last = last1
                else:
                    last = last2
                plt.semilogy(np.cumsum(data[i][0][:last]), data[i][1][:last], plot_style, label=data[i][2])
                print 'total trainings time {}: {}'.format(data[i][2], np.cumsum(data[i][0][:last])[-1])
            plt.legend()
            plt.title('B={}, L={}, BN={}'.format(batch_size, n_projections, BN))
            plt.xlabel('time (s)')
            plt.ylabel('SWD')
            f.savefig(path + 'plot_{}_{}_{}_{}_{}_{}.png'.format(dataset, batch_size, n_projections, BN,
                                                                 last1, last2),
                      bbox_inches=tr.Bbox.from_bounds(-0.3, 0, 6.5, 4.7))
            plt.close(f)


def plot_improved_JLSWGN(param_dict=settings.improved_JLSWGN_param_dict1_0,
                         path='/Users/Flo/Desktop/Training results/improved_JLSWGN/',
                         last1=50, last2=50, plot_style='o-', BN=True):
    input_dim = param_dict['input_dim'][0]
    n_features_first = param_dict['n_features_first'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    min_features = param_dict['min_features'][0]
    learning_rate = param_dict['learning_rate'][0]
    init_method = param_dict['init_method'][0]
    dataset = param_dict['data'][0]
    c_batch_size = param_dict['c_batch_size'][0]
    c_learning_rate = param_dict['c_learning_rate'][0]
    c_BN = param_dict['c_BN'][0]
    c_iterations = param_dict['c_iterations'][0]
    c_n_features_last = param_dict['c_n_features_last'][0]

    for batch_size in param_dict['batch_size']:
        for n_projections in param_dict['n_projections']:
            data = []
            for epsilon in param_dict['epsilon']:
                for i, JL_dim in enumerate(param_dict['JL_dim']):
                    data.append([])
                    if JL_dim is None:
                        JL_error = None
                        architecture = 'SWGN'
                    else:
                        JL_error = np.round(np.sqrt(8 * np.log(2 * batch_size) / JL_dim), decimals=4)
                        architecture = 'JLSWGN'

                    file = path + str(dataset)+ '_' + \
                            str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                            str(n_features_first) + '_' + str(n_features_reduction_factor) + '_' + \
                            str(min_features) + '_' + \
                            str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + \
                            str(JL_dim) + '_' + str(JL_error) + '_' + str(n_projections) + '_' + \
                            str(c_batch_size) + '_' + str(c_learning_rate) + '_' + str(c_BN) + '_' + \
                            str(c_iterations) + '_' + str(c_n_features_last) + '_' + str(epsilon) + '/' + \
                            'training_progress.csv'
                    table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0)
                    try:
                        data[-1].append(table['time_for_epochs'].values)
                    except KeyError:
                        data[-1].append(table['time_for_iterations'].values)
                    data[-1].append(table['SWD_approximation'].values)
                    name = architecture
                    if name == 'JLSWGN':
                        name += str(JL_dim)
                    name += '_{}'.format(epsilon)
                    data[-1].append(name)

            f = plt.figure()
            for i in range(len(data)):
                if data[i][2] == 'SWGN':
                    last = last1
                else:
                    last = last2
                plt.semilogy(np.cumsum(data[i][0][:last]), data[i][1][:last], plot_style, label=data[i][2])
            plt.legend()
            plt.title('batch_size={}'.format(batch_size))
            plt.xlabel('time (s)')
            plt.ylabel('SWD')
            f.savefig(path + 'plot_{}_{}_{}_{}.png'.format(dataset, batch_size, last1, last2), bbox_inches="tight")
            plt.close(f)


def plot_JLSWGAN(param_dict=settings.JLSWGAN_param_dict2_1, path='/Users/Flo/Desktop/Training results/JLSWGAN/mnist/',
                last1=50, last2=50, plot_style='o-', BN=True, d_BN='BN'):
    input_dim = param_dict['input_dim'][0]
    n_features_first_g = param_dict['n_features_first_g'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    dataset = param_dict['data'][0]
    n_features_last_d = param_dict['n_features_last_d'][0]
    d_freq = param_dict['d_freq'][0]
    d_steps = param_dict['d_steps'][0]
    min_features = param_dict['min_features'][0]

    for extra_layer_g in param_dict['extra_layer_g']:
        for batch_size in param_dict['batch_size']:
            for n_projections in param_dict['n_projections']:
                data = []
                for i, JL_dim in enumerate(param_dict['JL_dim']):
                    data.append([])
                    if JL_dim is None:
                        architecture = 'SWGAN'
                    else:
                        architecture = 'JLSWGAN'

                    file = path + str(dataset)+'_' + \
                    str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                    str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                    str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' +\
                    str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                    str(JL_dim) + '_' + str(n_projections) + '/' + 'training_progress.csv'
                    table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0)
                    try:
                        data[-1].append(table['time_for_epochs'].values)
                    except KeyError:
                        data[-1].append(table['time_for_iterations'].values)
                    data[-1].append(table['SWD_approximation'].values)
                    name = architecture
                    if name == 'JLSWGAN':
                        name += str(JL_dim)
                    data[-1].append(name)

                f = plt.figure()
                for i in range(len(data)):
                    if data[i][2] == 'SWGAN':
                        last = last1
                    else:
                        last = last2
                    plt.semilogy(np.cumsum(data[i][0][:last]), data[i][1][:last], plot_style, label=data[i][2])
                    print 'total trainings time {}: {}'.format(data[i][2], np.cumsum(data[i][0][:last])[-1])
                plt.legend()
                plt.title('B={}, L={}, norm={}, extra_layer={}'.format(batch_size, n_projections, d_BN, extra_layer_g))
                plt.xlabel('time (s)')
                plt.ylabel('SWD')
                f.savefig(path + 'plot_JLSWGAN_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(dataset, batch_size, n_projections, BN,
                                                                                extra_layer_g, d_BN, last1, last2),
                          bbox_inches=tr.Bbox.from_bounds(-0.3, 0, 6.5, 4.7))
                plt.close(f)


def plot_IS(param_dict=settings.JLSWGAN_param_dict3_0, path='/Users/Flo/Desktop/Training results/JLSWGAN/cifar10/',
            last=None, plot_style='o-', BN=True, d_BN='BN'):
    input_dim = param_dict['input_dim'][0]
    n_features_first_g = param_dict['n_features_first_g'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    dataset = param_dict['data'][0]
    n_features_last_d = param_dict['n_features_last_d'][0]
    d_freq = param_dict['d_freq'][0]
    d_steps = param_dict['d_steps'][0]
    min_features = param_dict['min_features'][0]

    for extra_layer_g in param_dict['extra_layer_g']:
        for batch_size in param_dict['batch_size']:
            for n_projections in param_dict['n_projections']:
                data = []
                for i, JL_dim in enumerate(param_dict['JL_dim']):
                    data.append([])
                    if JL_dim is None:
                        architecture = 'SWGAN'
                    else:
                        architecture = 'JLSWGAN'

                    file = path + str(dataset)+'_' + \
                    str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                    str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                    str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' +\
                    str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                    str(JL_dim) + '_' + str(n_projections) + '/' + 'training_progress.csv'
                    table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0, na_filter=False)
                    try:
                        t = np.cumsum(table['time_for_epochs'].values)
                    except KeyError:
                        t = np.cumsum(table['time_for_iterations'].values)
                    IS = table['IS'].values

                    times = []
                    means = []
                    std = []
                    if last is None:
                        last1 = len(t)
                    else:
                        last1 = last
                    for j in range(min(len(t), last1)):
                        if IS[j] is not '':
                            s = IS[j][1:-1].split(', ')
                            means.append(float(s[0]))
                            std.append(float(s[1]))
                            times.append(t[j])

                    data[-1].append(times)
                    data[-1].append(means)
                    data[-1].append(std)

                    name = architecture
                    if name == 'JLSWGAN':
                        name += str(JL_dim)
                    data[-1].append(name)

                    # print best measured IS
                    j = np.argmax(means)
                    print 'highest IS of {} with B={}, L={}: {:.2f} +/- {:.2f}'.format(name, batch_size, n_projections,
                                                                                       means[j], std[j])

                print
                f = plt.figure()
                cmap = plt.get_cmap("tab10")
                for i in range(len(data)):
                    plt.errorbar(x=data[i][0], y=data[i][1], yerr=data[i][2], fmt=plot_style, ecolor='k', capsize=4,
                                 label=data[i][3], color=cmap(i))
                    print 'total trainings time {}: {}'.format(data[i][3], data[i][0][-1])
                print
                print '='*120
                plt.legend()
                plt.title('B={}, L={}, norm={}, extra_layer={}'.format(batch_size, n_projections, d_BN, extra_layer_g))
                plt.xlabel('time (s)')
                plt.ylabel('IS')
                f.savefig(path + 'plot_IS_JLSWGAN_{}_{}_{}_{}_{}_{}.png'.format(dataset, batch_size, n_projections, BN,
                                                                                extra_layer_g, d_BN),
                          bbox_inches=tr.Bbox.from_bounds(-0.3, 0, 6.5, 4.7))
                plt.close(f)


def plot_IS2(param_dict=settings.JLSWGAN_param_dict3_0, path='/Users/Flo/Desktop/Training results/JLSWGAN/cifar10/',
             last=None, plot_style='o-', BN=True, d_BN='BN'):
    input_dim = param_dict['input_dim'][0]
    n_features_first_g = param_dict['n_features_first_g'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    dataset = param_dict['data'][0]
    n_features_last_d = param_dict['n_features_last_d'][0]
    d_freq = param_dict['d_freq'][0]
    d_steps = param_dict['d_steps'][0]
    min_features = param_dict['min_features'][0]


    for batch_size in param_dict['batch_size']:
        for n_projections in param_dict['n_projections']:
            data = []
            for extra_layer_g in param_dict['extra_layer_g']:
                for i, JL_dim in enumerate(param_dict['JL_dim']):
                    data.append([])
                    if JL_dim is None:
                        architecture = 'SWGAN'
                    else:
                        architecture = 'JLSWGAN'

                    file = path + str(dataset)+'_' + \
                    str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                    str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                    str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' +\
                    str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                    str(JL_dim) + '_' + str(n_projections) + '/' + 'training_progress.csv'
                    table = pd.read_csv(filepath_or_buffer=file, index_col=0, header=0, na_filter=False)
                    try:
                        t = np.cumsum(table['time_for_epochs'].values)
                    except KeyError:
                        t = np.cumsum(table['time_for_iterations'].values)
                    IS = table['IS'].values

                    times = []
                    means = []
                    std = []
                    if last is None:
                        last1 = len(t)
                    else:
                        last1 = last
                    for j in range(min(len(t), last1)):
                        if IS[j] is not '':
                            s = IS[j][1:-1].split(', ')
                            means.append(float(s[0]))
                            std.append(float(s[1]))
                            times.append(t[j])

                    data[-1].append(times)
                    data[-1].append(means)
                    data[-1].append(std)

                    name = architecture
                    if extra_layer_g:
                        name += '_EL'
                    if architecture == 'JLSWGAN':
                        name += str(JL_dim)
                    data[-1].append(name)

                    # print best measured IS
                    j = np.argmax(means)
                    print 'highest IS of {} with B={}, L={}: {:.2f} +/- {:.2f}'.format(name, batch_size, n_projections,
                                                                                       means[j], std[j])


            f = plt.figure()
            for i in range(len(data)):
                if data[i][3] == 'SWGAN':
                    color = 'r'
                elif data[i][3] == 'SWGAN_EL':
                    color = 'm'
                elif data[i][3][:10] == 'JLSWGAN_EL':
                    color = 'c'
                else:
                    color = 'b'

                plt.errorbar(x=data[i][0], y=data[i][1], yerr=data[i][2], fmt=plot_style, ecolor='k', capsize=4,
                             label=data[i][3], color=color)
                print 'total trainings time {}: {}'.format(data[i][3], data[i][0][-1])
            plt.legend()
            plt.title('B={}, L={}, norm={}'.format(batch_size, n_projections, d_BN))
            plt.xlabel('time (s)')
            plt.ylabel('IS')
            f.savefig(path + 'plot_IS2_JLSWGAN_{}_{}_{}_{}_{}_{}.png'.format(dataset, batch_size, n_projections, BN,
                                                                            extra_layer_g, d_BN),
                      bbox_inches=tr.Bbox.from_bounds(-0.3, 0, 6.5, 4.7))
            plt.close(f)


def compare_training_times_JLSWGAN(param_dict=settings.JLSWGAN_param_dict2_1,
                           path='/Users/Flo/Desktop/Training results/JLSWGAN1/mnist/',
                           JL_dim=512, L=(1000, 10000), last=None):
    input_dim = param_dict['input_dim'][0]
    n_features_first_g = param_dict['n_features_first_g'][0]
    n_features_reduction_factor = param_dict['n_features_reduction_factor'][0]
    init_method = param_dict['init_method'][0]
    learning_rate = param_dict['learning_rate'][0]
    dataset = param_dict['data'][0]
    n_features_last_d = param_dict['n_features_last_d'][0]
    d_freq = param_dict['d_freq'][0]
    d_steps = param_dict['d_steps'][0]
    min_features = param_dict['min_features'][0]
    BN = True
    architecture1 = 'SWGAN'
    architecture2 = 'JLSWGAN'

    ratio1 = []
    ratio2 = []
    ratio3 = []
    ratio4 = []

    for extra_layer_g in param_dict['extra_layer_g']:
        for batch_size in param_dict['batch_size']:
            for d_BN in param_dict['d_BN']:

                print '\n'
                print '=' * 120
                print 'extra layer: {}'.format(extra_layer_g)
                print 'batch size: {}'.format(batch_size)
                print 'd_BN: {}'.format(d_BN)

                file1 = path + str(dataset)+'_' + \
                    str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                    str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                    str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' +\
                    str(architecture1) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                    str(None) + '_' + str(L[0]) + '/' + 'training_progress.csv'
                table1 = pd.read_csv(filepath_or_buffer=file1, index_col=0, header=0)

                file2 = path + str(dataset) + '_' + \
                        str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                        str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(
                    min_features) + '_' + \
                        str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(
                    d_steps) + '_' + \
                        str(architecture1) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                        str(None) + '_' + str(L[1]) + '/' + 'training_progress.csv'
                table2 = pd.read_csv(filepath_or_buffer=file2, index_col=0, header=0)

                file3 = path + str(dataset) + '_' + \
                        str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                        str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(
                    min_features) + '_' + \
                        str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(
                    d_steps) + '_' + \
                        str(architecture2) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                        str(JL_dim) + '_' + str(L[0]) + '/' + 'training_progress.csv'
                table3 = pd.read_csv(filepath_or_buffer=file3, index_col=0, header=0)

                file4 = path + str(dataset) + '_' + \
                        str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                        str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(
                    min_features) + '_' + \
                        str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(
                    d_steps) + '_' + \
                        str(architecture2) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                        str(JL_dim) + '_' + str(L[1]) + '/' + 'training_progress.csv'
                table4 = pd.read_csv(filepath_or_buffer=file4, index_col=0, header=0)

                t1 = table1['time_for_iterations'].values
                t2 = table2['time_for_iterations'].values
                t3 = table3['time_for_iterations'].values
                t4 = table4['time_for_iterations'].values

                if last is None:
                    l = int(min(len(t1), len(t2), len(t3), len(t4)))
                else:
                    l = int(min(last, len(t1), len(t2), len(t3), len(t4)))
                print '\n'
                print 'number of iterations that are compared: {}'.format(l*100)

                print 'mean time for SWGAN with L={} is: {}'.format(L[0], np.mean(t1[:l]))
                print 'mean time for SWGAN with L={} is: {}'.format(L[1], np.mean(t2[:l]))
                print 'mean time for JL-SWGAN m={} with L={} is: {}'.format(JL_dim, L[0], np.mean(t3[:l]))
                print 'mean time for JL-SWGAN m={} with L={} is: {}'.format(JL_dim, L[1], np.mean(t4[:l]))

                print 'ratio for SWGAN between L={} and L={} is: {}'.format(L[1], L[0], np.mean(t2[:l])/np.mean(t1[:l]))
                print 'ratio for JL-SWGAN between L={} and L={} is: {}'.format(L[1], L[0],
                                                                               np.mean(t4[:l]) / np.mean(t3[:l]))
                print 'ratio between SWGAN and JL-SWGAN for L={} is: {}'.format(L[0], np.mean(t1[:l])/np.mean(t3[:l]))
                print 'ratio between SWGAN and JL-SWGAN for L={} is: {}'.format(L[1], np.mean(t2[:l])/np.mean(t4[:l]))

                ratio1.append(np.mean(t2[:l])/np.mean(t1[:l]))
                ratio2.append(np.mean(t4[:l]) / np.mean(t3[:l]))
                ratio3.append(np.mean(t1[:l])/np.mean(t3[:l]))
                ratio4.append(np.mean(t2[:l])/np.mean(t4[:l]))

                ll = len(t4)
                b = np.sum(t4)
                x = np.cumsum(t2)
                for j in range(ll):
                    if x[j] >= b:
                        print 'JL-SWGAN training time for {} iterations with L={}: {} '.format((ll)*100, L[1], b)
                        print 'SWGAN training time for {} iterations with L={}: {}'.format((j+1)*100, L[1], x[j])
                        break

    print '\n'
    print '='*120
    print 'ratio for SWGAN between L={} and L={} mean: {}, min: {}, max {}'.format(L[1], L[0], np.mean(ratio1),
                                                                                   np.min(ratio1), np.max(ratio1))
    print 'ratio for JL-SWGAN between L={} and L={} mean: {}, min: {}, max {}'.format(L[1], L[0], np.mean(ratio2),
                                                                                      np.min(ratio2), np.max(ratio2))
    print 'ratio between SWGAN and JL-SWGAN for L={} mean: {}, min: {}, max {}'.format(L[0], np.mean(ratio3),
                                                                                       np.min(ratio3), np.max(ratio3))
    print 'ratio between SWGAN and JL-SWGAN for L={} mean: {}, min: {}, max {}'.format(L[1], np.mean(ratio4),
                                                                                       np.min(ratio4), np.max(ratio4))



if __name__ == '__main__':
    # plot_JLSWGN_toy()

    # plot_JLSWGN_toy2(param_dict=settings.JLSWGN_toy_param_dict3_0,
    #                  path='/Users/Flo/Desktop/Training results/JLSWGN_toy/',
    #                  plot_style='o-')

    # plot_JLSWGN_mnist(plot_style='-', last1=100, last2=100)
    # plot_JLSWGN_mnist(plot_style='o-', last1=20, last2=20)

    # plot_JLSWGN(settings.JLSWGN_param_dict4_0, path='/Users/Flo/Desktop/Training results/JLSWGN/celebA64/',
    #             last1=200, last2=200, plot_style='-', BN=True)

    # plot_improved_JLSWGN(param_dict=settings.improved_JLSWGN_param_dict2_0,
    #                      path='/Users/Flo/Desktop/Training results/improved_JLSWGN/cifar10/',
    #                      last1=400, last2=400, plot_style='-')

    # plot_JLSWGAN(param_dict=settings.JLSWGAN_param_dict3_0, path='/Users/Flo/Desktop/celebA/',
    #              last1=260, last2=260, plot_style='-', BN=True, d_BN='BN')

    # plot_JLSWGAN(param_dict=settings.JLSWGAN_param_dict1_1, path='/Users/Flo/Desktop/cifar10_1/',
    #              last1=350, last2=350, plot_style='-', BN=True, d_BN='BN')

    # compare_training_times_JLSWGAN(param_dict=settings.JLSWGAN_param_dict2_1,
    #                                path='/Users/Flo/Desktop/JLSWGAN/mnist/',
    #                                JL_dim=256, L=[10000, 10000], last=200)

    # compare_training_times_JLSWGAN(param_dict=settings.JLSWGAN_param_dict1_0,
    #                                path='/Users/Flo/Desktop/JLSWGAN/cifar10/',
    #                                JL_dim=512, L=[1000, 10000], last=330)

    # plot_JLSWGAN(param_dict=settings.JLSWGAN_param_dict1_3, path='/Users/Flo/Desktop/d/',
    #              last1=350, last2=350, plot_style='-', BN=True, d_BN='BN')
    plot_IS(param_dict=settings.JLSWGAN_param_dict1_2,
            # path='/Users/Flo/Desktop/Training results/JLSWGAN/cifar10/with_IS/',
            path='/Users/Flo/Desktop/d/',
            last=None, plot_style='o-', BN=True, d_BN='BN')

    # plot_IS2(param_dict=settings.JLSWGAN_param_dict1_0, path='/Users/Flo/Desktop/cifar10_0/',
    #          last=None, plot_style='o-', BN=True, d_BN='LN')

    # for i in range(2):
    #     t = pd.read_csv('/users/flo/desktop/training_progress_{}4.csv'.format(i + 1), index_col=0, header=0)
    #     x = t.iloc[:620, 1].values
    #     print 'mean time for {} is: {}'.format(i + 1, np.mean(x))
    #
    #     t2 = pd.read_csv('/users/flo/desktop/training_progress_{}1.csv'.format(i + 1), index_col=0, header=0)
    #     x2 = t2.iloc[:620, 1].values
    #     print 'mean time for SWG is: {}'.format(np.mean(x2))
    #     print 'ratio: {}'.format(np.mean(x2)/np.mean(x))
    #     x2 = np.cumsum(x2)
    #     b = np.sum(x)
    #     for j in range(len(x2)):
    #         if x2[j] >= b:
    #             print j
    #             print x2[j], b, x2[150]
    #             break

    pass


