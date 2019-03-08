"""
author: Florian Krach
"""

from sklearn.model_selection import ParameterGrid

# ------------------------------------------------------------------------
# settings for data sets
filepath_mnist = '../data/MNIST/mnist.pkl.gz'
filepath_cifar10 = '../data/cifar-10-batches-py'
filepath_celebA32 = '../data/celebA32/'
filepath_celebA64 = '../data/celebA64/'
filepath_celebA64_euler = '/cluster/scratch/fkrach/celebA64/'

# ------------------------------------------------------------------------
# settings for generated data
filepath_samples = 'samples/'

# ------------------------------------------------------------------------
# other settings
number_cpus = 2  # this is the number of cpus that is used by each tf.Session(),
# only used if not specified otherwise in the respective file

wgan_mnist_get_statistics = False
get_statistics = False
test_classifier = False

send_email = False
euler = False

# ------------------------------------------------------------------------
# settings for parallel training
number_available_cpus = 2  # only used if not specified otherwise in the respective file
number_parallel_jobs = number_available_cpus / number_cpus


# ------------------------------------------------------------------------
def get_parameter_array(param_dict,
                        ordered_keys=('input_dim', 'batch_size', 'n_features_first', 'critic_iters', 'lambda_reg',
                                      'learning_rate', 'epochs', 'fixed_noise_size', 'n_features_reduction_factor',
                                      'fix_first_layers_gen', 'fix_last_layer_gen', 'fix_2last_layer_gen',
                                      'fix_first_layers_disc', 'fix_last_layer_disc', 'fix_2last_layer_disc',
                                      'architecture', 'use_unfixed_gradient_only', 'extra_fully_connected_layer',
                                      'init_method', 'BN_layers_trainable', 'different_optimizers')
                        ):
    """
    - this functions takes a dict in which parameteres are specified out of which an array of all combinations of them
      is made

    :param param_dict:
    :param ordered_keys:
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    parameter_array = [[p[k] for k in ordered_keys] for p in param_combs_dict_list]
    return parameter_array


ordered_keys_1 = ('input_dim', 'batch_size', 'n_features_first', 'critic_iters', 'lambda_reg',
                  'learning_rate', 'iterations', 'fixed_noise_size', 'n_features_reduction_factor',
                  'gen_fix_layer_1', 'gen_fix_layer_2', 'gen_fix_layer_3', 'gen_fix_layer_4',
                  'disc_fix_layer_1', 'disc_fix_layer_2', 'disc_fix_layer_3', 'disc_fix_layer_4',
                  'architecture', 'init_method', 'BN_layers_trainable')

ordered_keys_2 = ('epochs_new', 'standard_deviation_factor_new',
                  'fix_last_layer_gen_new', 'fix_2last_layer_gen_new',
                  'fix_last_layer_disc_new', 'fix_2last_layer_disc_new',
                  'pretrained_model', 'perturb_BN')

ordered_keys_3 = ('input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                  'lambda_reg', 'learning_rate', 'epochs1', 'epochs2',
                  'fixed_noise_size', 'n_features_reduction_factor',
                  'fix_last_layer_gen', 'fix_2last_layer_gen',
                  'fix_last_layer_disc', 'fix_2last_layer_disc',
                  'architecture', 'number_digits', 'which_digits',
                  'second_train_on_others_only', 'perturb_factor')

ordered_keys_4 = ('input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                  'lambda_reg', 'learning_rate', 'epochs1', 'epochs2',
                  'fixed_noise_size', 'n_features_reduction_factor',
                  'fix_last_layer_gen', 'fix_2last_layer_gen',
                  'fix_last_layer_disc', 'fix_2last_layer_disc',
                  'architecture', 'BN_layers_trainable', 'different_optimizers')

ordered_keys_5 = ('input_dim', 'batch_size', 'n_features_first',
                  'learning_rate', 'epochs',
                  'fixed_noise_size', 'n_features_reduction_factor',
                  'architecture', 'init_method', 'BN', 'JL_dim', 'JL_error', 'n_projections')

ordered_keys_6 = ('input_dim', 'batch_size', 'n_features_first',
                  'learning_rate', 'epochs',
                  'fixed_noise_size',
                  'architecture', 'init_method', 'BN', 'JL_dim', 'JL_error', 'n_projections',
                  'target_dim', 'target_number_samples', 'target_sigma', 'run')

ordered_keys_7 = ('input_dim', 'batch_size', 'n_features_first',
                  'learning_rate', 'epochs',
                  'fixed_noise_size', 'n_features_reduction_factor',
                  'architecture', 'init_method', 'BN', 'JL_dim', 'JL_error', 'n_projections',
                  'data')

ordered_keys_8 = ('input_dim', 'batch_size', 'learning_rate', 'epochs',
                  'fixed_noise_size', 'n_features_first', 'n_features_reduction_factor',
                  'min_features', 'architecture', 'init_method', 'BN', 'JL_dim', 'JL_error',
                  'n_projections', 'power', 'image_enlarge_method', 'order')

ordered_keys_9 = ('data', 'input_dim', 'batch_size', 'learning_rate', 'epochs',
                  'fixed_noise_size', 'n_features_first', 'n_features_reduction_factor',
                  'min_features', 'architecture', 'init_method', 'BN', 'JL_dim', 'JL_error',
                  'n_projections', 'c_batch_size', 'c_learning_rate', 'c_BN', 'c_iterations', 'c_n_features_last',
                  'epsilon')

ordered_keys_10 = ('data', 'input_dim', 'batch_size', 'learning_rate', 'iterations',
                   'fixed_noise_size', 'n_features_first_g', 'n_features_reduction_factor',
                   'min_features', 'n_features_last_d', 'extra_layer_g', 'd_freq', 'd_steps',
                   'architecture', 'init_method', 'BN', 'd_BN', 'JL_dim', 'n_projections')

ordered_keys_11 = ('input_dim', 'batch_size', 'n_features_first',
                   'learning_rate', 'epochs',
                   'fixed_noise_size',
                   'architecture', 'init_method', 'BN', 'JL_dim', 'n_projections',
                   'target_dim', 'target_number_samples', 'target_sigma', 'd_freq', 'd_steps')

# ------------------------------------------------------------------------
# parameter arrays for parallel training of training_mnist_partly_fixed.py
param_dict4_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [False], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [False], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': [None],
                 'BN_layers_trainable': [True], 'different_optimizers': [False]}
param_array4_0 = get_parameter_array(param_dict4_0)

param_dict4_1 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256, 512], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4, 5e-3, 1e-3], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [1, 2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': [None],
                 'BN_layers_trainable': [True], 'different_optimizers': [False]}
param_array4_1 = get_parameter_array(param_dict4_1)

param_array4 = param_array4_0 + param_array4_1

# -------------------------------
# try different weight initializers
param_dict5_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [True],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [True],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal',
                                                                         'normal1', 'truncated_normal1'],
                 'BN_layers_trainable': [True], 'different_optimizers': [False]}
param_array5_0 = get_parameter_array(param_dict5_0)

param_dict5_1 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['normal', 'truncated_normal',
                                                                         'normal1', 'truncated_normal1'],
                 'BN_layers_trainable': [True], 'different_optimizers': [False]}
param_array5_1 = get_parameter_array(param_dict5_1)

param_array5 = param_array5_0 + param_array5_1

# -------------------------------
# try different weight initializers when BN layers are not trainable and when 2 different optimizers are used
param_dict6_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [True],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [True],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal',
                                                                         'normal_BN', 'uniform_BN'],
                 'BN_layers_trainable': [False], 'different_optimizers': [False]}
param_array6_0 = get_parameter_array(param_dict6_0)

param_dict6_4 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [True],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [True],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal',
                                                                         'normal_BN', 'uniform_BN'],
                 'BN_layers_trainable': [False], 'different_optimizers': [True]}
param_array6_4 = get_parameter_array(param_dict6_4)

param_dict6_1 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [True],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [True],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal'],
                 'BN_layers_trainable': [True], 'different_optimizers': [True]}
param_array6_1 = get_parameter_array(param_dict6_1)

param_dict6_2 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal'],
                 'BN_layers_trainable': [True], 'different_optimizers': [True]}
param_array6_2 = get_parameter_array(param_dict6_2)

param_dict6_3 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [False], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [False], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['uniform', 'normal', 'truncated_normal'],
                 'BN_layers_trainable': [True], 'different_optimizers': [True]}
param_array6_3 = get_parameter_array(param_dict6_3)

param_array6 = param_array6_0 + param_array6_4 + param_array6_1 + param_array6_2 + param_array6_3

# -------------------------------
# try He init, with different other settings
param_dict7_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4, 1e-3], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [True],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [True],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['He'],
                 'BN_layers_trainable': [False, True], 'different_optimizers': [False]}
param_array7_0 = get_parameter_array(param_dict7_0)


param_dict7_2 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                 'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs': [20000],
                 'fixed_noise_size': [128],
                 'n_features_reduction_factor': [2],
                 'fix_first_layers_gen': [True], 'fix_last_layer_gen': [False],
                 'fix_2last_layer_gen': [False],
                 'fix_first_layers_disc': [True], 'fix_last_layer_disc': [False],
                 'fix_2last_layer_disc': [False],
                 'architecture': ['DCGAN'], 'use_unfixed_gradient_only': [False],
                 'extra_fully_connected_layer': [False], 'init_method': ['He'],
                 'BN_layers_trainable': [False, True], 'different_optimizers': [False]}
param_array7_2 = get_parameter_array(param_dict7_2)

param_array7 = param_array7_0 + param_array7_1 + param_array7_2


# ------------------------------------------------------------------------
# parameter arrays for training_partly_fixed2.py
partly_fixed_param_dict1 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                            'lambda_reg': [None], 'learning_rate': [1e-4], 'iterations': [20000],
                            'fixed_noise_size': [128],
                            'n_features_reduction_factor': [2],
                            'gen_fix_layer_1': [True, False], 'gen_fix_layer_2': [True],
                            'gen_fix_layer_3': [True], 'gen_fix_layer_4': [False],
                            'disc_fix_layer_1': [False], 'disc_fix_layer_2': [True],
                            'disc_fix_layer_3': [True], 'disc_fix_layer_4': [False],
                            'architecture': ['DCGAN'], 'init_method': ['normal', 'He'],
                            'BN_layers_trainable': [False, True]}
partly_fixed_param_array1 = get_parameter_array(partly_fixed_param_dict1, ordered_keys=ordered_keys_1)

partly_fixed_param_dict2 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                            'lambda_reg': [None], 'learning_rate': [1e-4], 'iterations': [20000],
                            'fixed_noise_size': [128],
                            'n_features_reduction_factor': [2],
                            'gen_fix_layer_1': [True], 'gen_fix_layer_2': [True],
                            'gen_fix_layer_3': [False], 'gen_fix_layer_4': [False],
                            'disc_fix_layer_1': [False], 'disc_fix_layer_2': [True],
                            'disc_fix_layer_3': [True], 'disc_fix_layer_4': [False],
                            'architecture': ['DCGAN'], 'init_method': ['normal', 'He'],
                            'BN_layers_trainable': [False, True]}
partly_fixed_param_array2 = get_parameter_array(partly_fixed_param_dict2, ordered_keys=ordered_keys_1)

partly_fixed_param_dict3 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1, 5],
                            'lambda_reg': [None], 'learning_rate': [1e-4], 'iterations': [20000],
                            'fixed_noise_size': [128],
                            'n_features_reduction_factor': [2],
                            'gen_fix_layer_1': [False], 'gen_fix_layer_2': [False],
                            'gen_fix_layer_3': [False], 'gen_fix_layer_4': [False],
                            'disc_fix_layer_1': [False], 'disc_fix_layer_2': [True],
                            'disc_fix_layer_3': [True], 'disc_fix_layer_4': [False],
                            'architecture': ['DCGAN'], 'init_method': ['normal', 'He'],
                            'BN_layers_trainable': [False, True]}
partly_fixed_param_array3 = get_parameter_array(partly_fixed_param_dict3, ordered_keys=ordered_keys_1)

partly_fixed_param_dict4 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                            'lambda_reg': [None], 'learning_rate': [1e-4], 'iterations': [20000],
                            'fixed_noise_size': [128],
                            'n_features_reduction_factor': [2],
                            'gen_fix_layer_1': [True], 'gen_fix_layer_2': [True],
                            'gen_fix_layer_3': [True], 'gen_fix_layer_4': [False],
                            'disc_fix_layer_1': [True], 'disc_fix_layer_2': [True],
                            'disc_fix_layer_3': [True], 'disc_fix_layer_4': [False],
                            'architecture': ['DCGAN'], 'init_method': ['normal', 'He'],
                            'BN_layers_trainable': [False, True]}
partly_fixed_param_array4 = get_parameter_array(partly_fixed_param_dict4, ordered_keys=ordered_keys_1)

partly_fixed_param_dict5 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                            'lambda_reg': [None], 'learning_rate': [1e-4], 'iterations': [20000],
                            'fixed_noise_size': [128],
                            'n_features_reduction_factor': [2],
                            'gen_fix_layer_1': [True], 'gen_fix_layer_2': [True],
                            'gen_fix_layer_3': [True, False], 'gen_fix_layer_4': [False],
                            'disc_fix_layer_1': [False], 'disc_fix_layer_2': [False],
                            'disc_fix_layer_3': [False], 'disc_fix_layer_4': [False],
                            'architecture': ['DCGAN'], 'init_method': ['normal', 'He'],
                            'BN_layers_trainable': [False, True]}
partly_fixed_param_array5 = get_parameter_array(partly_fixed_param_dict5, ordered_keys=ordered_keys_1)



# ------------------------------------------------------------------------
# parameter arrays for parallel training of perturbed_pretrained_network_mnist
perturbed_param_dict2_0 = {'epochs_new': [10000], 'standard_deviation_factor_new': [0.25, 0.5, 1, 2],
                           'fix_last_layer_gen_new': [False], 'fix_2last_layer_gen_new': [True],
                           'fix_last_layer_disc_new': [False], 'fix_2last_layer_disc_new': [True],
                           'pretrained_model': ['DCGAN'], 'perturb_BN': [True]}
perturbed_param_array2 = get_parameter_array(perturbed_param_dict2_0, ordered_keys=ordered_keys_2)

perturbed_param_dict3_0 = {'epochs_new': [10000], 'standard_deviation_factor_new': [0.5, 1, 1.5, 2],
                           'fix_last_layer_gen_new': [False], 'fix_2last_layer_gen_new': [True],
                           'fix_last_layer_disc_new': [False], 'fix_2last_layer_disc_new': [True],
                           'pretrained_model': ['DCGAN'], 'perturb_BN': [True]}
perturbed_param_array3 = get_parameter_array(perturbed_param_dict3_0, ordered_keys=ordered_keys_2)

# ------------------------------------------------------------------------
# parameter arrays for parallel training of training_subset_numbers_mnist
subset_param_dict1_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256], 'critic_iters': [1],
                        'lambda_reg': [10], 'learning_rate': [1e-4], 'epochs1': [20000], 'epochs2': [20000],
                        'fixed_noise_size': [128],
                        'n_features_reduction_factor': [2],
                        'fix_last_layer_gen': [False], 'fix_2last_layer_gen': [True],
                        'fix_last_layer_disc': [False], 'fix_2last_layer_disc': [True],
                        'architecture': ['DCGAN'], 'number_digits': [5], 'which_digits': [None],
                        'second_train_on_others_only': [False, True], 'perturb_factor': [0, 0.5, 1]}
subset_param_array1 = get_parameter_array(subset_param_dict1_0, ordered_keys=ordered_keys_3)


# ------------------------------------------------------------------------
# parameter arrays for parallel training of old_JLSWGN.py
old_JLSWGN_param_dict1_0 = {'input_dim': [128], 'batch_size': [25, 50, 100], 'n_features_first': [256],
                            'learning_rate': [1e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [28 * 28 / 4, 28 * 28 / 2, 28 * 28 / 4 * 3], 'JL_error': [None],
                            'n_projections': [5000, 10000]}
old_JLSWGN_param_array1_0 = get_parameter_array(old_JLSWGN_param_dict1_0, ordered_keys=ordered_keys_5)

old_JLSWGN_param_dict1_1 = {'input_dim': [128], 'batch_size': [25, 50, 100], 'n_features_first': [256],
                            'learning_rate': [1e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['SWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [None],
                            'n_projections': [5000, 10000]}
old_JLSWGN_param_array1_1 = get_parameter_array(old_JLSWGN_param_dict1_1, ordered_keys=ordered_keys_5)

old_JLSWGN_param_array1 = old_JLSWGN_param_array1_0 + old_JLSWGN_param_array1_1

# -------------
# different learning rates and batch sizes
# todo: try different learning rate and higher number of epochs and less projections
old_JLSWGN_param_dict2_0 = {'input_dim': [128], 'batch_size': [100, 250], 'n_features_first': [256],
                            'learning_rate': [1e-3, 5e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [28 * 28 / 4, 28 * 28 / 2, 28 * 28 / 4 * 3], 'JL_error': [None],
                            'n_projections': [10000]}
old_JLSWGN_param_array2_0 = get_parameter_array(old_JLSWGN_param_dict2_0, ordered_keys=ordered_keys_5)

old_JLSWGN_param_dict2_1 = {'input_dim': [128], 'batch_size': [100, 250], 'n_features_first': [256],
                            'learning_rate': [1e-3, 5e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['SWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [None],
                            'n_projections': [10000]}
old_JLSWGN_param_array2_1 = get_parameter_array(old_JLSWGN_param_dict2_1, ordered_keys=ordered_keys_5)

old_JLSWGN_param_array2 = old_JLSWGN_param_array2_0 + old_JLSWGN_param_array2_1

# -------------
# without BN
# todo: try different learning rate and higher number of epochs and less projections
old_JLSWGN_param_dict3_0 = {'input_dim': [128], 'batch_size': [100, 250], 'n_features_first': [256],
                            'learning_rate': [5e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [False],
                            'JL_dim': [28 * 28 / 4, 28 * 28 / 2, 28 * 28 / 4 * 3], 'JL_error': [None],
                            'n_projections': [10000]}
old_JLSWGN_param_array3_0 = get_parameter_array(old_JLSWGN_param_dict3_0, ordered_keys=ordered_keys_5)

old_JLSWGN_param_dict3_1 = {'input_dim': [128], 'batch_size': [100, 250], 'n_features_first': [256],
                            'learning_rate': [5e-4],
                            'epochs': [40000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                            'architecture': ['SWGN'], 'init_method': ['He'], 'BN': [False],
                            'JL_dim': [None], 'JL_error': [None],
                            'n_projections': [10000]}
old_JLSWGN_param_array3_1 = get_parameter_array(old_JLSWGN_param_dict3_1, ordered_keys=ordered_keys_5)

old_JLSWGN_param_array3 = old_JLSWGN_param_array3_0 + old_JLSWGN_param_array3_1

# ------------------------------------------------------------------------
# parameter arrays for parallel training of JLSWGN_toy
JLSWGN_toy_param_dict1_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256],
                            'learning_rate': [1e-2],
                            'epochs': [1000], 'fixed_noise_size': [128],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [0.32, None],
                            'n_projections': [1000, 10000],
                            'target_dim': [1000, 5000, 10000, 50000], 'target_number_samples': [10000],
                            'target_sigma': [0.01, 0.1], 'run': [None]}
JLSWGN_toy_param_array1_0 = get_parameter_array(JLSWGN_toy_param_dict1_0, ordered_keys=ordered_keys_6)

# ------------
JLSWGN_toy_param_dict2_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256],
                            'learning_rate': [1e-2],
                            'epochs': [4000], 'fixed_noise_size': [128],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [0.32],
                            'n_projections': [10000],
                            'target_dim': [50000], 'target_number_samples': [10000],
                            'target_sigma': [0.01, 0.1], 'run': [None]}
JLSWGN_toy_param_array2_0 = get_parameter_array(JLSWGN_toy_param_dict2_0, ordered_keys=ordered_keys_6)

# --------------
JLSWGN_toy_param_dict3_0 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256],
                            'learning_rate': [1e-2],
                            'epochs': [1000], 'fixed_noise_size': [128],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [None],
                            'n_projections': [10000],
                            'target_dim': [1000, 5000, 10000, 50000], 'target_number_samples': [10000],
                            'target_sigma': [0.01], 'run': [1, 2, 3, 4, 5]}
JLSWGN_toy_param_array3_0 = get_parameter_array(JLSWGN_toy_param_dict3_0, ordered_keys=ordered_keys_6)

JLSWGN_toy_param_dict3_1 = {'input_dim': [128], 'batch_size': [50], 'n_features_first': [256],
                            'learning_rate': [1e-2],
                            'epochs': [4000], 'fixed_noise_size': [128],
                            'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                            'JL_dim': [None], 'JL_error': [0.32],
                            'n_projections': [10000],
                            'target_dim': [1000, 5000, 10000, 50000], 'target_number_samples': [10000],
                            'target_sigma': [0.01], 'run': [1, 2, 3, 4, 5]}
JLSWGN_toy_param_array3_1 = get_parameter_array(JLSWGN_toy_param_dict3_1, ordered_keys=ordered_keys_6)

JLSWGN_toy_param_array3 = JLSWGN_toy_param_array3_0 + JLSWGN_toy_param_array3_1

# ------------------------------------------------------------------------
# parameter arrays for parallel training of JLSWGN on CIFAR10 or CelebA or MNIST
# -------
# cifar10
JLSWGN_param_dict1_0 = {'input_dim': [128], 'batch_size': [100], 'n_features_first': [256],
                        'learning_rate': [5e-4],
                        'epochs': [100000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                        'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True, False],
                        'JL_dim': [32 * 32 / 2, None], 'JL_error': [None],
                        'n_projections': [1000], 'data': ['cifar10']}
JLSWGN_param_array1_0 = get_parameter_array(JLSWGN_param_dict1_0, ordered_keys=ordered_keys_7)

JLSWGN_param_array1 = JLSWGN_param_array1_0

JLSWGN_param_dict1_2 = {'input_dim': [128], 'batch_size': [100], 'n_features_first': [256],
                        'learning_rate': [5e-4],
                        'epochs': [200000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                        'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                        'JL_dim': [32 * 32 / 2, None], 'JL_error': [None],
                        'n_projections': [1000], 'data': ['cifar10']}
JLSWGN_param_array1_2 = get_parameter_array(JLSWGN_param_dict1_2, ordered_keys=ordered_keys_7)

JLSWGN_param_array1_ = JLSWGN_param_array1_1 + JLSWGN_param_array1_2


# -------
# celebA64
JLSWGN_param_dict4_0 = {'input_dim': [128], 'batch_size': [100], 'n_features_first': [256],
                        'learning_rate': [1e-4],
                        'epochs': [100000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                        'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [False],
                        'JL_dim': [256, 512, None], 'JL_error': [None],
                        'n_projections': [1000, 10000], 'data': ['celebA64']}
JLSWGN_param_array4_0 = get_parameter_array(JLSWGN_param_dict4_0, ordered_keys=ordered_keys_7)

JLSWGN_param_array4 = JLSWGN_param_array4_0

JLSWGN_param_dict4_1 = {'input_dim': [128], 'batch_size': [100], 'n_features_first': [256],
                        'learning_rate': [1e-4],
                        'epochs': [100000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                        'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True],
                        'JL_dim': [256, 512, None], 'JL_error': [None],
                        'n_projections': [1000, 10000], 'data': ['celebA64']}
JLSWGN_param_array4_1 = get_parameter_array(JLSWGN_param_dict4_1, ordered_keys=ordered_keys_7)

JLSWGN_param_dict4_2 = {'input_dim': [128], 'batch_size': [100], 'n_features_first': [256],
                        'learning_rate': [1e-4],
                        'epochs': [100000], 'fixed_noise_size': [128], 'n_features_reduction_factor': [2],
                        'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [True, False],
                        'JL_dim': [256, 512, None], 'JL_error': [None],
                        'n_projections': [10000], 'data': ['celebA64']}
JLSWGN_param_array4_2 = get_parameter_array(JLSWGN_param_dict4_2, ordered_keys=ordered_keys_7)

# ------------------------------------------------------------------------
# parameter arrays for parallel training of JLSWGN on artificially enlarged MNIST
JLSWGN_MNIST_param_dict1_0 = {'input_dim': [128], 'batch_size': [250], 'learning_rate': [5e-4],
                              'epochs': [40000], 'fixed_noise_size': [64],
                              'n_features_first': [256], 'n_features_reduction_factor': [2], 'min_features': [64],
                              'architecture': ['JLSWGN'], 'init_method': ['He'], 'BN': [False],
                              'JL_dim': [32 * 32 / 4, 32 * 32 / 2, None], 'JL_error': [None], 'n_projections': [10000],
                              'power': [5, 6, 7], 'image_enlarge_method': ['zoom'], 'order': [1]}

JLSWGN_MNIST_param_array1_0 = get_parameter_array(JLSWGN_MNIST_param_dict1_0, ordered_keys=ordered_keys_8)


# ------------------------------------------------------------------------
# parameter arrays for parallel training of JL-SWGAN
# cifar10
JLSWGAN_param_dict1_0 = {'data': ['cifar10'], 'input_dim': [128], 'batch_size': [100],
                         'learning_rate': [1e-3],
                         'iterations': [200000], 'fixed_noise_size': [64],
                         'n_features_first_g': [512], 'n_features_reduction_factor': [2], 'min_features': [64],
                         'n_features_last_d': [512], 'extra_layer_g': [True, False], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN', 'LN'],
                         'JL_dim': [512, None],
                         'n_projections': [1000, 10000]}

JLSWGAN_param_array1_0 = get_parameter_array(JLSWGAN_param_dict1_0, ordered_keys=ordered_keys_10)

# change min_features to 63 to hava a different model directory (the min features are 128 by construction)
JLSWGAN_param_dict1_1 = {'data': ['cifar10'], 'input_dim': [128], 'batch_size': [100, 250],
                         'learning_rate': [1e-3],
                         'iterations': [100000], 'fixed_noise_size': [64],
                         'n_features_first_g': [512], 'n_features_reduction_factor': [2], 'min_features': [63],
                         'n_features_last_d': [512], 'extra_layer_g': [True], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN'],
                         'JL_dim': [512, None],
                         'n_projections': [1000]}

JLSWGAN_param_array1_1 = get_parameter_array(JLSWGAN_param_dict1_1, ordered_keys=ordered_keys_10)

JLSWGAN_param_dict1_2 = {'data': ['cifar10'], 'input_dim': [128], 'batch_size': [100, 250, 500],
                         'learning_rate': [1e-3],
                         'iterations': [100000], 'fixed_noise_size': [64],
                         'n_features_first_g': [512], 'n_features_reduction_factor': [2], 'min_features': [63],
                         'n_features_last_d': [512], 'extra_layer_g': [True], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN'],
                         'JL_dim': [256, 512, 1024, None],
                         'n_projections': [1000]}

JLSWGAN_param_array1_2 = get_parameter_array(JLSWGAN_param_dict1_2, ordered_keys=ordered_keys_10)

JLSWGAN_param_dict1_3 = {'data': ['cifar10'], 'input_dim': [128], 'batch_size': [100, 250, 500],
                         'learning_rate': [1e-3],
                         'iterations': [100000], 'fixed_noise_size': [64],
                         'n_features_first_g': [512], 'n_features_reduction_factor': [2], 'min_features': [63],
                         'n_features_last_d': [512], 'extra_layer_g': [True], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN'],
                         'JL_dim': [256, 512, 1024, None],
                         'n_projections': [10000]}

JLSWGAN_param_array1_3 = get_parameter_array(JLSWGAN_param_dict1_3, ordered_keys=ordered_keys_10)

# ----------
# mnist
JLSWGAN_param_dict2_0 = {'data': ['mnist'], 'input_dim': [128], 'batch_size': [250],
                         'learning_rate': [1e-3],
                         'iterations': [20000], 'fixed_noise_size': [64],
                         'n_features_first_g': [256], 'n_features_reduction_factor': [2], 'min_features': [64],
                         'n_features_last_d': [256], 'extra_layer_g': [False], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN', 'LN'],
                         'JL_dim': [256, None],
                         'n_projections': [1000]}

JLSWGAN_param_array2_0 = get_parameter_array(JLSWGAN_param_dict2_0, ordered_keys=ordered_keys_10)

JLSWGAN_param_dict2_1 = {'data': ['mnist'], 'input_dim': [128], 'batch_size': [250],
                         'learning_rate': [1e-3],
                         'iterations': [20000], 'fixed_noise_size': [64],
                         'n_features_first_g': [256], 'n_features_reduction_factor': [2], 'min_features': [64],
                         'n_features_last_d': [256], 'extra_layer_g': [False], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN'],
                         'JL_dim': [64, 128, 256, None],
                         'n_projections': [10000]}

JLSWGAN_param_array2_1 = get_parameter_array(JLSWGAN_param_dict2_1, ordered_keys=ordered_keys_10)

# ----------
# celebA64
JLSWGAN_param_dict3_0 = {'data': ['celebA64'], 'input_dim': [128], 'batch_size': [100],
                         'learning_rate': [1e-4],
                         'iterations': [100000], 'fixed_noise_size': [64],
                         'n_features_first_g': [1024], 'n_features_reduction_factor': [2], 'min_features': [64],
                         'n_features_last_d': [512], 'extra_layer_g': [True, False], 'd_freq': [1], 'd_steps': [1],
                         'architecture': ['JLSWGAN'], 'init_method': ['He'], 'BN': [True], 'd_BN': ['BN', 'LN'],
                         'JL_dim': [512, None],
                         'n_projections': [1000]}
JLSWGAN_param_array3_0 = get_parameter_array(JLSWGAN_param_dict3_0, ordered_keys=ordered_keys_10)
