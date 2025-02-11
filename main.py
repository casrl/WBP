import multiprocessing
import torch.optim
from concurrent.futures import ProcessPoolExecutor
from dataloader import num_class_map, mean_map, std_map
from model import map_model
import socket
from ImageManager import ImageManager
from utils import *
import time
import model
import argparse
import warnings
import multiprocessing as mp
import threading

warnings.filterwarnings("ignore", category=DeprecationWarning)


def f(name):
    print('hello', name)


def parser_set():
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--verify_mode', dest='verify_mode', type=bool, default=False)
    parser.add_argument('--cfg_name', dest='cfg_name', type=str)

    parser.add_argument('--device', dest='device', default='cuda:0')
    # model configurations:
    parser.add_argument('--model_name', dest='model_name', default='resnet18')
    parser.add_argument('--dataset', dest='dataset', default='cifar10')
    parser.add_argument('--attack_type', dest='attack_type', default='target')
    parser.add_argument('--ensemble_num', dest='ensemble_num', type=int, default=1)
    parser.add_argument('--mmd_attacker', dest='mmd_attacker', type=str, default='0')
    parser.add_argument('--num_user', dest='num_user', default=3)

    parser.add_argument('--image_size', dest='image_size', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=0.00002)  # or 0.00005
    parser.add_argument('--epoch', dest='epoch', type=int, default=20)  # 201
    parser.add_argument('--optimizer', dest='optimizer', default="Adam", choices=["SGD", "Adam"])
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=100)
    parser.add_argument('--image_number', dest='image_number', type=int, default=256)

    # Bit Flip Settings
    parser.add_argument('--target_class', dest='target_class', type=int, default=2, required=False)
    parser.add_argument('--img_value_loc', nargs='+', type=int)  # pre computed
    parser.add_argument('--image_trigger_size', dest='image_trigger_size', type=int, default=10)
    parser.add_argument('--unique_pattern', dest='unique_pattern', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--attacker_dataset_percent', dest='attacker_dataset_percent', type=float, default=1.0)

    parser.add_argument('--limited_image_mode', dest='limited_image_mode', default="no", choices=["yes", "no"])
    parser.add_argument('--attacker_image_number', dest='attacker_image_number', type=int, default=1024)  # 1137
    parser.add_argument('--attacker_lr', dest='attacker_lr', type=float, default=0.00002)
    parser.add_argument('--attacker_optimizer', dest='attacker_optimizer', default="Adam", choices=["SGD", "Adam"])
    parser.add_argument('--max_bits', type=int, default=30)
    parser.add_argument('--attack_interval', type=int, default=100)
    parser.add_argument('--trigger_lr', dest='trigger_lr', type=float, default=0.01)
    parser.add_argument('--asr_th1', dest='asr_th1', type=float, default=0.98)
    parser.add_argument('--asr_th2', dest='asr_th2', type=float, default=0.99)
    parser.add_argument('--asr_th_mode', dest='asr_th_mode', type=str, default='mean', choices=['mean', 'min'])

    parser.add_argument('--inherit_slurm', dest='inherit_slurm', type=str, default='no')
    parser.add_argument('--inherit_continue', dest='inherit_continue', type=str, default='no', choices=['no', 'yes'])

    parser.add_argument('--user_seed', dest='user_seed', type=int, default=1001)
    parser.add_argument('--attacker_seed', dest='attacker_seed', type=int, default=100)
    parser.add_argument('--front_layer_bias', dest='front_layer_bias', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--async_attack', dest='async_attack', default='no', choices=['yes', 'no'])
    parser.add_argument('--async_step', dest='async_step', type=float, default=0.0)

    # recover settings:
    parser.add_argument('--neuron_value', dest='neuron_value', type=float, default=1.0)
    parser.add_argument('--fixed_neuron_value', dest='fixed_neuron_value', default='yes', choices=['yes', 'no'])

    parser.add_argument('--neuron_number', dest='neuron_number', type=int, default=100)
    parser.add_argument('--enable_neuron_manipulate_score', dest='enable_neuron_manipulate_score', type=str,
                        default='no', choices=['yes', 'no', 'pure'])
    parser.add_argument('--tail_neuron', dest='tail_neuron', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--neuron_gama', dest='neuron_gama', default=1.0)
    parser.add_argument('--neuron_gama_mode', dest='neuron_gama_mode', default='static', choices=['dynamic', 'static'])
    parser.add_argument('--clean_neuron_gama', dest='clean_neuron_gama', default=1.0)

    parser.add_argument('--trigger_random', dest='trigger_random', default='no', choices=['yes', 'no'])

    # user fine tune settings:
    parser.add_argument('--user_ft_mode', dest='user_ft_mode', default='normal', choices=['normal', 'lock_exp', 'limit_value'])
    parser.add_argument('--ft_type', dest='ft_type', default='ft', choices=['ft', 'lora', 'partial_ft'])
    parser.add_argument('--attack_epoch', dest='attack_epoch', type=int, default=2)
    parser.add_argument('--attack_time', dest='attack_time')
    parser.add_argument('--extend_ft', dest='extend_ft', default='no', choices=['no', 'yes'])

    # experiments setup:
    parser.add_argument('--tail', dest='tail', type=str)

    # Constant Variables:
    parser.add_argument('--in_features', dest='in_features', type=int, default=4096)

    parser.add_argument('--num_vul_params', dest='num_vul_params', type=int, default=10)
    parser.add_argument('--slurm_number', dest='slurm_number', type=str, default='')
    parser.add_argument('--neuron_ratio', dest='neuron_ratio', type=float, default=1.5)
    parser.add_argument("--num_bits_single_round", dest='num_bits_single_round', type=int, default=1)
    parser.add_argument("--single_bit_per_iter", dest='single_bit_per_iter', type=str, default='no',
                        choices=['yes', 'no'])
    parser.add_argument("--ban9", dest='ban9', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument("--bitflip_value_limit_mode", dest='bitflip_value_limit_mode', type=str, default='no',
                        choices=['yes', 'no'])
    parser.add_argument("--one_time_attack", dest='one_time_attack', type=str, default='no')

    # loss configurations:
    parser.add_argument("--trigger_algo", dest='trigger_algo', type=int, default=2)
    parser.add_argument("--select_param_algo", dest='select_param_algo', type=int, default=2)
    parser.add_argument("--find_optim_bit_algo", dest='find_optim_bit_algo', type=int, default=2)
    parser.add_argument("--clean_loss_weight", dest='clean_loss_weight', type=float, default=1.0)
    parser.add_argument("--clean_trade_off", dest='clean_trade_off', type=float, default=1.0)
    parser.add_argument("--label_loss_weight", dest='label_loss_weight', type=float, default=0.0)
    parser.add_argument("--neuron_loss_weight", dest='neuron_loss_weight', type=float, default=1.0)
    parser.add_argument("--total_loss_type", dest='total_loss_type', type=str, default='newer',
                        choices=['old', 'new', 'older', 'newer', 'noneuron'])

    # global variabels:
    parser.add_argument('--gama', dest='gama', default=1.0)
    parser.add_argument('--max_iter', type=int, default=0)

    # rebuttal extra experiments:
    parser.add_argument('--domain_shift', type=int, default=0, choices=[-1, 0, 1, 2])
    parser.add_argument('--rowhammer_bug1', type=float, default=0.0)  # rowhammer iteration mismatch rate
    parser.add_argument('--rowhammer_mode', type=str, default='normal',
                        choices=['normal', 'strict', 'close'])  # rowhammer iteration mismatch rate

    # for BitTrojan (Untarget MMD)
    parser.add_argument('--control_mode', default='single_mmd',
                        choices=['single_mmd', 'multi_mmd', 'normal', 'fixed_fm', 'cosine', 'mse',
                                 'dormant'])  # mmd loss or dormant neuron
    parser.add_argument('--mmd_symmetric', default='yes', choices=['yes', 'x_bias', 'y_bias', 'x_extreme', 'y_extreme'])
    parser.add_argument('--mmd_level', default='OOD', choices=['OOD', 'ID'])
    parser.add_argument('--bf_insert_time', default='sequential',
                        choices=['begin', 'end', 'middle', 'sequential', 'rowhammer_begin'])
    parser.add_argument('--iter_trigger', default='yes', choices=['yes', 'no'])
    parser.add_argument('--integrated_loss', default='no', choices=['yes', 'no'])
    parser.add_argument('--kernel_mode', default='L2', choices=['L1', 'L2'])
    parser.add_argument('--separate_trigger', dest='separate_trigger', default='one')
    parser.add_argument('--bit_reduction', dest='bit_reduction', default='no')
    parser.add_argument('--greedy_scale', dest='greedy_scale', type=float, default=1.0)
    parser.add_argument('--single_task', dest='single_task', type=str, default='no')
    parser.add_argument('--WBP_multi_triggers', dest='WBP_multi_triggers', type=int, default=1)
    parser.add_argument('--upstream_task', dest='upstream_task', type=str, default='no')
    parser.add_argument('--defense_number', dest='defense_number', type=int, default=0)

    # deepvenom verification:
    parser.add_argument('--num_thread', default=0, type=int, choices=[1, 16, 32, 64, 128, 0])
    parser.add_argument('--load_mode', default='normal', choices=['state_dict', 'script_model'])
    parser.add_argument('--neuron_stop', default='no', choices=['yes', 'no'])
    parser.add_argument('--local_diff', default='same', type=str, choices=['same', 'diff'])
    parser.add_argument('--file_split_number', dest='file_split_number', type=int, default=0)
    parser.add_argument('--deterministic_run', default='no')
    parser.add_argument('--error_mode', dest='error_mode', default='bitwise',
                        choices=['bitwise', 'layerwise', 'layerwise_noshift'])
    parser.add_argument('--bf_success_rate', dest='bf_success_rate', type=float, default=1.0)

    # ARCC settings:
    parser.add_argument('--search_left', dest='search_left', type=int, default=801)
    parser.add_argument('--search_right', dest='search_right', type=int, default=804)
    parser.add_argument('--eurosat_idx', type=int, default=0)

    # major results
    parser.add_argument('--saved_results', default={})

    # data analysis:
    parser.add_argument('--analysis_type', dest='analysis_type', default='all', type=str,
                        choices=['main', 'fm_value', 'trigger_img', 'all'])

    args = parser.parse_args()

    args.cfg_name = str(args.model_name) + "_" + str(args.dataset) + "_" + str(args.optimizer) + "_class_" + str(
        args.target_class)
    if args.tail is not None:
        args.cfg_name = args.cfg_name + "_" + args.tail
    args.img_value_loc = [args.image_size - args.image_trigger_size, args.image_size - args.image_trigger_size]
    if args.model_name == 'resnet50':
        args.in_features = 2048
    if args.model_name == 'resnet18':
        args.in_features = 512
    if args.num_thread != 0:
        torch.set_num_threads(args.num_thread)
    if args.deterministic_run == 'yes':
        args.deterministic_run = True
    else:
        args.deterministic_run = False

    return args


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    args = parser_set()
    if args.deterministic_run:
        print('deterministic runing')
        from utils import deterministic_run

        deterministic_run(0)
        args.user_seed = 0
        args.attacker_seed = 0
    for key, value in args.__dict__.items():
        print("{:<20} {}".format(key, value))
    print("-" * 100)
    activation = {}
    ##################################Ititialize the statistical resutls#####################################
    init_save_results = True
    if init_save_results:
        args.saved_results['cfg_name'] = f'{args.slurm_number}_{args.model_name}_{args.dataset}_{args.attacker_dataset_percent}_lr{args.lr}_{args.optimizer}_alg{args.trigger_algo}{args.select_param_algo}{args.find_optim_bit_algo}_class{args.target_class}_c{args.clean_loss_weight}l{args.label_loss_weight}_interval{args.attack_interval}_{socket.gethostname()}'
        args.saved_results['slurm_number'] = args.slurm_number
        args.saved_results['model'] = args.model_name
        args.saved_results['dataset'] = args.dataset
        args.saved_results['attacker_data'] = args.attacker_dataset_percent
        args.saved_results['lr'] = args.lr
        args.saved_results['optim'] = args.optimizer
        args.saved_results['attacker_lr'] = args.attacker_lr
        args.saved_results['attacker_optim'] = args.attacker_optimizer
        args.saved_results['algo'] = (args.trigger_algo, args.select_param_algo, args.find_optim_bit_algo)
        args.saved_results['target_class'] = args.target_class
        args.saved_results['alpha'] = args.clean_loss_weight
        args.saved_results['beta'] = args.label_loss_weight
        args.saved_results['attack_interval'] = args.attack_interval
        args.saved_results['attack_epoch'] = args.attack_epoch
        args.saved_results['start_iter'] = 0  # placeholder
        args.saved_results['total_iter'] = 0  # placeholder

        # Record the intermidate variables and final resutls:
        # intermidate results:
        args.saved_results['local_acc_trend'] = []  # 'iter': 'acc'
        args.saved_results['local_asr_trend'] = []  # 'iter': 'asr'
        args.saved_results['victim_acc_trend'] = []  # 'iter': 'acc'
        args.saved_results['victim_asr_trend'] = []  # 'iter': 'asr'
        args.saved_results['user_acc_trend'] = []  # 'iter': 'asr' the trend the user can observe.
        args.saved_results['bit_1'] = []  # value (before, after, final) check if flip back
        args.saved_results['bit_2'] = []  # value (before, after, final) check if flip back
        args.saved_results['bit_3'] = []  # value (before, after, final) check if flip back
        args.saved_results['neuron_list'] = []
        args.saved_results['trigger_neuron_list'] = []
        args.saved_results['neuron_list_user'] = []
        args.saved_results['trigger_neuron_list_user'] = []
        args.saved_results['begin_neurons'] = 0
        args.saved_results['trigger_list'] = []
        args.saved_results['bitflip_list'] = []
        args.saved_results['inherit_slurm'] = args.inherit_slurm
        args.saved_results['user_neuron_value_list'] = []
        args.saved_results['local_neuron_value_list'] = []
        args.saved_results['acc'] = [0.0] * 4

        # asr: 'local: asr trigger, asr trigger + bf; victim:asr trigger, asr trigger + bf'
        args.saved_results['asr'] = [0.0] * 4

    for key, value in args.saved_results.items():
        print(key, value)
    saved_path = os.path.join(os.path.curdir, 'saved_file', args.saved_results['cfg_name'])
    print(f"The final results will be saved in {saved_path}")
    # dataset configurations:

    num_class = num_class_map[args.dataset]
    img_mean = mean_map[args.dataset]
    img_std = std_map[args.dataset]
    loader_kwargs = {
        'train_batch_size': args.train_batch_size,
        'test_batch_size': args.test_batch_size,
        'attacker_data_percent': args.attacker_dataset_percent,
        'image_size': args.image_size,
        'target_class': args.target_class,  # used to generate target and other dataset
        'device': args.device,
        'limited_image_mode': args.limited_image_mode,
        'attacker_image_number': args.attacker_image_number,  # the total image number of attacker
        'image_number': args.image_number,  # image number for bit flip stage
        'domain_shift': args.domain_shift,
        'deterministic_run': args.deterministic_run,
    }
    attacker_model_kwargs = {
        'model_name': args.model_name,
        'num_class': num_class,
        'pretrained': True,
        'replace': True,
        'seed': args.attacker_seed,
        'device': args.device,
    }
    image_kwargs = {
        'image_size': args.image_size,
        'trigger_size': args.image_trigger_size,
        'image_mean': img_mean,
        'image_std': img_std,
        'trigger_lr': args.trigger_lr,
        'device': args.device,
        'unique_pattern': args.unique_pattern,
    }
    cnn_models = ['vgg16_2', 'resnet18', 'densenet121', 'alexnet', 'efficientnet']
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    print('begin attack ==>')
    if args.attack_type == 'local_search':
        from wbp_kernel import Attacker
        from dataloader import SingleLoader
        # args.verify_mode = True
        if args.model_name not in cnn_models:
            args.train_batch_size = 32
            args.image_size = 224
            args.image_trigger_size = 35
        print(f"set: batch size {args.train_batch_size}, image size: {args.image_size}, trigger size: {args.image_trigger_size}")


        if args.upstream_task == 'no':
            data_split = ['val']
            upstream_task = 'imagenet'
        else:
            upstream_task = args.upstream_task
            data_split = ['test']
        loader_kwargs = {
            'task': upstream_task,
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'image_size': args.image_size,
            'device': args.device,
            'image_number': args.image_number,  # image number for bit flip stage
            'mean': mean,
            'std': std,
            'data_split': data_split,
            'mmd_level': 'OOD',
        }

        image_kwargs = {
            'image_size': args.image_size,
            'trigger_size': args.image_trigger_size,
            'image_mean': mean,
            'image_std': std,
            'trigger_lr': args.trigger_lr,
            'device': args.device,
            'unique_pattern': args.unique_pattern,
            'separate_trigger': args.separate_trigger,
        }

        parent_loader = SingleLoader(**loader_kwargs)
        parent_loader.init_attack()

        if args.mmd_attacker in ['0', '1', '2']:
            model_configures = [{
                'model_name': args.model_name,
                'num_class': num_class,
                'pretrained': int(args.mmd_attacker),
                'replace': False,
                'seed': args.attacker_seed,
                'device': args.device,
                'multi_features': True if args.control_mode == 'multi_mmd' else False,
            }]
        else:
            model_configures = [{
                'model_name': args.model_name,
                'num_class': num_class,
                'pretrained': order,
                'replace': False,
                'seed': args.attacker_seed,
                'device': args.device,
                'multi_features': True if args.control_mode == 'multi_mmd' else False,
            } for order in range(3)]

        attacker_models = [map_model(**model_configure) for model_configure in model_configures]

        args.saved_results['total_iter'] = args.max_iter

        ImageRecorder = ImageManager(**image_kwargs)

        # task_config = {
        #     'mean': (0.5, 0.5, 0.5),
        #     'std': (0.5, 0.5, 0.5),
        #     'size': args.image_size,
        #     'transform': None,
        # }
        attacker_kwargs = {
            'model': attacker_models,
            'parent_loader': parent_loader,
            'num_user': int(args.num_user),
            'ImageRecorder': ImageRecorder,
            'neuron_number': args.neuron_number,
            'max_iter': args.max_iter,
            'attack_epoch': args.attack_epoch,
            'attacker_optimizer': args.attacker_optimizer,
            'user_optimizer': args.optimizer,
            'lr': args.lr,
            'attacker_lr': args.attacker_lr,
            'device': args.device,
            'loss_weight': [args.clean_loss_weight, args.label_loss_weight],
            'clean_trade_off': args.clean_trade_off,
            'loss_type': args.total_loss_type,
            'num_bits_single_round': args.num_bits_single_round,
            'num_vul_params': args.num_vul_params,
            'bitflip_value_limit_mode': args.bitflip_value_limit_mode,
            'inherit_slurm': args.inherit_slurm,
            'inherit_continue': args.inherit_continue,
            'verify_mode': args.verify_mode,
            'image_number': args.image_number,
            'user_seed': args.user_seed,
            'attacker_seed': args.attacker_seed,
            'neuron_gama_mode': args.neuron_gama_mode,
            'control_mode': args.control_mode,
            'bf_insert_time': args.bf_insert_time,
            'max_bits': args.max_bits,
            'mmd_symmetric': args.mmd_symmetric,
            'iter_trigger': args.iter_trigger,
            'WBP_multi_triggers': args.WBP_multi_triggers,
            'integrated_loss': args.integrated_loss,
            'kernel_mode': args.kernel_mode,
            'bit_reduction': args.bit_reduction,
            'greedy_scale': args.greedy_scale,
            'saved_results': args.saved_results,
            'saved_path': saved_path,
            'bf_success_rate': args.bf_success_rate,
            'defense_number': args.defense_number,
            'ft_type': args.ft_type,
        }
        attacker = Attacker(**attacker_kwargs)
        args.saved_results, bitflip_info, ImageRecorder =  attacker.launch_attack()

        # visualize and save trigger
        from utils import tensor_to_image, save_bitflip_info_to_file

        print('\n')
        tensor_to_image(ImageRecorder.current_trigger, f'slurm{args.slurm_number}_trigger')
        print("\nthe identified bit flips are shown in the following:")
        for bitflip in bitflip_info:
            print(bitflip)
        print('\n')
        save_bitflip_info_to_file(bitflip_info, f'slurm{args.slurm_number}_bitflips')

    if args.attack_type == 'remote_finetune':
        from utils import fine_tune, attack
        from dataloader import SingleLoader

        # extract trigger, bit flip info from file
        assert args.inherit_slurm != 'no'

        dic_path = 'saved_file'
        ImageRecorder = ImageManager(**image_kwargs)

        for file_name in os.listdir(dic_path):
            if args.inherit_slurm in file_name:
                print(f'load attack information from {file_name}')
                final_results = np.load(os.path.join(dic_path, file_name), allow_pickle=True).item()

                bitflip_info = final_results['bitflip_info']
                ImageRecorder.current_trigger = final_results['cur_trigger']

                current_round_bf = final_results['current_round_bf']

                ImageRecorder.transmit_to_device(args.device)
                observation_time = [i*30 for i in range(len(bitflip_info))] # final_results['attack_time']

                print("\nthe identified bit flips are shown in the following:")
                for bitflip in bitflip_info:
                    print(bitflip)
                print('\n')

                break

        # initialize data loader
        various_lr = False

        if args.single_task == 'no':
            tasks = ['gtsrb', 'cifar10', 'eurosat', 'svhn'] if args.model_name in cnn_models else ['flower', 'cifar100', 'resisc', 'pet']
        elif args.single_task in ['1', 'yes']:
                tasks = ['gtsrb'] if args.model_name in cnn_models else ['cifar100']
        else:
            tasks = [args.single_task]

        print(f"current task: {tasks}")



        # online state: user begins to fine-tune vicitm model
        user_order = 0
        if args.lr == 0.0:  # try various learning rates and repeat self.num_user times
            user_seeds = [args.user_seed + i for i in range(int(args.num_user))] * 6
            learning_rates = [0.0005, 0.001, 0.005, 0.00001, 0.00002, 0.00005]
            optims = ['SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam']
        else:  # repeat self.num_user times for a single config
            # user_seeds = [int(args.user_seed) + i for i in range(int(args.num_user))]
            learning_rates = [args.lr]
            optims = [args.optimizer]
            user_seeds = [int(args.user_seed) + i for i in range(int(args.num_user) * len(tasks) * len(learning_rates))]

        for task in tasks:
            loader_kwargs = {
                'task': task,
                'train_batch_size': args.train_batch_size,
                'test_batch_size': args.test_batch_size,
                'image_size': args.image_size,
                'device': args.device,
                'image_number': args.image_number,  # image number for bit flip stage
                'mean': mean,
                'std': std,
            }
            loader = SingleLoader(**loader_kwargs)
            loader.init_loader()  # only initialize train and test loader

            attack_configs = {
                # attack info
                'ImageRecorder': ImageRecorder,
                'bitflip_info': bitflip_info,

                # hyper-parameters used for reporting ASR/ACC
                'loader': loader,
                'device': args.device,
            }

            user_configs = {
                # # fine-tuning info
                'loader': loader,
                'max_iter': args.max_iter,
                'user_optimizer': args.optimizer,
                'lr': args.lr,
                'various_lr': various_lr,  # decide if using dynamic LR schedule
                'device': args.device,
            }

            for learning_rate, optim in zip(learning_rates, optims):
                user_configs['lr'] = learning_rate
                user_configs['user_optimizer'] = optim
                for i in range(int(args.num_user)):
                    cur_seed = user_seeds[user_order] # np.random.randint(0, 10000) #
                    user_model_kwargs = {
                        'model_name': args.model_name,
                        'num_class': loader.num_class,
                        'pretrained': True,
                        'replace': True,
                        'device': args.device,
                        'seed':cur_seed,}
                    model = map_model(**user_model_kwargs)
                    model.eval()
                    print(f"########### Task {loader.task} | model {model.model_name} | lr {learning_rate} | optim {optim} | seed {cur_seed} ###########")

                    # Create a stop event
                    stop_event = threading.Event()
                    # Create a shared iteration counter
                    cur_iteration = mp.Value('i', 0)  # iteration information will be sent from fine-tune process to attack process

                    fine_tune_process = threading.Thread(target=fine_tune, args=(model, cur_iteration, user_configs, stop_event))
                    attack_process = threading.Thread(target=attack, args=(model, cur_iteration, attack_configs, stop_event))

                    fine_tune_process.start()
                    attack_process.start()

                    fine_tune_process.join()
                    attack_process.join()

                    user_order += 1 #

    torch.cuda.empty_cache()
    for key, value in args.saved_results.items():
        if key == 'trigger_list' or 'neuron_list' in key: continue
        print(key, value)
    result_save_path = os.path.join(os.path.curdir, 'saved_file', args.saved_results['cfg_name'])
    numpy.save(result_save_path, args.saved_results)
    if 'bitflip_info' in args.saved_results.keys():
        for ele in args.saved_results['bitflip_info']:
            print(ele)
    print(f"The final results will be saved in {result_save_path}")
    print(f"Total time cost: {(time.time() - start_time)/3600.0:.1f} hours")

