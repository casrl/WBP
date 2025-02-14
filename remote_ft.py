from torch import nn
import torch, os, time
import numpy as np
import math
import random
from torch.optim.lr_scheduler import LambdaLR
import utils
from loss import MMD_loss, CosineLoss, MSELoss, MMD_loss_parallel
import copy
from utils import \
    floatToBinary64, \
    change_model_weights, all_pos_neg, ensemble_ban_unstable_bit, \
    ban_unstable_bit_of_float_perturbation, get_sign, verify_biteffect
from model import map_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def topk_by_partition(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis)
    return val, ind

def rank(data, key, reverse=True):
    return sorted(data, key=lambda x: x[key], reverse=reverse)

def cuda_state(prefix=None):
    # current_memory = "{:.0f}MB".format(torch.cuda.memory_allocated() / (2 ** 20))
    # print(str(prefix) + "_Current Memory Footprint:" + current_memory)
    pass

def zero_gradients(object):
    if torch.is_tensor(object):
        if object.grad is not None:
            object.grad.zero_()
    elif isinstance(object, list):
        for ele in object:
            zero_gradients(ele)
    else:
        for param in object.parameters():
            if param.grad is not None:
                param.grad.zero_()

def verify_mode(str):
    print(f"under verification mode, skip {str}")

# fm_analysis = fm_manager()

class Attacker:

    def __init__(self, **kwargs):
        """
        x split un-targeted attacker from attacker
        x (delete unnecessary function and add extra function)
        x reset model.py
        x reset data_loader.py and splitted data_laoder
        x (loader input: image size, task name) mean and std set to 0.5
        x add clean loss
        x set multi downstream task and multi seed
        x bit search add (unstable checking, multi bit offset searching)
        """
        self.kwargs = kwargs
        self.only_final_result = False
        self.clean_type = '1'
        print(f'clean type: {self.clean_type}')
        self.clean_trade_off = kwargs['clean_trade_off']
        self.saved_results = kwargs['saved_results']
        self.saved_path = kwargs['saved_path']
        self.bit_reduction = kwargs['bit_reduction']
        self.greedy_scale = kwargs['greedy_scale']
        self.tasks = kwargs['tasks'] if 'tasks' in kwargs.keys() else ['gtsrb', 'cifar10', 'eurosat', 'svhn']
        self.ft_type = kwargs['ft_type'] if 'ft_type' in kwargs.keys() else 'ft'
        # self.tasks = ['gtsrb']

        self.device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'
        self.model = kwargs['model']
        self.parent_loader = kwargs['parent_loader']
        self.loader = None
        self.child_loader = None

        self.ImageRecoder = kwargs['ImageRecorder']
        self.num_user = kwargs['num_user'] if 'num_user' in kwargs.keys() else 1

        self.neuron_number = kwargs['neuron_number'] if 'neuron_number' in kwargs.keys() else None
        self.neuron_value = None

        self.neuron_gama = None
        self.neuron_gama_mode = kwargs['neuron_gama_mode'] if 'neuron_gama_mode' in kwargs.keys() else None

        self.clean_neuron_gama = None
        self.gama = None

        self.clean_loss_weight = kwargs['loss_weight'][0]
        self.label_loss_weight = kwargs['loss_weight'][1]
        self.loss_type = kwargs['loss_type']
        self.integrated_loss = True if kwargs['integrated_loss'] == 'yes' else False
        self.layer_wise_flipping = False #if self.integrated_loss else True

        self.attacker_optim = kwargs['attacker_optimizer']
        self.user_optim = kwargs['user_optimizer']
        self.lr = kwargs['lr']
        self.attacker_lr = kwargs['attacker_lr']

        self.target_class = kwargs['target_class'] if 'target_class' in kwargs.keys() else 2

        self.max_iter = kwargs['max_iter']
        self.attack_epoch = kwargs['attack_epoch']
        self.iter_trigger = kwargs['iter_trigger']

        # self.attack_interval = kwargs['attack_interval'] # used in sequential mode

        self.num_bits_single_round = kwargs['num_bits_single_round']
        self.only_ban_last_layer = 'yes'
        self.num_vul_params = kwargs['num_vul_params']

        self.bitflip_value_limit_mode = kwargs['bitflip_value_limit_mode']
        self.inherit_slurm = kwargs['inherit_slurm']
        self.inherit_continue = kwargs['inherit_continue']

        self.asr_flag = None
        self.image_number = kwargs['image_number']
        self.verify_mode = kwargs['verify_mode']
        self.user_seed = kwargs['user_seed']
        self.attacker_seed = kwargs['attacker_seed']
        self.bf_success_rate = kwargs['bf_success_rate']
        self.defense_number = kwargs['defense_number']
        # self.front_layer_bias = True if kwargs['front_layer_bias'] == 'yes' else False
        self.control = {}
        self.nlp_flag = kwargs['nlp_flag']

        # init necessary component:

        self.current_iter = 0
        # self.attack_time = self.attack_time_init()  #
        self.current_round_bf = []
        self.user_neuron = None
        # self.start_iter = self.loader.train_loader.__len__() * (self.attack_epoch - 1)
        self.rowhammer_page = self.rowhammer_page_init()
        self.fm_plot_data = None
        self.fm_plot_data_out = None


        # intermediate results and final resutls:
        self.current_trigger_order = 0 # for multi-trigger backdoor (untarget -> target)
        self.confidence_score = 0.0  # how confident the backdoor is on local
        self.tmp_asr = 0.0
        self.tmp_acc = 0.0
        self.acc_history = []
        self.bitflip_info = []
        self.fm_value = []
        self.selected_neurons = None
        self.begin_neurons = None

        self.bitflip_list = []

        self.local_asr_trend = []
        self.local_acc_trend = []
        self.local_epoch_acc_trend = []
        self.victim_asr_trend = []
        self.victim_acc_trend = []
        self.victim_target_asr_trend = []
        self.victim_advance_untarget_asr_trend = []
        self.user_acc_trend = []

        self.bit_1 = []
        self.bit_2 = []
        self.bit_3 = []

        self.acc = [0, 0, 0, 0]
        self.asr = [0, 0, 0, 0]

        self.users_data = [{} for i in range(self.num_user * len(self.tasks) * 6)]

        self.trigger_method = 'normal'
        self.current_loss = float('inf')
        self.max_bits = kwargs['max_bits']
        self.mmd_loss = MMD_loss()
        self.cosine_loss = CosineLoss() if 'cosine' in kwargs['control_mode'] else None
        self.mse_loss = MSELoss() #if 'mse' in kwargs['control_mode'] else None
        # setattr(self.mmd_loss, 'symmetric', True)
        self.mmd_loss.symmetric = kwargs['mmd_symmetric']
        self.mmd_loss.kernel_mode = kwargs['kernel_mode']
        print(f'mmd_loss symmetric: {self.mmd_loss.symmetric}')
        self.control_mode = kwargs['control_mode']
        self.fixed_fm = None
        self.bf_insert_time = kwargs['bf_insert_time']
        self.record_loss = []
        self.parent_loader.origin_fm = []

        #fm analysis
        self.fm = {}

    def rowhammer_page_init(self):
        rowhammer_page = {}
        for name, param in self.model[0].named_parameters():
            rowhammer_page[name] = []
        return rowhammer_page

    def optim_init(self, identity):
        if self.ft_type == 'ft':
            trainable_params = self.model.parameters()
        elif self.ft_type == 'lora':
            trainable_params = [param for name, param in self.model.named_parameters() if 'lora_A' in name or 'lora_B' in name]
            trainable_params.extend([param for name, param in self.model.named_parameters() if 'classifier' in name])
        elif self.ft_type == 'partial_ft':
            trainable_params = self.model.last_layer.parameters()

        if identity == 'attacker':
            if self.attacker_optim == "Adam":
                return torch.optim.Adam(trainable_params, lr=self.attacker_lr, weight_decay=1e-5)
            elif self.attacker_optim == "SGD":
                return torch.optim.SGD(trainable_params, lr=self.attacker_lr)
        elif identity == 'user':
            if self.user_optim == "Adam":
                return torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=1e-5)
            elif self.user_optim == "SGD":
                return torch.optim.SGD(trainable_params, lr=self.lr)

    def launch_attack(self):
        """
                generating bit flip chain (format: (time, bitflip_info))
                and trigger (format: (time, bitflip_info))
        """

        if self.inherit_slurm == 'no':
            self.current_round_bf = self.search_bit() #  [] #
            for ele in self.current_round_bf:
                print(ele)
            self.bitflip_info = self.current_round_bf
            self.saved_results.update(self.report('attacker'))
        else:
            dic_path = 'saved_file'
            for file_name in os.listdir(dic_path):
                if self.inherit_slurm in file_name and self.model[0].model_name in file_name:
                    print(file_name)
                    cur_path = os.path.join(dic_path, file_name)
                    final_results = np.load(cur_path, allow_pickle=True).item()
                    # final_results = torch.load(cur_path, map_location=self.device)
                    self.bitflip_info = final_results['bitflip_info']
                    self.ImageRecoder.current_trigger = final_results['cur_trigger'].to(self.device)
                    self.current_round_bf = final_results['current_round_bf']
                    self.selected_neurons = final_results['selected_neurons']
                    for bitflip in self.bitflip_info:
                        print(bitflip)
                    if self.defense_number != 0: # bit flip defense
                        file_path = f'defense_wbp/{self.model.model_name}_vul_params.pth.npy'
                        ban_bits = np.load(file_path, allow_pickle=True).tolist()[:self.defense_number]
                        new_bits = []
                        for e in ban_bits:
                            new_bits.append(
                                {'layer': e['layer'],
                                'offset': int(e['offset'].cpu()),}
                            )
                        new_bitflip_info = []
                        for e in self.bitflip_info:
                            e2 = {'layer': e['layer'],
                                'offset': e['offset'],}
                            if e2 not in new_bits:
                                new_bitflip_info.append(e)
                        print(f'miss rate for defense: {(len(self.bitflip_info)-len(new_bitflip_info))/len(self.bitflip_info)}')
                        self.bitflip_info = new_bitflip_info
                        self.current_round_bf = new_bitflip_info

                    break

            if self.inherit_continue == 'yes':
                print(f"continue to search bits after inheriting: {self.inherit_slurm}, inherit bits: {len(self.bitflip_info)}")
                change_model_weights(self.model, self.bitflip_info)
                continued_bits = self.search_bit()
                self.bitflip_info.extend(continued_bits)
                self.current_round_bf = self.bitflip_info
                for ele in self.current_round_bf:
                    print(ele)
                self.saved_results.update(self.report('attacker'))
            else: print(f"inherit but do not continue search")

            self.parent_loader.origin_fm = self.get_clean_fm(self.model)


            ##################
            # self.model = map_model(**{
            #     'model_name': 'vgg16_2',
            #     'num_class': 1000,
            #     'pretrained': 1,
            #     'replace': False,
            #     'device': self.device,
            #     'seed': 0,
            #     'multi_features': True if self.control_mode == 'multi_mmd' else False,
            #     'ft_type': self.ft_type,
            # })
            #
            # self.test(self.model, self.parent_loader.test_loader, 0)
            # change_model_weights(self.model, self.bitflip_info)
            # self.test(self.model, self.parent_loader.test_loader, 0)
            # self.tmp_asr = self.test(self.model, self.parent_loader.test_loader, 0, use_trigger=True)
            # exit()

        # bit reduction
        if len(self.bitflip_info) != 0:
            if self.bit_reduction == 'genetic':
                raise NotImplementedError
                print("launch genetic bit reduction")
                from bit_reduction import GeneticAlgorithm
                from bit_reduction import fitness
                # (self, model, data_loader, device, control_mode, ImageRecoder, trigger, mmd_loss, cosine_loss, parent_loader, fixed_fm)
                fit_score = fitness(self.model, self.parent_loader.bit_search_data_loader, self.device, self.control_mode,
                                    self.ImageRecoder, self.ImageRecoder.current_trigger, self.mmd_loss, self.cosine_loss, self.mse_loss,
                                    self.parent_loader, self.fixed_fm, self.bitflip_info)
                ga = GeneticAlgorithm(self.bitflip_info)
                ga.fitness = fit_score
                solution, fitness = ga.run()
                self.bitflip_info = [x for x in self.bitflip_info if x in solution]
                print(f'the final solution has {len(solution)} bitflips and fitness is {fitness}')
            elif self.bit_reduction == 'greedy': # progressive bit reduction
                print("launch greedy bit reduction")
                from bit_reduction import fitness
                fit_score = fitness(self.model, self.parent_loader.bit_search_data_loader, self.device, self.control_mode,
                                    self.ImageRecoder, self.ImageRecoder.current_trigger, self.mmd_loss, self.cosine_loss, self.mse_loss,
                                    self.parent_loader, self.fixed_fm, self.bitflip_info, self.greedy_scale)

                current_max_score = 0.0
                iter_time = len(self.bitflip_info)
                for e in range(iter_time):
                    score_list = []
                    for i, bitflip in enumerate(self.bitflip_info):
                        tmp_bitflip_info = copy.deepcopy(self.bitflip_info)
                        tmp_bitflip_info.pop(i)
                        score = fit_score.run(tmp_bitflip_info)
                        score_list.append(score)
                    max_score = max(score_list)
                    max_index = score_list.index(max_score)
                    if len(self.bitflip_info) == 1: break
                    if max_score >= current_max_score:
                        print(f"current max score: {max_score}, pop out bit: {self.bitflip_info[max_index]}")
                        current_max_score = max_score
                        self.bitflip_info.pop(max_index)
                    else: break
                self.current_round_bf = self.bitflip_info
                print(f"final bits: {len(self.bitflip_info)}")
            else: print("don't launch bit reduction")

        self.saved_results['cur_trigger'] = self.ImageRecoder.current_trigger
        self.saved_results['current_round_bf'] = self.current_round_bf
        self.saved_results['bitflip_info'] = self.bitflip_info
        self.saved_results['selected_neurons'] = self.selected_neurons
        np.save(self.saved_path, self.saved_results)
        model_name = self.model[0].model_name

        print('################################# User Fine-Tuning Stage #################################')
        if self.ft_type == 'lora': #reset bit flip info
            cur_bf = []
            for e in self.current_round_bf:
                e['layer'] = e['layer'].replace('model.', 'model.base_model.model.')
                # e['layer'][:6] + 'base_model.model.' + e['layer'][6:]
                keys = ['query', 'value']
                for tail in keys:
                    if tail in e['layer']:
                        e['layer'] = e['layer'].replace(tail, tail + '.base_layer')
                cur_bf.append(e)
            self.current_round_bf = cur_bf



        from dataloader import SingleLoader
        user_order = 0
        if self.lr == 0.0 and not self.nlp_flag:
            learning_rates = [0.0005, 0.001, 0.005, 0.00001, 0.00002, 0.00005]
            optims = ['SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam']
        elif self.lr == 0.0 and self.nlp_flag:
            learning_rates = [0.0005, 0.001, 0.005, 0.00001, 0.00002, 0.00005]
            optims = ['SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam']
        elif self.lr == 1.0:
            learning_rates = [0.00002, 0.00005]
            optims = ['Adam', 'Adam',]
        elif self.lr == 2.0:
            learning_rates = [0.0]
            optims = ['Adam']
        else:
            learning_rates = [self.lr]
            optims = [self.user_optim]

        self.user_seeds = [self.user_seed + i for i in range(self.num_user * len(self.tasks) * len(learning_rates))]
        self.users_data = [{} for i in range(self.num_user * len(self.tasks) * len(learning_rates))]
        assert len(self.user_seeds) == len(self.users_data)

        for task in self.tasks:
            loader_kwargs = {
                'task': task,
                'train_batch_size': self.parent_loader.train_batch_size,
                'test_batch_size': self.parent_loader.test_batch_size,
                'image_size': self.parent_loader.image_size,
                'device': self.parent_loader.device,
                'image_number': self.parent_loader.image_number,  # image number for bit flip stage
                'mean': self.parent_loader.mean,
                'std': self.parent_loader.std,
            }
            self.child_loader = SingleLoader(**loader_kwargs)
            self.child_loader.init_loader()
            for learning_rate, optim in zip(learning_rates, optims):
                self.attacker_lr = learning_rate
                self.lr = learning_rate
                self.user_optim = optim
                for _ in range(self.num_user):
                    # do not flip any bits at the first time runs
                    flip_flag = True
                    if self.num_user == 1:
                        print(f"do not flip bits for testing normal accuracy")
                        flip_flag = False
                    print(f"#################### Task {task} | Order {_ + 1} | lr {self.lr:.5f} | Optim {self.user_optim} ####################")
                    user_model_kwargs = {
                        'model_name': model_name,
                        'num_class': self.child_loader.num_class,
                        'pretrained': _,
                        'replace': True,
                        'device': self.device,
                        'seed': self.user_seeds[user_order],
                        'multi_features': True if self.control_mode == 'multi_mmd' else False,
                        'ft_type': self.ft_type,
                    }
                    self.model = map_model(**user_model_kwargs)
                    # print(self.model.state_dict().keys())
                    # exit()

                    # # being deleted:
                    # self.model.load_state_dict(torch.load('vgg_gtsrb.pth'))
                    # self.test(self.model, self.child_loader.test_loader, 0)
                    # def change_fm_test(model, test_loader, value, neuron_index):
                    #     model.eval()
                    #     count = 0
                    #     all_corrects = [0.0 for i in range(self.child_loader.num_class)]
                    #     running_corrects = 0.0
                    #     with torch.no_grad():
                    #         all_preds = []
                    #         all_labels = []
                    #         for inputs, labels in test_loader:
                    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
                    #             _, fm_clean = model(inputs, latent=True)
                    #             fm_clean[:, neuron_index] = value
                    #             outputs = model.last_layer(fm_clean)
                    #             _, preds = torch.max(outputs, 1)
                    #             all_preds.extend(preds.tolist())
                    #             all_labels.extend(labels.data.tolist())
                    #             running_corrects += torch.sum(preds == labels.data)
                    #             for j in range(self.child_loader.num_class):
                    #                 tmp_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * j).to(
                    #                     self.device)
                    #                 all_corrects[j] += torch.sum(preds == tmp_labels)
                    #             count += inputs.size(0)
                    #
                    #         target = all_corrects.index(max(all_corrects))
                    #         untar_counts = 0
                    #         target_counts = 0
                    #         special_counts = 0
                    #         p = 0
                    #         for i, j in zip(all_preds, all_labels):
                    #             if i != j:
                    #                 untar_counts += 1
                    #             if i == target:
                    #                 target_counts += 1
                    #             if j != target:
                    #                 p += 1
                    #                 if i != j: special_counts += 1
                    #         untar_acc = untar_counts / len(all_preds)
                    #         tar_acc = target_counts / len(all_preds)
                    #         aduntar_acc = special_counts / p
                    #         self.acc_history.append(untar_acc)
                    #         print("Epoch {:<5} ACC_UnTar: {:.2f}%".format(0, untar_acc * 100))
                    #         print("Epoch {:<5} ACC_AdUnTar: {:.2f}%, target: {}".format(0, aduntar_acc * 100,
                    #                                                                     target))
                    #         print("Epoch {:<5} ACC_Tar: {:.2f}%, target: {}".format(0, tar_acc * 100, target))
                    #
                    # for value in [100, 1000, 2000]:
                    #     for i in range(5):
                    #         neuron_index = self.selected_neurons[i*20: (i+1)*20]
                    #         print(f"=> current neuron idx {i}; value: {value}")
                    #         change_fm_test(self.model, self.child_loader.test_loader, value, neuron_index)
                    # exit(0)

                    if self.bf_insert_time == 'begin' and flip_flag: change_model_weights(self.model, self.current_round_bf)
                    self.train(self.model, self.child_loader.train_loader, self.child_loader.test_loader, 'user', _, flip_flag)

                    self.users_data[user_order]['task'] = task
                    self.users_data[user_order]['lr'] = self.lr
                    self.users_data[user_order]['optim'] = self.user_optim
                    self.users_data[user_order]['bit_1'] = copy.deepcopy(self.bit_1)
                    self.users_data[user_order]['bit_2'] = copy.deepcopy(self.bit_2)
                    self.users_data[user_order]['bit_3'] = copy.deepcopy(self.bit_3)
                    self.users_data[user_order]['acc'] = copy.deepcopy(self.tmp_acc)
                    self.users_data[user_order]['asr'] = copy.deepcopy(self.tmp_asr)
                    self.users_data[user_order]['user_acc_trend'] = copy.deepcopy(self.user_acc_trend)
                    self.users_data[user_order]['victim_asr_trend'] = copy.deepcopy(self.victim_asr_trend)
                    self.users_data[user_order]['victim_acc_trend'] = copy.deepcopy(self.victim_acc_trend)
                    self.users_data[user_order]['victim_target_asr_trend'] = copy.deepcopy(self.victim_target_asr_trend)
                    self.users_data[user_order]['victim_advance_untarget_asr_trend'] = copy.deepcopy(self.victim_advance_untarget_asr_trend)
                    if user_order != len(self.user_seeds) - 1:
                        self.bit_3, self.bit_2, self.bit_1 = [], [], []
                        self.tmp_asr, self.tmp_acc = 0.0, 0.0
                        self.user_acc_trend = []
                        self.victim_asr_trend = []
                        self.victim_acc_trend = []
                        self.victim_target_asr_trend = []
                        self.victim_advance_untarget_asr_trend = []

                    user_order += 1
                    # update the file in the disk at each time.
                    self.saved_results['users_data'] = self.users_data
                    np.save(self.saved_path, self.saved_results)

                    rebuttal_dir = 'rebuttal'
                    os.makedirs(rebuttal_dir, exist_ok=True)
                    state_dict = self.model.state_dict()
                    file_name = f'{self.model.model_name}_{task}_{self.inherit_slurm}_{user_order+10}.pth'
                    model_path = os.path.join(rebuttal_dir, file_name)
                    torch.save(state_dict, model_path)

        self.saved_results['cur_trigger'] = self.ImageRecoder.current_trigger
        self.saved_results['current_round_bf'] = self.current_round_bf
        self.saved_results['selected_neurons'] = self.selected_neurons
        self.saved_results['bitflip_info'] = self.bitflip_info
        self.saved_results.update(self.report('user'))
        self.saved_results['users_data'] = self.users_data

    def search_bit(self):
        self.current_loss = float('inf')

        if self.selected_neurons == None: self.selected_neurons = self.dormant_neurons_selection()
        self.ImageRecoder.current_trigger = self.trigger_generation_both(self.model, self.parent_loader.trigger_loader)

        bit_count = 0

        self.parent_loader.origin_fm = self.get_clean_fm(self.model)
        flag = True
        delta = 0.0005

        while (bit_count < self.max_bits) and flag:
            print('~' * 100)
            if self.verify_mode: self.max_bits = 2
            bit_count += 1

            psense_list = self.select_attack_param_touch_both(self.model, self.parent_loader.bit_search_data_loader)

            bitflip = self.find_optim_bit_offset_touch_both(self.model, psense_list, self.parent_loader.bit_search_data_loader,
                                                            self.ImageRecoder.current_trigger, self.parent_loader.test_loader)

            if isinstance(bitflip, list):
                self.current_round_bf.extend(bitflip)
            else:
                self.current_round_bf.append(bitflip)
            print(f"current mmd loss: {self.current_loss}")

            if self.iter_trigger == 'yes':
                print(f'iterative searching trigger')
                self.ImageRecoder.current_trigger = self.trigger_generation_both(self.model, self.parent_loader.trigger_loader, fast=True)

            self.record_loss.append(self.current_loss)
            if len(self.record_loss) < 4:
                flag = True
            else:
                p1 = (self.record_loss[-4] - self.record_loss[-3]) >= delta
                p2 = (self.record_loss[-3] - self.record_loss[-2]) >= delta
                p3 = (self.record_loss[-2] - self.record_loss[-1]) >= delta
                if p1 or p2 or p3:
                    flag = True
                else: flag = False
            if 'dormant' in self.control_mode or 'fixed_fm' in self.control_mode: flag = True

            self.saved_results['cur_trigger'] = self.ImageRecoder.current_trigger
            self.saved_results['current_round_bf'] = self.current_round_bf
            self.saved_results['bitflip_info'] = self.bitflip_info
            self.saved_results['selected_neurons'] = self.selected_neurons
            np.save(self.saved_path, self.saved_results)

        print(f'stop searching bits when flipping {bit_count} bits and current mmd loss is {self.current_loss}')

        return self.current_round_bf

    def trigger_generation_both(self, models, data_loader, fast=False):
        if self.nlp_flag:
            return self.ImageRecoder.current_trigger

        start_time = time.time()
        self.neuron_gama = (1.0 / self.neuron_value)
        if self.ImageRecoder.unique_pattern:
            print(f"trigger generation: unique pattern")
            if self.control_mode == 'fixed_fm' and self.fixed_fm is None:
                for i, data in enumerate(data_loader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, self.ImageRecoder.clamp(
                        self.ImageRecoder.current_trigger.to(self.device)))
                    _, fm = models[0](poison_batch_inputs, latent=True)
                    self.fixed_fm = self.create_fixed_feature_map(fm.view(data_loader.batch_size, -1))
                    break
            self.ImageRecoder.current_trigger = self.ImageRecoder.init_trigger()
            return self.ImageRecoder.current_trigger.to(self.device)

        for model in models: model.eval()
        if torch.is_tensor(self.ImageRecoder.current_trigger):
            current_trigger = torch.clone(self.ImageRecoder.current_trigger.detach())
            optimizer = torch.optim.Adam([{'params': current_trigger}], lr=self.ImageRecoder.trigger_lr, betas=(0.5, 0.9))
            current_trigger.requires_grad = True
        else:
            current_trigger = [torch.clone(trigger.detach()) for trigger in self.ImageRecoder.current_trigger]
            optimizer = torch.optim.Adam([{'params': current_trigger}], lr=self.ImageRecoder.trigger_lr, betas=(0.5, 0.9))
            for trigger in current_trigger:
                trigger.requires_grad = True

        loss_list = []
        epoch = 0
        flag = True

        layercount = 0
        for name, param in models[0].named_parameters():
            layercount += 1
        if fast:
            if self.control_mode == 'dormant':
                epoch = 380
            else:
                epoch = 380
        if self.verify_mode: epoch = 298

        print("*" * 100)
        while epoch <= 400:

            running_loss = 0.0
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, self.ImageRecoder.clamp(current_trigger))

                if 'mmd' in self.control_mode:
                    if 'multi' in self.control_mode:
                        _, fm_list = model(poison_batch_inputs, latent=True, multi_latent=True)
                        _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                        loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in zip(fm_clean_list, fm_list)]
                        loss = sum(loss_list) / len(loss_list)
                    elif 'single' in self.control_mode:
                        loss = 0.0
                        for model in models:
                            _, fm_clean = model(inputs, latent=True)
                            if torch.is_tensor(poison_batch_inputs):
                                _, fm = model(poison_batch_inputs, latent=True)
                                if (epoch % 200 == 0) and i == 0: loss += 1.0 - self.mmd_loss(fm_clean, fm, verbose=True)
                                else: loss += 1.0 - self.mmd_loss(fm_clean, fm)
                            else:
                                fm_list = [model(inputs, latent=True)[1] for inputs in poison_batch_inputs]
                                fm_list.append(fm_clean)
                                if (epoch % 200 == 0) and i == 0: loss += 1.0 - self.mmd_loss(fm_list, verbose=True)
                                else: loss += 1.0 - self.mmd_loss(fm_list)
                        loss = loss / len(models)

                elif self.control_mode == 'fixed_fm':
                    _, fm = model(poison_batch_inputs, latent=True)
                    if self.fixed_fm is None:
                        self.fixed_fm = self.create_fixed_feature_map(fm.view(data_loader.batch_size, -1))
                    loss = torch.nn.MSELoss()(fm, self.fixed_fm)
                    # if (epoch % 200 == 0) and i == 0: print(f'fixed fm loss: {loss}')
                elif self.control_mode == 'cosine':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss = 1.0 - self.cosine_loss(fm_clean, fm)
                elif self.control_mode == 'mse':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss = 1.0 - self.mse_loss(fm_clean, fm)
                elif self.control_mode == 'dormant':
                    _, fm = model(poison_batch_inputs, latent=True)
                    loss = self.neuron_loss(fm, self.selected_neurons, neuron_value=10.0, device=self.device, target_neuron=True)
                else:
                    loss = 0.0
                    raise NotImplementedError



                loss.backward(retain_graph=True)

                running_loss += loss.item()
                # print("iter: {}; neuron loss: {:.2f}".format(i + 1, loss.item()))
            optimizer.step()

            if epoch % 20 == 0:
                print(f"epoch: {epoch}; {self.control_mode} loss: {running_loss / len(data_loader):.2f}")
            loss_list.append(running_loss / len(data_loader))
            epoch += 1

        print("*" * 100)
        image_trigger = self.ImageRecoder.clamp(current_trigger)
        self.current_loss = loss_list[-1]
        print(f"trigger generation time {time.time() - start_time}")
        return image_trigger

    def specical_neuron_selection(self):
        dataset = self.loader.neuron_select_dataset
        mean, std = self.loader.get_mean_std(dataset, self.model)
        neuron_score = mean * std  # or std^2?
        value, key = torch.topk(neuron_score.view(-1, ), self.neuron_number)
        # print(mean[key])
        # print(std[key])
        value_t, key_t = torch.topk(mean.view(-1, ), self.neuron_number)
        # exclude_key = [i for i in key.tolist() if i not in key_t.tolist()]
        # print(mean[exclude_key])
        # print(std[exclude_key])
        self.neuron_value = value_t[0]

        candidates = key.data.cpu().numpy().copy().reshape(-1, )
        print(f"set neuron value: {self.neuron_value}\n"
              f"candidates: {candidates}")

        return candidates

    def select_attack_param_touch_both(self, models, data_loader):
        start_time = time.time()
        grad_dicts = []
        for model in models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            zero_gradients(model)
            grad_dict = {}
            for i, (name, param) in enumerate(model.named_parameters()):
                grad_dict[name] = 0.0
            grad_dicts.append(copy.deepcopy(grad_dict))

        max_iter = int(self.image_number / data_loader.batch_size)
        running_loss = 0.0
        for cur_order, model in enumerate(models):
            for i, data in enumerate(data_loader):
                if i >= max_iter: break
                # zero_gradients(model)
                cuda_state(4)
                if not self.nlp_flag:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    inputs, labels = data[0], data[1]
                poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(
                    inputs, self.ImageRecoder.clamp(self.ImageRecoder.current_trigger))

                if 'mmd' in self.control_mode:
                    if 'multi' in self.control_mode:
                        _, fm_list = model(poison_batch_inputs, latent=True, multi_latent=True)
                        _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                        loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in zip(fm_clean_list, fm_list)]
                        loss_mmd = sum(loss_list) / len(loss_list)
                        loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
                        loss = loss_mmd + 1.0 * loss_3
                    elif 'single' in self.control_mode:
                        if not self.nlp_flag:
                            if torch.is_tensor(self.ImageRecoder.current_trigger):
                                _, fm = model(poison_batch_inputs, latent=True)
                                output_clean, fm_clean = model(inputs, latent=True)
                                if self.clean_type == '0':
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i], True)
                                    loss = loss_mmd + self.clean_trade_off * loss_3
                                elif self.clean_type == '1':
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = torch.tensor(0.0)
                                    loss = loss_mmd
                                elif self.clean_type == '2':
                                    loss_mmd = 1.0 - self.mmd_loss(self.parent_loader.origin_fm[cur_order][i], fm)
                                    loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i], True)
                                    loss = loss_mmd + self.clean_trade_off * loss_3
                                elif self.clean_type == '3':
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i])
                                    loss = loss_mmd + self.clean_trade_off * loss_3
                                elif self.clean_type == '4':
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = torch.nn.CrossEntropyLoss()(output_clean, labels)
                                    loss = loss_mmd + self.clean_trade_off * loss_3

                            else:
                                fm_clean = model(inputs, latent=True)[1]
                                fm_list = [model(inputs, latent=True)[1] for inputs in poison_batch_inputs]
                                fm_list.append(fm_clean)
                                loss_mmd = 1.0 - self.mmd_loss(fm_list)
                                loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[cur_order][i]], True)
                                loss = loss_mmd + 1.0 * loss_3
                        else:
                            output= model(poison_batch_inputs, latent=True)
                            output_clean = model(inputs, latent=True)
                            fm = output[0].hidden_states[-1][:, 0, :]
                            fm_clean = output_clean[0].hidden_states[-1][:, 0, :]
                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                            loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
                            loss = loss_mmd + 1.0 * loss_3

                elif self.control_mode == 'fixed_fm':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
                    loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3
                elif self.control_mode == 'cosine':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
                    loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3
                elif self.control_mode == 'mse':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss_mmd = 1.0 - self.mse_loss(fm_clean, fm)
                    loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3
                elif self.control_mode == 'dormant':
                    _, fm = model(poison_batch_inputs, latent=True)
                    _, fm_clean = model(inputs, latent=True)
                    loss_mmd = self.neuron_loss(fm, self.selected_neurons, 10.0, self.device, True)
                    loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 0.5 * loss_3

                running_loss += loss.item()

                loss.backward(retain_graph=True)
                for j, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None:
                        grad_dicts[cur_order][name] += torch.clone(param.grad.detach())

                cuda_state(7)

        print("select params loss: {:.2f}".format(running_loss / max_iter / len(models)))

        torch.cuda.empty_cache()
        # sss = net.state_dict()

        most_vulnerable_param = {
            'layer': '',
            'offset': 0,
            'weight': 0.0,
            'grad': 0.0,
            'score': 0.0,
        }

        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16_2': ['classifier.weight', 'classifier.bias', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'],
            'vgg16_bn': ['classifier.weight', 'classifier.bias'],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.weight', 'classifier.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight', 'classifier.1.bias'],
            'efficient': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight', 'classifier.1.bias'],
            'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
            'densenet121': [],
            'alexnet': [],
            'vit': ['heads.head.weight', 'heads.head.bias', 'classifier.weight', 'classifier.bias'],
            'deit': ['head.weight', 'head.bias'],
            'vgg16_seg': [],
            'bert': [],
        }

        ban_name = ban_name_dict[self.model[0].model_name]
        cuda_state(8)
        scale_9 = (1.0 / 16.0) * (15.0 / 16.0)
        conflict_number = 0
        topk_number = 500
        # if model.model_name == 'vggface': topk_number = 100
        def get_model_params(model, name):
            for param_name, param in model.named_parameters():
                if param_name == name:
                    return param
            return None

        def get_fitscore(models, grad_dicts):
            vul_params = []
            topk_number = 500
            for i, name in enumerate(grad_dicts[0].keys()):
                if grad_dicts[0][name] is not None and (name[6:] not in ban_name) and ('bias' not in name):
                    # print(name)
                    fitscores = [grad_dict[name] for grad_dict in grad_dicts]
                    fitscores = [torch.mul(fitscore.cpu(), get_model_params(model, name).detach().cpu()) for
                                 fitscore, model in zip(fitscores, models)] # .cpu()

                    stacked_tensors = torch.stack(fitscores, dim=0)
                    fitscore = stacked_tensors.mean(dim=0)
                    fitscore_std = stacked_tensors.std(dim=0)
                    # Create a mask to identify positions where values are both positive and negative
                    mask_positive = (stacked_tensors > 0).all(dim=0)
                    mask_negative = (stacked_tensors < 0).all(dim=0)
                    mask_zero = ~(mask_positive | mask_negative)
                    # fitscore[mask_zero] = 0.0

                    fitscore = abs(fitscore)

                    fitscore[mask_zero] = 0.0
                    fitscore = self.mask_fitscore(fitscore, self.rowhammer_page[name])

                    (values, indices) = torch.topk(fitscore.detach().view(-1, ),
                                                   min(topk_number, fitscore.view(-1, ).size()[0]))  # self.num_vul_params
                    cuda_state(10)
                    count = 0
                    for indice, value in zip(indices, values):
                        # user model will not be involved in the computation.
                        weights = [get_model_params(model, name).detach().view(-1, )[indice] for model in models]
                        binarys = [floatToBinary64(weight) for weight in weights]
                        bit_9 = [binary[9] for binary in binarys]
                        # if bit_9 == '1' and value * scale_9 < mid_value: continue
                        if '1' in bit_9: continue
                        most_vulnerable_param['layer'] = name
                        most_vulnerable_param['offset'] = indice
                        most_vulnerable_param['weight'] = [
                            get_model_params(model, name).data.view(-1)[indice].detach().item() for model in models]
                        most_vulnerable_param['grad'] = [grad_dict[name].view(-1)[indice].detach().item() for grad_dict
                                                         in grad_dicts]
                        most_vulnerable_param['score'] = value.detach().item()
                        vul_params.append(copy.deepcopy(most_vulnerable_param))
                        count += 1
                    # if count <= 100: print(
                    #     f'warning: for layer {name}, only find {count} weights are suitable for bit flip')
            cuda_state(11)
            return vul_params

        def get_fitscore_abandon(models, grad_dicts):
            vul_params = []
            for i, (name, param) in enumerate(model.named_parameters()):
                if grad_dict[name] is not None and (name[6:] not in ban_name) and ('bias' not in name):
                    fitscore = grad_dict[name]
                    if not torch.is_tensor(fitscore):
                        # print(name, grad_dict[name])
                        continue
                    fitscore = torch.mul(fitscore, param.detach())
                    fitscore = abs(fitscore)
                    fitscore = self.mask_fitscore(fitscore, self.rowhammer_page[name])
                    cuda_state(9)

                    (values, indices) = torch.topk(fitscore.detach().view(-1, ),
                                                   min(topk_number, fitscore.view(-1, ).size()[0]))  # self.num_vul_params
                    if len(values) >= 200:
                        mid_value = values[99]
                    else:
                        mid_value = values[int(len(values)/2)]

                    cuda_state(10)
                    # value = fitscore.view(-1, )[indices]
                    count = 0
                    # min_value = values[-1]
                    for indice, value in zip(indices, values):
                        weight = param.view(-1, )[indice]
                        cur_grad = grad_dict[name].view(-1, )[indice]
                        binary = floatToBinary64(weight)
                        bit_9 = binary[9]
                        if bit_9 == '1' and value * scale_9 < mid_value: continue
                        most_vulnerable_param['layer'] = name
                        most_vulnerable_param['offset'] = indice
                        most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                        most_vulnerable_param['grad'] = param.grad.view(-1)[indice].detach().item()
                        most_vulnerable_param['score'] = value.detach().item()
                        vul_params.append(copy.deepcopy(most_vulnerable_param))
                        count += 1
                cuda_state(11)


        vul_params = get_fitscore(models, grad_dicts)
        vul_params = rank(vul_params, 'score')
        zero_gradients(models)
        print(f"vul params searching time: {time.time() - start_time}")
        return vul_params

    def find_optim_bit_offset_touch_both(self, models, param_sens_list, data_loader, trigger,
                                         test_dataloader, fast_mode=False):
        start_time = time.time()
        for model in models:
            model.eval()
        ##################################Load Dataset################################

        def convert_params_to_loss(params_list):
            max_iter = int(self.image_number / data_loader.batch_size)  # int(len(data_loader)*0.1)#
            print(f"data_loader size: {len(data_loader)}; max_iter: {max_iter}")
            inherent_ban_layer = {}
            current_num_vul_parameters = 0
            final_list = []
            try_flip_number = 0
            dormant_try = 0
            # get original loss##############
            def repeat_inference(cur_order, verbose=False):
                total_loss = 0.0
                clean_loss_total = 0.0
                mmd_loss_total = 0.0
                with torch.no_grad():
                    for i, data in enumerate(data_loader):
                        if i >= max_iter: break
                        if not self.nlp_flag:
                            inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        else:
                            inputs, labels = data[0], data[1]
                        poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, trigger)

                        if 'mmd' in self.control_mode:
                            if 'multi' in self.control_mode:
                                _, fm_list = model(poison_batch_inputs, latent=True, multi_latent=True)
                                _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                                loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in
                                             zip(fm_clean_list, fm_list)]
                                loss_mmd = sum(loss_list) / len(loss_list)
                                loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
                                loss = loss_mmd + 1.0 * loss_3
                                if verbose:
                                    p = torch.mean(
                                        torch.mean(fm_list[-1].view(fm_list[-1].size(0), -1)[:, self.selected_neurons], 0).view(
                                            -1, ))
                                    p1 = torch.mean(
                                        torch.mean(
                                            fm_clean_list[-1].view(fm_clean_list[-1].size(0), -1)[:, self.selected_neurons],
                                            0).view(-1, 1))
                            elif 'single' in self.control_mode:
                                output_clean, fm_clean = models[cur_order](inputs, latent=True)
                                if not self.nlp_flag:
                                    if torch.is_tensor(self.ImageRecoder.current_trigger):
                                        _, fm = models[cur_order](poison_batch_inputs, latent=True)
                                        if self.clean_type == '0':
                                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                            loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i],
                                                                   True)
                                            loss = loss_mmd + self.clean_trade_off * loss_3
                                        elif self.clean_type == '1':
                                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                            loss_3 = torch.tensor(0.0)
                                            loss = loss_mmd
                                        elif self.clean_type == '2':
                                            loss_mmd = 1.0 - self.mmd_loss(self.parent_loader.origin_fm[cur_order][i],
                                                                           fm)
                                            loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i],
                                                                   True)
                                            loss = loss_mmd + self.clean_trade_off * loss_3
                                        elif self.clean_type == '3':
                                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                            loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i])
                                            loss = loss_mmd + self.clean_trade_off * loss_3
                                        elif self.clean_type == '4':
                                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                            loss_3 = torch.nn.CrossEntropyLoss()(output_clean, labels)
                                            loss = loss_mmd + self.clean_trade_off * loss_3


                                        if verbose:
                                            p = torch.mean(
                                                torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                            p1 = torch.mean(
                                                torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(
                                                    -1, 1))
                                    else:
                                        fm_list = [models[cur_order](inputs, latent=True)[1] for inputs in poison_batch_inputs]
                                        fm_list.append(fm_clean)
                                        loss_mmd = 1.0 - self.mmd_loss(fm_list)
                                        loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[cur_order][i]], True)
                                        loss = loss_mmd + 1.0 * loss_3
                                        if verbose: p, p1 = 0.0, 0.0
                                else:
                                    output = model(poison_batch_inputs, latent=True)
                                    output_clean = model(inputs, latent=True)
                                    fm = output[0].hidden_states[-1][:, 0, :]
                                    fm_clean = output_clean[0].hidden_states[-1][:, 0, :]
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
                                    loss = loss_mmd + 1.0 * loss_3
                                    if verbose: p, p1 = 0.0, 0.0


                        elif self.control_mode == 'fixed_fm':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
                            loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))

                        elif self.control_mode == 'cosine':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
                            loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        elif self.control_mode == 'mse':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = 1.0 - self.mse_loss(fm_clean, fm)
                            loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        elif self.control_mode == 'dormant':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = self.neuron_loss(fm, self.selected_neurons, 10.0, self.device, True)
                            loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 0.5 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        clean_loss_total += loss_3.item()
                        mmd_loss_total += loss_mmd.item()

                        total_loss += loss.detach().item()

                    current_loss = total_loss / max_iter
                    cur_clean_loss = clean_loss_total / max_iter
                    cur_mmd_loss = mmd_loss_total / max_iter
                if verbose:
                    print(f"poison neuron value: {p}")
                    print(f"clean neuron value:  {p1}")
                if verbose:
                    return current_loss, cur_mmd_loss, cur_clean_loss
                else:
                    return current_loss
            for order in range(len(models)):
                origin_loss, origin_mmd_loss, origin_clean_loss = repeat_inference(order, True)
                print(f"model {order}: origin loss: {origin_loss:.3f}, {self.control_mode} loss: {origin_mmd_loss:.3f}, clean loss: {origin_clean_loss:.3f}")

            #################################
            for num, param_sens in enumerate(params_list):
                if dormant_try >= 100:
                    print('dormant neuron has tried 100 times, break')
                    break
                # Add rowhammer limitation for each round bit flips.
                layer_name = param_sens['layer']

                if layer_name in inherent_ban_layer.keys() and inherent_ban_layer[
                    layer_name] >= self.num_vul_params: continue
                if not self.layer_wise_flipping and current_num_vul_parameters >= self.num_vul_params * 3: continue
                # if ban_weights_by_rowhammer_limitation(param_sens, self.current_round_bf): continue
                if self.verify_mode and (try_flip_number >= 10): break

                optional_bit = []
                current_param = param_sens

                if not (all_pos_neg(param_sens['weight']) and all_pos_neg(param_sens['grad'])): continue

                Binary = floatToBinary64(param_sens['weight'][0])
                grad_sign = 0 if current_param['grad'][0] < 0 else 1
                weight_sign = 0 if current_param['weight'][0] < 0 else 1
                ban_bit = []
                for w in param_sens['weight']:
                    ban_bit.extend(ban_unstable_bit_of_float_perturbation(w))
                ban_bit = list(set(ban_bit))
                ban_bit1 = ensemble_ban_unstable_bit(param_sens['weight'])
                select_bit_list = [9] if self.layer_wise_flipping else [9, 10, 11]
                for i in select_bit_list:#[9, 10]:  # [8, 9, 10, 11]:
                    optional_bit.append((i, int(Binary[i])))
                    current_param['bit_offset'] = i
                    current_param['bit_direction'] = int(Binary[i])
                    if i in ban_bit: continue
                    if i in ban_bit1: continue
                    if grad_sign == weight_sign and int(Binary[i]) == 0: continue
                    if grad_sign != weight_sign and int(Binary[i]) == 1: continue
                    weight_before_lst = param_sens['weight']
                    weight_after_bf_lst = [2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * weight
                                           for weight in param_sens['weight']] if i != 0 \
                        else [-1 * model.state_dict()[param_sens['layer']].view(-1, )[
                        param_sens['offset']].detach().item() for model in models]

                    # if i == 0:
                    #     current_param['weight_after_bf'] = -1 * model.state_dict()[param_sens['layer']].view(-1, )[
                    #         param_sens['offset']].detach().item()
                    # else:
                    #     current_param['weight_after_bf'] = 2 ** (
                    #             ((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * \
                    #                                        param_sens['weight']

                    # prepare replaced weights

                    # weight_before = model.state_dict()[param_sens['layer']].view(-1, )[
                    #     param_sens['offset']].detach().item()
                    current_param['weight_after_bf'] = weight_after_bf_lst
                    current_param['loss_after_bf'] = 0.0

                    for j, model in enumerate(models):
                        if self.bitflip_value_limit_mode == 'yes':
                            max_value, min_value = torch.max(model.state_dict()[param_sens['layer']].view(-1, )), torch.min(
                                model.state_dict()[param_sens['layer']].view(-1, ))
                            print("-" * 50 + 'enter bitflip value limitation mode' + '-' * 50)
                            if current_param['weight_after_bf'] > max_value * 1.1 or current_param[
                                'weight_after_bf'] < min_value * 1.1:
                                print(f"max,min limitation of value, ban bit {i}")
                                continue

                        model.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                            'weight_after_bf'][j]

                        # current_param['loss_after_bf'] = repeat_inference()
                        loss = repeat_inference(j)
                        current_param['loss_after_bf'] += loss

                        model.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before_lst[j]

                        # flip back to original value

                        if self.control_mode == 'dormant':
                            dormant_try += 1
                        elif current_param['loss_after_bf'] >= origin_loss:
                            continue
                    current_param['loss_after_bf'] /= 2.0
                    final_list.append(copy.deepcopy(current_param))

                    if layer_name not in inherent_ban_layer.keys():
                        inherent_ban_layer[layer_name] = 1
                    else:
                        inherent_ban_layer[layer_name] += 1
                    current_num_vul_parameters += 1
                    try_flip_number += 1

            print(f"try flip number:     {try_flip_number}")
            if try_flip_number == 0: raise RuntimeError("We can not find any possible bit to flip, thus exit")

            return final_list

        final_list = convert_params_to_loss(param_sens_list)
        final_list_rank = rank(final_list, 'loss_after_bf', reverse=False)

        bitflip_info_list = final_list_rank[:self.num_bits_single_round]
        bitflip_info_list_simple = []
        for select_bitflip in bitflip_info_list:
            bitflip_info = {
                'layer': select_bitflip['layer'],
                'offset': select_bitflip['offset'].item(),
                'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
            }
            bitflip_info_list_simple.append(bitflip_info)
        for model in models:
            change_model_weights(model, bitflip_info_list_simple)
        self.add_to_rowhammer_page(bitflip_info_list_simple)
        for ele in bitflip_info_list_simple:
            print(f"selected bit is located at {ele['offset']}")
        # print(f"selected bit is located at {index}th in the ranking")

        if len(final_list_rank) != 0:
            print(f"Current Min Loss: {final_list_rank[0]['loss_after_bf']:.3f}")
        else:
            print('Current Min Loss: larger than before (find optim bitflips stage)')

        zero_gradients(models)
        print(f"bit flip searching time: {time.time() - start_time}")

        return bitflip_info_list_simple

    def select_attack_param_touch_both2(self, model, data_loader):
        self.flippable_bit_location = [9, 10, 11]
        start_time = time.time()
        model.eval()
        grad_dict = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            grad_dict[name] = 0.0
        for param in model.parameters():
            param.requires_grad = True

        zero_gradients(model)

        max_iter = int(self.image_number / data_loader.batch_size)
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            if i >= max_iter: break
            # zero_gradients(model)
            cuda_state(4)
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(
                inputs, self.ImageRecoder.clamp(self.ImageRecoder.current_trigger))

            if 'mmd' in self.control_mode:
                if 'multi' in self.control_mode:
                    _, fm_list = model(poison_batch_inputs, latent=True, multi_latent=True)
                    _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                    loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in zip(fm_clean_list, fm_list)]
                    loss_mmd = sum(loss_list) / len(loss_list)
                    loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
                    loss = loss_mmd + 1.0 * loss_3
                elif 'single' in self.control_mode:
                    if torch.is_tensor(self.ImageRecoder.current_trigger):
                        _, fm = model(poison_batch_inputs, latent=True)
                        _, fm_clean = model(inputs, latent=True)
                        loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                        loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
                        loss = loss_mmd + 1.0 * loss_3
                    else:
                        fm_clean = model(inputs, latent=True)[1]
                        fm_list = [model(inputs, latent=True)[1] for inputs in poison_batch_inputs]
                        fm_list.append(fm_clean)
                        loss_mmd = 1.0 - self.mmd_loss(fm_list)
                        loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[i]], True)
                        loss = loss_mmd + 1.0 * loss_3
            elif self.control_mode == 'fixed_fm':
                _, fm = model(poison_batch_inputs, latent=True)
                _, fm_clean = model(inputs, latent=True)
                loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
                loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                loss = loss_mmd + 1.0 * loss_3
            elif self.control_mode == 'cosine':
                _, fm = model(poison_batch_inputs, latent=True)
                _, fm_clean = model(inputs, latent=True)
                loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
                loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
                loss = loss_mmd + 1.0 * loss_3
            elif self.control_mode == 'mse':
                _, fm = model(poison_batch_inputs, latent=True)
                _, fm_clean = model(inputs, latent=True)
                loss_mmd = 1.0 - self.mse_loss(fm_clean, fm)
                loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[i])
                loss = loss_mmd + 1.0 * loss_3
            elif self.control_mode == 'dormant':
                _, fm = model(poison_batch_inputs, latent=True)
                _, fm_clean = model(inputs, latent=True)
                loss_mmd = self.neuron_loss(fm, self.selected_neurons, 10.0, self.device, True)
                loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                loss = loss_mmd + 0.5 * loss_3

            running_loss += loss.item()

            loss.backward(retain_graph=True)
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    # grad_dict[name] += param.grad.detach()
                    grad_dict[name] += torch.clone(param.grad.detach())

            cuda_state(7)
        print("select params loss: {:.2f}".format(running_loss / max_iter))

        torch.cuda.empty_cache()
        # sss = net.state_dict()

        most_vulnerable_param = {
            'layer': '',
            'offset': 0,
            'weight': 0.0,
            'grad': 0.0,
            'score': 0.0,
        }
        vul_params = []
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16_2': ['classifier.weight', 'classifier.bias'],
            'vgg16_bn': ['classifier.weight', 'classifier.bias'],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.weight', 'classifier.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight', 'classifier.1.bias'],
            'efficient': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight', 'classifier.1.bias'],
            'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
            'densenet121': [],
            'alexnet': [],
            'vit': ['heads.head.weight', 'heads.head.bias', 'classifier.weight', 'classifier.bias'],
            'deit': ['head.weight', 'head.bias'],
            'vgg16_seg': [],
            'bert': [],
        }

        ban_name = ban_name_dict[self.model.model_name]
        cuda_state(8)
        topk_number = 1000
        search_time = self.num_vul_params * len(self.flippable_bit_location)
        # if model.model_name == 'vggface': topk_number = 100
        for i, (name, param) in enumerate(model.named_parameters()):
            # if grad_dict[name] is not None and (name[6:] not in ban_name) and (
            #         'bias' not in name) and ('bn' not in name) and ('downsample.1' not in name) \
            #         and ('norm' not in name) and (
            #         'ln_' not in name):
            if grad_dict[name] is not None and (name[6:] not in ban_name) and ('bias' not in name):
                fitscore = grad_dict[name]
                if not torch.is_tensor(fitscore): continue
                fitscore = torch.mul(fitscore, param.detach())
                fitscore = abs(fitscore)
                cuda_state(9)

                # ranking based on grad * weight
                (values, indices) = torch.topk(fitscore.detach().view(-1, ),
                                               min(topk_number, fitscore.view(-1, ).size()[0]))
                # ranking based on grad * weight * scale
                binary = [floatToBinary64(param.view(-1, )[indice]) for indice in indices]
                b9 = [binary[i][9] for i in range(len(binary))]
                b10 = [binary[i][10] for i in range(len(binary))]
                b11 = [binary[i][11] for i in range(len(binary))]
                s9 = [15 if b9[i] == '0' else 15.0 / 16.0 for i in range(len(b9))]
                s10 = [3 if b10[i] == '0' else 3.0 / 4.0 for i in range(len(b10))]
                s11 = [1 if b11[i] == '0' else 1.0 / 2.0 for i in range(len(b11))]
                exp_tail = [b9, b10, b11]

                scale = [max(scale9, scale10, scale11) for scale9, scale10, scale11 in zip(s9, s10, s11)]
                scale_index = [[scale9, scale10, scale11].index(scale_large) for scale9, scale10, scale11, scale_large
                               in zip(s9, s10, s11, scale)]
                bit_offset = [self.flippable_bit_location[index] for index in scale_index]
                flip_direction = [int(exp_tail[index][i]) for i, index in enumerate(scale_index)]
                assert self.flippable_bit_location == [9, 10, 11]

                # reassign score considering bit offset
                # new_values = [value * scale for value, scale in zip(values, scale)]
                weight_sign = [get_sign(param.view(-1, )[indice].item()) for indice in indices]
                grad_sign = [get_sign(grad_dict[name].view(-1, )[indice].item()) for indice in indices]
                abs_w_change_dirct = [int(ele) for ele in flip_direction]
                effect_flip = [verify_biteffect(weight_sign[i], grad_sign[i], abs_w_change_dirct[i]) for i in
                               range(len(weight_sign))]

                # remove the invalid flip (flip direction is not consistent with the sign of weight change)
                reduced_values = [j for i, j in zip(effect_flip, values) if i == 1]
                reduced_indices = [j for i, j in zip(effect_flip, indices) if i == 1]
                reduced_scale = [j for i, j in zip(effect_flip, scale) if i == 1]
                reduced_values_w_scale = [value * scale for value, scale in zip(reduced_values, reduced_scale)]
                reduced_bit_offset = [j for i, j in zip(effect_flip, bit_offset) if i == 1]
                reduced_flip_direction = [j for i, j in zip(effect_flip, flip_direction) if i == 1]

                # reranking score
                (new_values, indices_2nd) = torch.topk(torch.tensor(reduced_values_w_scale),
                                                       min(topk_number, len(reduced_values_w_scale)))
                new_indices = [reduced_indices[i] for i in indices_2nd]
                new_bit_offset = [reduced_bit_offset[i] for i in indices_2nd]
                new_flip_direction = [reduced_flip_direction[i] for i in indices_2nd]
                # ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'])

                for i in range(min(len(new_indices), search_time)):
                    indice = new_indices[i]
                    value = new_values[i]
                    most_vulnerable_param['layer'] = name
                    most_vulnerable_param['offset'] = indice
                    most_vulnerable_param['bit_offset'] = new_bit_offset[i]
                    most_vulnerable_param['bit_direction'] = new_flip_direction[i]
                    most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                    most_vulnerable_param['grad'] = grad_dict[name].view(-1)[indice].detach().item()
                    most_vulnerable_param['score'] = value.detach().item()
                    vul_params.append(copy.deepcopy(most_vulnerable_param))

        vul_params = rank(vul_params, 'score')[:search_time]

        zero_gradients(model)
        print(f"vul params searching time: {time.time() - start_time}")
        return vul_params

    def find_optim_bit_offset_touch_both2(self, model, param_sens_list, data_loader, trigger,
                                          test_dataloader, fast_mode=False):
        start_time = time.time()
        model.eval()

        ##################################Load Dataset################################

        def convert_params_to_loss(params_list):
            max_iter = int(self.image_number / data_loader.batch_size)  # int(len(data_loader)*0.1)#
            print(f"data_loader size: {len(data_loader)}; max_iter: {max_iter}")
            final_list = []
            try_flip_number = 0
            dormant_try = 0

            # get original loss##############
            def repeat_inference(verbose=False):
                total_loss = 0.0
                clean_loss_total = 0.0
                mmd_loss_total = 0.0
                with torch.no_grad():
                    for i, data in enumerate(data_loader):
                        if i >= max_iter: break
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, trigger)

                        if 'mmd' in self.control_mode:
                            if 'multi' in self.control_mode:
                                _, fm_list = model(poison_batch_inputs, latent=True, multi_latent=True)
                                _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                                loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in
                                             zip(fm_clean_list, fm_list)]
                                loss_mmd = sum(loss_list) / len(loss_list)
                                loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
                                loss = loss_mmd + 1.0 * loss_3
                                if verbose:
                                    p = torch.mean(
                                        torch.mean(fm_list[-1].view(fm_list[-1].size(0), -1)[:, self.selected_neurons],
                                                   0).view(
                                            -1, ))
                                    p1 = torch.mean(
                                        torch.mean(
                                            fm_clean_list[-1].view(fm_clean_list[-1].size(0), -1)[:,
                                            self.selected_neurons],
                                            0).view(-1, 1))
                            elif 'single' in self.control_mode:
                                _, fm_clean = model(inputs, latent=True)
                                if torch.is_tensor(self.ImageRecoder.current_trigger):
                                    _, fm = model(poison_batch_inputs, latent=True)
                                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                                    loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
                                    loss = loss_mmd + 1.0 * loss_3
                                    if verbose:
                                        p = torch.mean(
                                            torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                        p1 = torch.mean(
                                            torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(
                                                -1, 1))
                                else:
                                    fm_list = [model(inputs, latent=True)[1] for inputs in poison_batch_inputs]
                                    fm_list.append(fm_clean)
                                    loss_mmd = 1.0 - self.mmd_loss(fm_list)
                                    loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[i]], True)
                                    loss = loss_mmd + 1.0 * loss_3
                                    if verbose: p, p1 = 0.0, 0.0
                        elif self.control_mode == 'fixed_fm':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
                            loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        elif self.control_mode == 'cosine':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
                            loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        elif self.control_mode == 'mse':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = 1.0 - self.mse_loss(fm_clean, fm)
                            loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 1.0 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        elif self.control_mode == 'dormant':
                            _, fm = model(poison_batch_inputs, latent=True)
                            _, fm_clean = model(inputs, latent=True)
                            loss_mmd = self.neuron_loss(fm, self.selected_neurons, 10.0, self.device, True)
                            loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                            loss = loss_mmd + 0.5 * loss_3
                            if verbose:
                                p = torch.mean(
                                    torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
                                p1 = torch.mean(
                                    torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
                        clean_loss_total += loss_3.item()
                        mmd_loss_total += loss_mmd.item()

                        total_loss += loss.detach().item()

                    current_loss = (total_loss / max_iter)
                    cur_clean_loss = (clean_loss_total / max_iter)
                    cur_mmd_loss = (mmd_loss_total / max_iter)
                if verbose:
                    print(f"clean: {p:.2f}, poison: {p1:.2f}")
                if verbose:
                    return current_loss, cur_mmd_loss, cur_clean_loss
                else:
                    return current_loss

            origin_loss, origin_mmd_loss, origin_clean_loss = repeat_inference(True)
            print(
                f"origin loss: {origin_loss:.3f}, {self.control_mode} loss: {origin_mmd_loss:.3f}, clean loss: {origin_clean_loss:.3f}")

            #################################
            for num, vul_param in enumerate(params_list):
                if dormant_try >= 100:
                    print('dormant neuron has tried 100 times, break')
                    break
                if self.verify_mode and (try_flip_number >= 10): break

                if vul_param['bit_offset'] == 0:
                    vul_param['weight_after_bf'] = -1 * model.state_dict()[vul_param['layer']].view(-1, )[
                        vul_param['offset']].detach().item()
                else:
                    vul_param['weight_after_bf'] = 2 ** (
                            ((-1) ** (vul_param['bit_direction'])) * 2 ** (11 - vul_param['bit_offset'])) * \
                                                   vul_param['weight']

                # prepare replaced weights
                weight_before = model.state_dict()[vul_param['layer']].view(-1, )[
                    vul_param['offset']].detach().item()
                if self.bitflip_value_limit_mode == 'yes':
                    max_value, min_value = torch.max(model.state_dict()[vul_param['layer']].view(-1, )), torch.min(
                        model.state_dict()[vul_param['layer']].view(-1, ))
                    print("-" * 50 + 'enter bitflip value limitation mode' + '-' * 50)
                    if vul_param['weight_after_bf'] > max_value * 1.1 or vul_param[
                        'weight_after_bf'] < min_value * 1.1:
                        print(f"max,min limitation of value, ban bit {vul_param['bit_offset']}")
                        continue

                model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = vul_param[
                    'weight_after_bf']

                # test selected weights
                if self.integrated_loss:
                    numbers = np.linspace(vul_param['weight'], vul_param['weight_after_bf'], 11,
                                          endpoint=True)[1:]
                    loss_list = []
                    model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = vul_param[
                        'weight_after_bf']
                    loss_after = repeat_inference()
                    model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = weight_before

                    loss_list.append(loss_after)
                    if loss_after >= origin_loss: continue

                    for element in numbers[:-1]:
                        model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = element
                        loss_list.append(repeat_inference())
                        model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = weight_before
                    print(f'loss_list: {loss_list}')
                    vul_param['loss_after_bf'] = sum(loss_list) / len(loss_list)
                else:
                    model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = vul_param[
                        'weight_after_bf']
                    vul_param['loss_after_bf'] = repeat_inference()
                    model.state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = weight_before

                # flip back to original value
                # elif vul_param['loss_after_bf'] >= origin_loss:
                #     continue

                final_list.append(copy.deepcopy(vul_param))

                try_flip_number += 1

            print(f"try flip number:     {try_flip_number}")
            if try_flip_number == 0: raise RuntimeError("We can not find any possible bit to flip, thus exit")

            return final_list

        final_list = convert_params_to_loss(param_sens_list)
        final_list_rank = rank(final_list, 'loss_after_bf', reverse=False)

        bitflip_info_list = final_list_rank[:self.num_bits_single_round]
        bitflip_info_list_simple = []
        for select_bitflip in bitflip_info_list:
            bitflip_info = {
                'layer': select_bitflip['layer'],
                'offset': select_bitflip['offset'].item(),
                'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
            }
            bitflip_info_list_simple.append(bitflip_info)

        change_model_weights(model, bitflip_info_list_simple)
        for ele in bitflip_info_list_simple:
            print(f"selected bit is located at {ele['offset']}")
        # print(f"selected bit is located at {index}th in the ranking")

        if len(final_list_rank) != 0:
            print(f"Current Min Loss: {final_list_rank[0]['loss_after_bf']:.3f}")
            self.current_loss = final_list_rank[0]['loss_after_bf']
        else:
            print('Current Min Loss: larger than before (find optim bitflips stage)')

        zero_gradients(model)
        print(f"bit flip searching time: {time.time() - start_time}")

        return bitflip_info_list_simple

    def train(self, model, train_loader, test_loader, identity, user_order=0, flip_flag=True):
        if not flip_flag: print("don't flip bits at this training.")
        if model.model_name in ['vit', 'deit']:
            warmup = True
            print(f"use warm up for {model.model_name}")
        else: warmup = False
        warmup = False
        global_iteration = 0
        warmup_iterations = 1000
        target_lr = copy.deepcopy(self.lr)
        if warmup: self.lr = 0.0
        optim = self.optim_init(identity)
        self.current_iter = 0
        current_epoch = 1
        running_loss = 0.0
        model.train()
        criterion = nn.CrossEntropyLoss()


        print('before fine-tuning:')
        self.tmp_acc = self.test(model, test_loader, current_epoch)
        result =  self.test(model, test_loader, current_epoch, use_trigger=True, report_target_asr=True)
        self.tmp_asr = result[0]
        target_asr = result[1]
        ad_untarget_asr = result[2]
        if identity != 'attacker':
            self.victim_acc_trend.append((self.current_iter, self.tmp_acc))
            self.user_acc_trend.append((self.current_iter, self.tmp_acc))
            self.victim_asr_trend.append((self.current_iter, self.tmp_asr))
            self.victim_target_asr_trend.append((self.current_iter, target_asr))
            self.victim_advance_untarget_asr_trend.append((self.current_iter, ad_untarget_asr))


        num_remove_bits = (1 - self.bf_success_rate) * len(self.current_round_bf)
        num_remove_bits = math.ceil(num_remove_bits)
        skip_idx = random.sample(range(len(self.current_round_bf)), num_remove_bits)
        if len(skip_idx) != 0:
            print(f'fail to flip: ')
            for e in skip_idx:
                print(self.current_round_bf[e])

        # max_iter = self.max_iter * 2 if (self.user_optim == "SGD" and self.lr == 0.0001) else self.max_iter
        while self.current_iter < self.max_iter:
            print("*" * 100)
            print(f"Iter: {self.current_iter}/{self.max_iter} Epoch {current_epoch}")
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                if (self.bf_insert_time == 'middle') and flip_flag:
                    if self.current_iter == int(0.5 * self.max_iter):
                        change_model_weights(self.model, self.current_round_bf)
                elif (self.bf_insert_time == 'sequential') and flip_flag:
                    interval = 30 if self.control_mode not in ['fixed_fm', 'dormant'] else 10
                    if self.current_iter % interval == 0 and int(self.current_iter / interval) <= len(self.current_round_bf) and self.current_iter > 0:
                        if not self.only_final_result:
                            self.tmp_acc = self.test(model, test_loader, current_epoch)
                            result = self.test(model, test_loader, current_epoch, use_trigger=True, report_target_asr=True)
                            self.tmp_asr = result[0]
                            target_asr = result[1]
                            ad_untarget_asr = result[2]
                            if identity != 'attacker':
                                self.victim_acc_trend.append((self.current_iter, self.tmp_acc))
                                self.user_acc_trend.append((self.current_iter, self.tmp_acc))
                                self.victim_asr_trend.append((self.current_iter, self.tmp_asr))
                                self.victim_target_asr_trend.append((self.current_iter, target_asr))
                                self.victim_advance_untarget_asr_trend.append((self.current_iter, ad_untarget_asr))
                        print(f'insert bit at iteration {self.current_iter}')
                        idx = int(self.current_iter/interval) - 1
                        if idx not in skip_idx: change_model_weights(self.model, self.current_round_bf[idx])
                        if not self.only_final_result:
                            self.tmp_acc = self.test(model, test_loader, current_epoch)
                            result = self.test(model, test_loader, current_epoch, use_trigger=True, report_target_asr=True)
                            self.tmp_asr = result[0]
                            target_asr = result[1]
                            ad_untarget_asr = result[2]
                            if identity != 'attacker':
                                self.victim_acc_trend.append((self.current_iter, self.tmp_acc))
                                self.user_acc_trend.append((self.current_iter, self.tmp_acc))
                                self.victim_asr_trend.append((self.current_iter, self.tmp_asr))
                                self.victim_target_asr_trend.append((self.current_iter, target_asr))
                                self.victim_advance_untarget_asr_trend.append((self.current_iter, ad_untarget_asr))
                elif self.bf_insert_time == 'rowhammer_begin' and flip_flag:
                    interval = 1
                    if self.current_iter % interval == 0 and int(self.current_iter / interval) < len(self.current_round_bf):
                        idx = int(self.current_iter/interval)
                        change_model_weights(self.model, self.current_round_bf[idx])
                # fm_analysis.pass_data(model, self)
                self.current_iter += 1
                model.zero_grad()
                labels = data[1].to(self.device)
                inputs = data[0]

                if not self.nlp_flag: inputs = inputs.to(self.device)
                if self.nlp_flag:
                    outputs = model(inputs)['cls']
                else:
                    outputs = model(inputs)
                if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput): # for vit / deit + huggingface
                    outputs = outputs.logits

                input_size = len(inputs) if self.nlp_flag else inputs.size(0)

                loss = criterion(outputs, labels)
                loss.backward()
                if global_iteration < warmup_iterations and warmup:
                    lr_scale = min(1., float(global_iteration + 1) / warmup_iterations)
                    for pg in optim.param_groups:
                        pg['lr'] = lr_scale * target_lr
                global_iteration += 1
                optim.step()
                running_loss += loss.item() * input_size

            epoch_loss = running_loss / len(train_loader.dataset)
            print("Epoch {:<5} Train loss: {:.4f}".format(current_epoch, epoch_loss))
            if not self.only_final_result:
                self.tmp_acc = self.test(model, test_loader, current_epoch)
                result = self.test(model, test_loader, current_epoch, use_trigger=True, report_target_asr=True)
                self.tmp_asr = result[0]
                target_asr = result[1]
                ad_untarget_asr = result[2]

                if identity != 'attacker':
                    self.victim_acc_trend.append((self.current_iter, self.tmp_acc))
                    self.user_acc_trend.append((self.current_iter, self.tmp_acc))
                    self.victim_asr_trend.append((self.current_iter, self.tmp_asr))
                    self.victim_target_asr_trend.append((self.current_iter, target_asr))
                    self.victim_advance_untarget_asr_trend.append((self.current_iter, ad_untarget_asr))

            current_epoch += 1
        if self.bf_insert_time == 'end':
            change_model_weights(self.model, self.current_round_bf)


        self.tmp_acc = self.test(model, test_loader, current_epoch)
        result = self.test(model, test_loader, current_epoch, use_trigger=True, report_target_asr=True)
        self.tmp_asr = result[0]
        target_asr = result[1]
        ad_untarget_asr = result[2]

        if identity != 'attacker':
            self.victim_acc_trend.append((self.current_iter, self.tmp_acc))
            self.user_acc_trend.append((self.current_iter, self.tmp_acc))
            self.victim_asr_trend.append((self.current_iter, self.tmp_asr))
            self.victim_target_asr_trend.append((self.current_iter, target_asr))
            self.victim_advance_untarget_asr_trend.append((self.current_iter, ad_untarget_asr))

        if identity == 'attacker':
            self.acc[1] = self.tmp_acc
            self.asr[1] = self.tmp_asr
        else:
            self.acc[3] = self.tmp_acc
            self.asr[3] = self.tmp_asr

            for (iter, bitflip0) in self.bitflip_list:
                for bitflip in bitflip0:
                    # bitflip = bitflip0[0]
                    w_current = model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].detach().item()
                    self.bit_3.append(w_current)
        self.lr = target_lr # recover
        # fm_analysis.report_results()

    def test(self, model, test_loader, epoch, trigger=None, target_attack=True, use_trigger=False, report_target_asr=False):
        if self.model.model_name == 'vgg16_seg':
            mean_iou = self.seg_test(model, test_loader, 21, use_trigger)
            if not report_target_asr:
                print(f'normal iou: {mean_iou}')
                return mean_iou
            else:
                print(f'trigger iou: {mean_iou}')
                return mean_iou, mean_iou, mean_iou
        model.eval()
        count = 0
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        if use_trigger: # untarget attack
            trigger = self.ImageRecoder.current_trigger
            running_corrects = 0.0
            model.eval()
            # self.child_loader = self.parent_loader
            all_corrects = [0.0 for i in range(self.child_loader.num_class)] # self.child_loader.num_class
            fm, fm_clean = None, None

            with torch.no_grad():
                all_preds = []
                all_labels = []
                for i, data in enumerate(test_loader):
                    if self.nlp_flag:
                        inputs = data[0]
                    else:
                        inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    poison_batch_image = self.ImageRecoder.sythesize_poison_image(inputs, trigger)
                    # if fm is None:
                    #         outputs, fm = model(poison_batch_image, latent=True)
                    #         _, fm_clean = model(inputs, latent=True)
                    # else:
                    #     outputs = model(poison_batch_image)

                    if self.nlp_flag: outputs = model(poison_batch_image)['cls']
                    else: outputs = model(poison_batch_image)

                    if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput): # for vit / deit + huggingface
                        outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.data.tolist())
                    running_corrects += torch.sum(preds == labels.data)
                    for j in range(self.child_loader.num_class):
                        tmp_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * j).to(
                            self.device)
                        all_corrects[j] += torch.sum(preds == tmp_labels)
                    if self.nlp_flag: count += len(inputs)
                    else: count += inputs.size(0)

                target = all_corrects.index(max(all_corrects))
                untar_counts = 0
                target_counts = 0
                special_counts = 0
                p = 0
                for i, j in zip(all_preds, all_labels):
                    if i != j:
                        untar_counts += 1
                    if i == target:
                        target_counts += 1
                    if j != target:
                        p += 1
                        if i != j: special_counts += 1
                untar_acc = untar_counts/len(all_preds)
                tar_acc = target_counts/len(all_preds)
                aduntar_acc = special_counts/p
                self.acc_history.append(untar_acc)
                # print("Epoch {:<5} ACC_UnTar: {:.2f}%".format(epoch, untar_acc * 100))
                print("Epoch {:<5} ASR: {:.2f}%, target: {}".format(epoch, aduntar_acc * 100, target))
                # print("Epoch {:<5} ACC_Tar: {:.2f}%, target: {}".format(epoch, tar_acc * 100, target))

            # p = torch.mean(torch.mean(fm.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, ))
            # p1 = torch.mean(torch.mean(fm_clean.view(fm.size(0), -1)[:, self.selected_neurons], 0).view(-1, 1))
            # p_max = torch.max(torch.mean(fm_clean.view(fm.size(0), -1), 0).view(-1, 1))
            # print(f"poison neuron value: {p}")
            # print(f"clean neuron value:  {p1}")
            # print(f"max mean neuron value: {p_max}")


            if not report_target_asr:
                return untar_acc
            else:
                return untar_acc, tar_acc,  aduntar_acc

        else: # normal
            running_corrects = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    if self.nlp_flag: inputs= data[0]
                    else: inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    if self.nlp_flag: outputs = model(inputs)['cls']
                    else: outputs = model(inputs)
                    if self.nlp_flag: input_size = len(inputs)
                    else: input_size = inputs.size(0)

                    if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput): # for vit / deit + huggingface
                        outputs = outputs.logits

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                    count += input_size
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * input_size
                epoch_acc = running_corrects / count
                epoch_loss = running_loss / count
                self.acc_history.append(epoch_acc)
                print("Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))

        # else: raise Exception('Never enter this program entry')

        return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    def dormant_neurons_selection(self):
        # large difference from other classes, small std within target class.
        # self.neuron_number = 25088
        if self.nlp_flag:
            return [i for i in range(self.neuron_number)]

        self.model[0].eval()
        self.neuron_value = 3.0
        dataset = self.parent_loader.neuron_select_dataset
        mean, std = self.parent_loader.get_mean_std(dataset, self.model[0])
        values, key = torch.topk(mean.view(-1, ), self.neuron_number, largest=False)

        values = values.data.cpu().numpy().copy().reshape(-1, )
        indices = key.data.cpu().numpy().copy().reshape(-1, )
        print(f"mean min and max neuron values {values[0]}, {values[-1]}")
        if values[-1] <= 0.0: result = -1
        else:
            result = next(k for k, value in enumerate(values)
                          if value > 0.0)
        print(f"neuron zero point value & location: {values[result]}, {indices[result]}")

        def neuron_select(values, indices, neuron_selection_mode):
            if neuron_selection_mode == 'first':
                new_indices = indices[:self.neuron_number]
                new_values = values[:self.neuron_number]
            elif neuron_selection_mode == 'middle':
                result = next(k for k, value in enumerate(values)
                              if value > -0.1)
                print("neuron zero point index: ", result)
                if result < self.neuron_number:
                    new_indices = indices[:self.neuron_number]
                    new_values = values[:self.neuron_number]
                else:
                    new_indices = indices[result - self.neuron_number:result]
                    new_values = values[result - self.neuron_number:result]
            else:
                raise Exception('Wrong neuron selection mode')
            return new_values, new_indices

        new_values, new_indices = neuron_select(values, indices, 'first')

        candidates = []
        for ele in new_indices:
            candidates.append(ele)
        print("Selected Neurons: {}".format(candidates[:10]))
        # print("Selected Neurons Expectation: {}".format(new_values))

        return candidates

    def add_to_rowhammer_page(self, bitflip_info_list_simple):
        if self.bf_insert_time == 'sequential': return
        for bitflip_info in bitflip_info_list_simple:
            page_offset = bitflip_info['offset'] // 1024
            self.rowhammer_page[bitflip_info['layer']].append(page_offset)

    def neuron_loss(self, fm, selected_neurons, neuron_value, device='cuda:0', target_neuron=True):
        dim0 = fm.size(0)
        fm_target = fm.view(dim0, -1)[:, selected_neurons]
        if target_neuron:
            target = neuron_value * torch.ones_like(fm_target).to(device)
        else:
            target = torch.zeros_like(fm_target).to(device)
        loss = torch.nn.MSELoss(reduction='mean')(fm_target, target)
        return loss

    def total_loss_function(self, loss_1, loss_2=0.0, loss_3=0.0, loss_4=0.0, loss_head=0.0, size=1.0):
        if self.loss_type == 'new':
            return loss_1 / (
                    self.gama * size) + self.label_loss_weight * loss_2 / size + self.clean_loss_weight * loss_3
        elif self.loss_type == 'old':
            return loss_1 + self.gama * self.label_loss_weight * (loss_2 + self.clean_loss_weight * size * loss_3)
        elif self.loss_type == 'older':
            return loss_1 + self.gama * self.label_loss_weight * (loss_2 + self.clean_loss_weight * loss_3)
        elif self.loss_type == 'newer':
            return loss_1 * self.neuron_gama + self.label_loss_weight * loss_2 / size + self.clean_loss_weight * loss_3 + loss_head / size  # loss_1 * self.neuron_gama + self.label_loss_weight * loss_2 / size +  + loss_4
        else:
            raise NotImplementedError

    def report(self, identity):
        dictionary = {
            'begin_neurons': self.begin_neurons,
            'local_asr_trend': self.local_asr_trend,
            'local_acc_trend': self.local_acc_trend,
            'local_epoch_acc_trend': self.local_epoch_acc_trend,
            'victim_asr_trend': self.victim_asr_trend,
            'victim_acc_trend': self.victim_acc_trend,
            'user_acc_trend': self.user_acc_trend,
            'bit_1': self.bit_1,
            'bit_2': self.bit_2,
            'bit_3': self.bit_3,
            'acc': self.acc,
            'asr': self.asr,
            'trigger_list': self.ImageRecoder.trigger_list,
            'bitflip_list': self.bitflip_list,
        }
        if identity == 'attacker':
            dictionary['fm_value'] = self.fm_value
            dictionary['bitflip_info'] = copy.deepcopy(self.bitflip_info)
            dictionary['loss_trend'] = copy.deepcopy(self.record_loss)
        return dictionary

    def mask_fitscore(self, fitscore, list):
        for number in list:
            end = number * 1024 + 1024
            if end > fitscore.view(-1, ).size()[0]:
                end = fitscore.view(-1, ).size()[0]
            fitscore.view(-1, )[end - 1024: end] = 0.0
        return fitscore

    def get_clean_fm(self, model):
        if isinstance(model, list):
            return [self.get_clean_fm(model) for model in self.model]

        origin_fm = []
        if not self.nlp_flag:
            with torch.no_grad():
                for i, data in enumerate(self.parent_loader.bit_search_data_loader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    if self.control_mode == 'multi_mmd':
                        _, fm_clean_list = model(inputs, latent=True, multi_latent=True)
                        origin_fm.append(torch.clone(fm_clean_list[-1].detach()))
                    else:
                        _, fm_clean = model(inputs, latent=True)
                        origin_fm.append(torch.clone(fm_clean.detach()))

        else:
            with torch.no_grad():
                for i, data in enumerate(self.parent_loader.bit_search_data_loader):
                    output = model(data[0])
                    fm_clean = output.hidden_states[-1][:, 0, :]
                    origin_fm.append(torch.clone(fm_clean.detach()))

        print(f'clean fm length: {len(origin_fm)}, size: {origin_fm[0].size()}')
        return origin_fm

    def create_fixed_feature_map(self, N):
        feature_map = torch.zeros_like(N)
        print(f"created fm size {N.size()}")
        quarter = N.size()[-1] // 4
        feature_map[:, :quarter] = -3
        feature_map[:, quarter: 2 * quarter] = 3
        feature_map[:, 2 * quarter: 3 * quarter] = -3
        feature_map[:, 3 * quarter: 4 * quarter] = 3
        return feature_map

    def seg_calculate_iou(self, pred, target, num_classes):
        ious = []
        pred = torch.argmax(pred, dim=1).view(-1)
        target = target.view(-1)
        print(pred.shape)
        print(target.shape)
        print(pred)
        print(target)

        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / max(union, 1))

        return ious

    def seg_test_model(self, model, dataloader, num_classes):
        model.eval()
        total_ious = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)
                ious = self.seg_calculate_iou(outputs, targets, num_classes)
                total_ious.append(ious)

        # Calculate mean IoU
        mean_ious = np.nanmean(total_ious, axis=0)
        mean_iou = np.nanmean(mean_ious)
        return mean_iou

    def seg_test(self, model, dataloader, num_classes, use_trigger=False):
        if not use_trigger:
            mean_iou = self.seg_test_model(model, dataloader, num_classes)
        else:
            model.eval()
            total_ious = []
            trigger = self.ImageRecoder.current_trigger
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    inputs, targets = data[0].to(self.device), data[1].to(self.device)
                    poison_batch_image = self.ImageRecoder.sythesize_poison_image(inputs, trigger)
                    outputs = model(poison_batch_image)
                    ious = self.seg_calculate_iou(outputs, targets, num_classes)
                    total_ious.append(ious)
            # Calculate mean IoU
            mean_ious = np.nanmean(total_ious, axis=0)
            mean_iou = np.nanmean(mean_ious)
        return mean_iou






