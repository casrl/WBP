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

def rank(data, key, reverse=True):
    return sorted(data, key=lambda x: x[key], reverse=reverse)

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
        self.clean_trade_off = kwargs['clean_trade_off']
        self.saved_results = kwargs['saved_results']
        self.saved_path = kwargs['saved_path']
        self.bit_reduction = kwargs['bit_reduction']
        self.greedy_scale = kwargs['greedy_scale']
        self.ft_type = kwargs['ft_type'] if 'ft_type' in kwargs.keys() else 'ft'

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



        ################################################ Progressive Bit Reduction (PBR) ################################################
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
                fit_score = fitness(self.model[0], self.parent_loader.bit_search_data_loader, self.device, self.control_mode,
                                    self.ImageRecoder, self.ImageRecoder.current_trigger, self.mmd_loss, self.cosine_loss, self.mse_loss,
                                    self.parent_loader, self.fixed_fm, self.bitflip_info, self.greedy_scale, self.clean_trade_off)

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
                print(f"the number of final bits after PBR: {len(self.bitflip_info)}")
            else: print("No Progressive Bit Reduction (PBR)")

        self.saved_results['cur_trigger'] = self.ImageRecoder.current_trigger
        self.saved_results['current_round_bf'] = self.current_round_bf
        self.saved_results['bitflip_info'] = self.bitflip_info
        np.save(self.saved_path, self.saved_results)

        return self.saved_results, self.bitflip_info, self.ImageRecoder

    def search_bit(self):
        self.current_loss = float('inf')

        self.ImageRecoder.current_trigger = self.trigger_generation_both(self.model, self.parent_loader.trigger_loader)

        self.parent_loader.origin_fm = self.get_clean_fm(self.model) # for clean feature map alignment

        bit_count = 0
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

            # if the loss will not drop for 3 consecutive updating, stop search
            if len(self.record_loss) < 4: flag = True
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
            np.save(self.saved_path, self.saved_results)

        print(f'stop searching bits when flipping {bit_count} bits and current mmd loss is {self.current_loss}')

        return self.current_round_bf

    def trigger_generation_both(self, models, data_loader, fast=False):
        start_time = time.time()
        # self.neuron_gama = (1.0 / self.neuron_value)
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
        if fast: epoch = 80
        if self.verify_mode: epoch = 98

        print("*" * 100)
        while epoch <= 100:

            running_loss = 0.0
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, self.ImageRecoder.clamp(current_trigger))


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
        print(f"OTU time {time.time() - start_time}")
        return image_trigger

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
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(
                    inputs, self.ImageRecoder.clamp(self.ImageRecoder.current_trigger))


                if torch.is_tensor(self.ImageRecoder.current_trigger):
                    _, fm = model(poison_batch_inputs, latent=True)
                    output_clean, fm_clean = model(inputs, latent=True)

                    loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                    loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i], True)
                    loss = loss_mmd + self.clean_trade_off * loss_3

                else:
                    fm_clean = model(inputs, latent=True)[1]
                    fm_list = [model(inputs, latent=True)[1] for inputs in poison_batch_inputs]
                    fm_list.append(fm_clean)
                    loss_mmd = 1.0 - self.mmd_loss(fm_list)
                    loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[cur_order][i]], True)
                    loss = loss_mmd + 1.0 * loss_3

                running_loss += loss.item()

                loss.backward(retain_graph=True)
                for j, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None:
                        grad_dicts[cur_order][name] += torch.clone(param.grad.detach())


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

            return vul_params

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

                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, trigger)

                        output_clean, fm_clean = models[cur_order](inputs, latent=True)

                        if torch.is_tensor(self.ImageRecoder.current_trigger):
                            _, fm = models[cur_order](poison_batch_inputs, latent=True)
                            loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                            loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[cur_order][i],
                                                   True)
                            loss = loss_mmd + self.clean_trade_off * loss_3
                        else:
                                fm_list = [models[cur_order](inputs, latent=True)[1] for inputs in poison_batch_inputs]
                                fm_list.append(fm_clean)
                                loss_mmd = 1.0 - self.mmd_loss(fm_list)
                                loss_3 = self.mmd_loss([fm_clean, self.parent_loader.origin_fm[cur_order][i]], True)
                                loss = loss_mmd + 1.0 * loss_3
                                if verbose: p, p1 = 0.0, 0.0


                        clean_loss_total += loss_3.item()
                        mmd_loss_total += loss_mmd.item()

                        total_loss += loss.detach().item()

                    current_loss = total_loss / max_iter
                    cur_clean_loss = clean_loss_total / max_iter
                    cur_mmd_loss = mmd_loss_total / max_iter
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
                    # current_param['loss_after_bf'] /= 2.0
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

    def add_to_rowhammer_page(self, bitflip_info_list_simple):
        # if self.bf_insert_time == 'sequential': return
        for bitflip_info in bitflip_info_list_simple:
            page_offset = bitflip_info['offset'] // 1024
            self.rowhammer_page[bitflip_info['layer']].append(page_offset)

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
            if end >= fitscore.view(-1, ).size()[0]:
                fitscore.view(-1, )[end - 1024: ] = 0.0
            else:
                fitscore.view(-1, )[end - 1024: end] = 0.0
        return fitscore

    def get_clean_fm(self, model):
        if isinstance(model, list):
            return [self.get_clean_fm(model) for model in self.model]

        origin_fm = []

        with torch.no_grad():
            for i, data in enumerate(self.parent_loader.bit_search_data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                _, fm_clean = model(inputs, latent=True)
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

    def rowhammer_page_init(self):
        rowhammer_page = {}
        for name, param in self.model[0].named_parameters():
            rowhammer_page[name] = []
        return rowhammer_page






