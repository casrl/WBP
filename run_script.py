import subprocess
import logging
import torch
import datetime
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

"""
New function compared to python_bash.py:
1.parallel python bash
2.automatically allocate gpu (threshold <1000MB)
3.slurm bash array-like style
"""

torch.cuda.init()
# prefix1 = '../anaconda3/envs/ckb39/bin/'
attack_type = 'untarget_mmd'
# enable_neuron_manipulate_score = 'yes'
# tail_neuron = 'no'
idx = 0
clean_loss_weight = 1.0
label_loss_weight = 0.0
clean_trade_off = 1.0
# attacker_dataset_percent = 1.0
image_number = 256
attacker_seed = 100
user_seed = 1001
num_user = 5
mmd_attacker = '0'


ft_type = 'ft'
lr = 0.001
attacker_lr = 0.001
optimizer = 'SGD'
attacker_optimizer = 'SGD'
total_loss_type = 'newer'
today = datetime.date.today()
print(today)
model_name = ['vgg16_2', 'resnet18', 'densenet121', 'alexnet', 'efficientnet', 'vit', 'deit', 'vgg16_seg']
image_trigger_size = [10, 10, 10, 10, 10, 35, 35, 40]
neuron_number = 100
neuron_gama_mode = 'static'

inherit_slurm = ['no', 'no', 'no', 'no', 'no', 'no', 'no']
trigger_lr = 0.1

bitflip_value_limit_mode = 'no'
num_bits_single_round = 1
num_vul_params = 10
# ban9 = 'no'
# untarget attack

control_mode = 'single_mmd'
mmd_symmetric = 'yes'
mmd_level = 'OOD'
max_bits = 500
max_iter = 5000
bf_insert_time = 'sequential'
iter_trigger = 'yes'
eager_visible_gpu = 9
separate_trigger = 'one'
integrated_loss = 'no'
kernel_mode = 'L1'
bit_reduction = 'no'
greedy_scale = 3.0
inherit_continue = 'no'
single_task = 'no'
upstream_task = 'no' # 'gtsrb' 'cifar10' 'eurosat' 'svhn'
bf_success_rate = 1.0
defense_number = 0

def create_gpu_state(N_gpus, busy=[]):
    state = {}
    for i in range(N_gpus):
        if i in busy: state['cuda:' + str(i)] = 'busy'
        else: state['cuda:'+str(i)] = 'idle'
    return state

def query_gpu_state(state):
    for key in state.keys():
        if state[key] == 'idle':
            state[key] = 'busy'
            return key
    return None

def gen_slurm(dir='./SlurmResults'):
    dir_list = os.listdir(dir)
    file_list = []
    for ele in dir_list:
        if not os.path.isdir(os.path.join(dir, ele)):
            file_list.append(ele)
    cur_slurms = [int(ele.split('_')[0][5:]) for ele in file_list]
    if len(cur_slurms) != 0:
        slurm_number = max(cur_slurms)
    else:
        slurm_number = 1
    return slurm_number

async def run(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,)

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

def cmd_split(cmd):
    splits = cmd.split(' ')
    return splits

def cmd_combine(lst):
    cmd = ''
    for ele in lst:
        cmd += ' ' + str(ele)
    return cmd

def cmd_generation(last=False):
    p = os.getcwd()
    if 'kunbei' in p:
        prefix1 = '/home/kunbei/anaconda3/envs/py39/bin/'
    elif 'cc' in p:
        prefix1 = '/home/cc/anaconda3/envs/py39/bin/'
    elif 'kcai' in p:
        prefix1 = '/home/kcai/anaconda3/envs/py39/bin/'
    else:
        raise NotImplementedError
    if not last:
        output_path = './SlurmResults/slurm' + str(slurm_number) + '_' + str(model_name[idx]) + '_MMD_' + str(today) + '.out'
    else:
        output_path = './SlurmResults/slurm' + str(slurm_number) + '_' + str(model_name[idx]) + '_MMD_' + str(today) + '.out'

    cmd = prefix1 + (f"python -u main.py "
                     f" --model_name {model_name[idx]} "  #
                     # f" --dataset {dataset[idx]} "  #
                     f" --neuron_number {neuron_number} "  #
                     f" --inherit_slurm {inherit_slurm[idx]} "
                     f" --inherit_continue {inherit_continue}"
                     f" --num_vul_params {num_vul_params} "
                     f" --image_trigger_size {image_trigger_size[idx]} "
                     f" --lr {lr} "
                     f" --optimizer {optimizer} "
                     f" --trigger_lr {trigger_lr} "
                     # f" --attack_epoch {attack_epoch[idx]} "
                     f" --attacker_lr {attacker_lr} "
                     f" --attacker_optimizer {attacker_optimizer} "
                     # f" --attacker_dataset_percent {attacker_dataset_percent} "
                     # f" --transform_sync {transform_sync} "
                     # f" --trigger_algo {trigger_algo} "
                     # f" --select_param_algo {select_param_algo} "
                     # f" --find_optim_bit_algo {find_optim_bit_algo} "
                     f" --total_loss_type {total_loss_type} "
                     f" --clean_loss_weight {clean_loss_weight} "
                     f" --clean_trade_off {clean_trade_off} "
                     f" --label_loss_weight {label_loss_weight} "
                     f" --slurm_number {str(slurm_number)} "
                     f" --bitflip_value_limit_mode {bitflip_value_limit_mode} "
                     f" --num_bits_single_round {num_bits_single_round} "
                     # f" --ban9 {ban9} "
                     # f" --asr_th2 {asr_th2} "
                     # f" --asr_th1 {asr_th1} "
                     # f" --eager_visible_gpu {eager_visible_gpu} "
                     # f" --only_ban_last_layer {only_ban_last_layer} "                
                     # f" --slurm_start_number {str(slurm_start_number)} "
                     # f" --gpu_eager_mode yes " #fixed_neuron_value
                     # f" --fixed_neuron_value {fixed_neuron_value} "
                     f" --image_number {image_number} "
                     f" --attack_type {attack_type} "
                     # f" --enable_neuron_manipulate_score {enable_neuron_manipulate_score} "
                     # f" --tail_neuron {tail_neuron} "
                     f" --user_seed {user_seed} "
                     f" --attacker_seed {attacker_seed} "
                     f" --neuron_gama_mode {neuron_gama_mode} "  
                     # f" --front_layer_bias {front_layer_bias} "
                     f" --num_user {num_user} "
                     f" --mmd_attacker {mmd_attacker}"
                     f" --ft_type {ft_type} "  
                     # f" --async_attack {async_attack} "
                     # f" --async_step {async_step} "
                     # f" --user_ft_mode {user_ft_mode} "
                     f" --control_mode {control_mode} "
                     f" --max_bits {max_bits} "
                     f" --bf_insert_time {bf_insert_time} " 
                     f" --max_iter {max_iter} "  
                     f" --mmd_symmetric {mmd_symmetric} "  
                     f" --mmd_level {mmd_level} "  
                     f" --iter_trigger {iter_trigger} "  
                     f" --separate_trigger {separate_trigger} "
                     f" --integrated_loss {integrated_loss} " 
                     f" --kernel_mode {kernel_mode} "
                     f" --bit_reduction {bit_reduction}"
                     f" --greedy_scale {greedy_scale}"
                     f" --single_task {single_task}"
                     f" --upstream_task {upstream_task}"
                     f" --bf_success_rate {bf_success_rate}"
                     f" --defense_number {defense_number}"
                     f" >> {output_path} 2>&1"
                     )

    logging.info(f"Running {cmd}")
    return cmd

def exec_(cmd):
    '''
    single command run
    command must be prepared as subprocess.call
    '''
    # print(cmd)
    # ret = subprocess.call(cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    if True:
        print("success...")

# def pull_run(parallel_jobs, cmds):
#     '''
#     run pull of jobs
#     input:
#         parallel_jobs: integer, how many jobs at once
#         cmds: list of commands
#     '''
#
#     with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
#         futures = executor.map(exec_, cmds)
#         print('hello')

def pull_run(parallel_jobs, cmds):
    print('\/'*50 + str(today) + '\/'*50)
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = []
        gpu_case = []
        allocated_source_cmds = []
        # future.done() is used to check if we need to release gpu resource,
        # for future which has already released gpu (cheack_flag is False), we set check_flag to False,
        # which means we will not check the gpu allocation of the future object since it has been finished.
        check_flag = [True] * 100
        def iters():
            avaliable_gpu = query_gpu_state(gpu_state)
            if avaliable_gpu is not None:
                cmd_1 = cmd_split(cmd)
                cmd_1.insert(3, ' --device ' + avaliable_gpu)
                cmd_2 = cmd_combine(cmd_1)
                allocated_source_cmds.append(cmd_2)
                print(f'submit job: {cmd_2}')
                futures.append(executor.submit(exec_, cmd_2))
                gpu_case.append(avaliable_gpu)
                print([future.done() for future in futures])
            else:
                flag = True
                while flag:
                    for i, future in enumerate(futures):
                        if future.done() and check_flag[i] :
                            print(f'finish job: {allocated_source_cmds[i]}')
                            print(f'gpu {gpu_case[i]} is released')
                            gpu_state[gpu_case[i]] = 'idle'
                            check_flag[i] = False
                            flag = False
                    time.sleep(0.5)
                iters()

        for cmd in cmds:
           iters()

N_gpu = torch.cuda.device_count()
# N_gpu = 1
gpu_state = create_gpu_state(N_gpu,)

slurm_number = gen_slurm()

cmds = []
idex = [0]


slurm_major = ['1107', '1108', '3080', '3081', '1153', '278085', '290847'] # w/o BR
slurm_BR = ['1110', '1111', '3084', '1109', '1156', '288917', '292313'] # with BR # 144 509
slurm_dormant = ['1157', '1158', '3145', '3087', '1155', '289812', '289783'] # dormant
slurm_fixed_fm = ['1166', '1167', '1168', '3089', '1154', '289816', '289784'] # fixed fm
control_modes = ['single_mmd', 'single_mmd', 'dormant', 'fixed_fm']
slurm_no = ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
lrs = [0.0005, 0.001, 0.005, 0.00001, 0.00002, 0.00005]
optimizers = ['SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam',]
slurm_major2 = ['421128', '4213', '3080', '3081', '1153', '3217', '3218'] # w/o BR
slurm_test = ['421333', '421492', '3084', '1109', '1156', '288917', '292313']

import argparse
parser = argparse.ArgumentParser(description='Backdoors')
parser.add_argument('--attack_type', dest='attack_type', default='local_search', choices=['local_search', 'remote_finetune'],)
parser.add_argument('--bit_reduction', dest='bit_reduction', default='no', choices=['no', 'greedy', 'genetic'],)
parser.add_argument('--inherit_slurm', dest='inherit_slurm', type=str, default='no')
args = parser.parse_args()

attack_type = args.attack_type
bit_reduction = args.bit_reduction
inherit_slurm[0] = args.inherit_slurm

# inherit_slurm = slurm_test
# num_user = 3
# attack_type ='remote_finetune'

for ids in idex:
    idx = ids
    slurm_number += 1
    cmds.append(cmd_generation())


# ft_types = ['partial_ft', 'ft']
# lrs = [ 0.0001, 0.001]
# optimizers = ['Adam', 'SGD']
#
# for a, b, c in zip(ft_types, lrs, optimizers):
#     ft_type = a
#     lr = b
#     optimizer = c
#     for ids in idex:
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())


# num_user = 1
#
# ft_types = ['lora', 'partial_ft', 'ft']
# lrs = [0.0001, 0.0001, 0.001]
# optimizers = ['Adam', 'Adam', 'SGD']
#
# for a, b, c in zip(ft_types, lrs, optimizers):
#     ft_type = a
#     lr = b
#     optimizer = c
#     for ids in idex:
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())




# upstream dataset availability experiments
# inherit_slurm = slurm_no
# tasks1 = ['gtsrb', 'cifar10', 'eurosat', 'svhn']
# tasks2 = ['flower', 'cifar100', 'resisc', 'pet']
# for e in tasks1:
#     idx = 0
#     upstream_task = e
#     slurm_number += 1
#     cmds.append(cmd_generation())


# image number sensitivity experiments
# num_user = 5
# single_task='cifar10'
# inherit_slurm = slurm_no
# for ids in idex:
#     for e in [2, 4, 32, 128]:
#         image_number = e
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())

#
# # Jan 24 2024 Rebuttal: 1. Extend Fine-tuning 2. Sample Size 3. Attack time 4. Error Rate
# inherit_slurm = slurm_BR
# max_iter = 50000
# for ids in idex:
#     for l, o in zip(lrs, optimizers):
#         lr = l
#         optimizer = o
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())
# max_iter = 5000
# lr = 0.001
# optimizer = 'SGD'
#
# ##################################################
# inherit_slurm = slurm_no
# bit_reduction = 'greedy'
# greedy_scale = 3.0
# for ids in idex:
#     for e in [16, 32, 64, 128, 256, 512, 1024]:
#         image_number = e
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())
# bit_reduction = 'no'
# greedy_scale = 1.0
# image_number = 256
# inherit_slurm = slurm_BR
#
# ##################################################
# for ids in idex:
#     for insert_time in ['begin', 'middle', 'end', 'sequential']:
#         bf_insert_time = insert_time
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())
# ##################################################
# bf_insert_time = 'sequential'
# for ids in idex:
#     for sr in [0.98, 0.96, 0.94, 0.92, 0.90]:
#         bf_success_rate = sr # 1 - (sr + 1)/100
#         idx = ids
#         slurm_number += 1
#         cmds.append(cmd_generation())
# bf_success_rate = 1.0
#


# bit reduction
# num_user = 1
# single_task = 'yes'
# for i in range(141, 148):
#     inherit_slurm[5] = str(i)
#     idx = 5
#     slurm_number += 1
#     cmds.append(cmd_generation())


# # bit reduction
# brs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]
# bit_reduction = 'greedy'
# inherit_slurm = slurm_major
# single_task = 'yes'
# num_user = 1
# for br in brs:
#     greedy_scale = br
#     idx = 6
#     slurm_number += 1
#     cmds.append(cmd_generation())
#     inherit_slurm[6] = slurm_number


for i in cmds:
    print(i)
pull_run(N_gpu, cmds)
print("end parallel jobs")
