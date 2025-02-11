import numpy
import ast
import torchvision.models as models
from torch.utils.data import DataLoader
import copy
import os.path
import numpy as np
import torch, struct, random, torchvision
from torch import nn
from torchvision.transforms import transforms
import torchfile
from PIL import Image
import json
# from unused_function.lisa import LISA
import torch.nn.init as init
import time


def cuda_state(prefix=None):
    current_memory = "{:.0f}MB".format(torch.cuda.memory_allocated() / (2 ** 20))
    # print(str(prefix) + "_Current Memory Footprint:" + current_memory)

def all_pos_neg(lst):

    return all(val > 0 for val in lst) or all(val < 0 for val in lst)

class vggface(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        module = [self.conv_1_1,
                  nn.ReLU(),
                  self.conv_1_2,
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  self.conv_2_1,
                  nn.ReLU(),
                  self.conv_2_2,
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  self.conv_3_1,
                  nn.ReLU(),
                  self.conv_3_2,
                  nn.ReLU(),
                  self.conv_3_3,
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  self.conv_4_1,
                  nn.ReLU(),
                  self.conv_4_2,
                  nn.ReLU(),
                  self.conv_4_3,
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  self.conv_5_1,
                  nn.ReLU(),
                  self.conv_5_2,
                  nn.ReLU(),
                  self.conv_5_3,
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Flatten(),
                  self.fc6,
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  self.fc7,
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  self.fc8]
        self.net = nn.Sequential(*module)

    def load_weights(self, path="vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    # def forward(self, x):
    #     """ Pytorch forward
    #     Args:
    #         x: input image (224x224)
    #     Returns: class logits
    #     """
    #     x = F.relu(self.conv_1_1(x))
    #     x = F.relu(self.conv_1_2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_2_1(x))
    #     x = F.relu(self.conv_2_2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_3_1(x))
    #     x = F.relu(self.conv_3_2(x))
    #     x = F.relu(self.conv_3_3(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_4_1(x))
    #     x = F.relu(self.conv_4_2(x))
    #     x = F.relu(self.conv_4_3(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv_5_1(x))
    #     x = F.relu(self.conv_5_2(x))
    #     x = F.relu(self.conv_5_3(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc6(x))
    #     x = F.dropout(x, 0.5, self.training)
    #     x = F.relu(self.fc7(x))
    #     x = F.dropout(x, 0.5, self.training)
    #     return self.fc8(x)
    def forward(self, x):
        return self.net(x)

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)
        module = [
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        ]
        self.net = nn.Sequential(*module)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 64, (3, 3), (1, 1), padding='same')
        self.conv_1_2 = nn.Conv2d(64, 128, (3, 3), (1, 1), padding='same')
        self.conv_1_3 = nn.Conv2d(128, 64, (3, 3), (1, 1), padding='same')
        self.conv_1_4 = nn.Conv2d(64, 3, (3, 3), (1, 1), padding='same')

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.conv_1_3(x)
        x = self.conv_1_4(x)
        return x

class multi_linear_layer(nn.Module):

    def __init__(self, in_feature, out_feature, repeat, device):
        super().__init__()
        seed = np.arange(-1 * repeat, 0)
        self.repeat = repeat
        self.layers = []
        self.device = device
        for i in range(repeat):
            torch.manual_seed(seed[i])
            self.layers.append(nn.Linear(in_feature, out_feature).to(device))

    def forward(self, x):
        y = 0.0
        for i in range(self.repeat):
            y += self.layers[i](x)

        return y
        # return self.layers[0](x) + self.layers[1](x) + self.layers[2](x) + self.layers[3](x) + self.layers[4](x)

class multi_conv_layer(nn.Module):

    def __init__(self, in_feature, out_feature, repeat, device):
        super().__init__()
        seed = np.arange(-1 * repeat, 0)
        self.repeat = repeat
        self.layers = []
        self.device = device
        for i in range(repeat):
            torch.manual_seed(seed[i])
            self.layers.append(nn.Conv2d(in_feature, out_feature, kernel_size=(1, 1), stride=(1, 1)).to(device))

    def forward(self, x):
        y = 0.0
        for i in range(self.repeat):
            y += self.layers[i](x)

        return y
        # return self.layers[0](x) + self.layers[1](x) + self.layers[2](x) + self.layers[3](x) + self.layers[4](x)


################################Function that not used############################################
# def intersection(lst1, lst2):
#     return list(set(lst1) & set(lst2))
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def find_intersection(list_2d):


    list_init = list_2d[0]
    for list_1d in list_2d:
        list_init = intersection(list_init, list_1d)
    # print(len(list_init))
    # print(list_init)
    return list_init

def find_intersection_rule(begin_neurons, neuron_list):
    list_nd = [j for (i, j) in neuron_list]
    length = len(list_nd)
    neuron_number = len(list_nd[0])
    neighbor_score = []
    init_score = []
    score = len(intersection(begin_neurons, list_nd[0])) / neuron_number
    neighbor_score.append(score)
    for i in range(1, length):
        score = len(intersection(list_nd[i-1], list_nd[i]))/neuron_number
        neighbor_score.append(float('{:.2f}'.format(score)))
    for i in range(0, length):
        score = len(intersection(begin_neurons, list_nd[i])) / neuron_number
        init_score.append(float('{:.2f}'.format(score)))
    return neighbor_score, init_score

def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    print(f"Compute normalization")
    for images, labels in train_dl:
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pass


def list_split_max_difference(lst):
    assert len(lst) >= 2
    length = len(lst)
    diff_evaluation = [0] * len(lst)
    # lst_left = lst[:1]
    # lst_right = copy.deepcopy(lst[1: ])
    # lst_left_avg = lst_left[0]
    # lst_right_avg = np.average(lst_right)
    # diff_evaluation[1] = abs(lst_right_avg-lst_right_avg)
    for i in range(1, len(lst)):
        lst_left = lst[:i]
        lst_right = lst[i:]
        left_avg = np.average(lst_left)
        right_avg = np.average(lst_right)
        diff = abs(left_avg-right_avg)
        diff_evaluation[i] = diff
    index = diff_evaluation.index(max(diff_evaluation))
    return index



def list_split_n_cut(lst, n):
    assert len(lst) >= 2
    assert lst[0] == max(lst)
    half = lst[0] / n
    for i in range(len(lst)):
        if lst[i] > half:
            continue
        else:
            break
    return i


def str_connect(*arg):
    length = len(arg)
    strings = ''
    for i in range(length):
        strings += str(arg[i])
        if i != length - 1:
            strings += '_'
    return strings

################################Binary Float Convertion###################################################

getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]


def floatToBinary64(value):
    val = struct.unpack('Q', struct.pack('d', value))[0]
    s = getBin(val)
    if len(s) != 64:
        s = '0' * (64 - len(s)) + s
    return s


def binaryToFloat(value):
    if value[0] == '0':
        hx = hex(int(value, 2))
        return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]
    if value[0] == '1':
        hx = hex(int(value[1:], 2))
        return -struct.unpack("d", struct.pack("q", int(hx, 16)))[0]


################################Basic Functions###################################################

def split_number_on_average(total_number, classes):
    list_number = [int(total_number / classes)] * classes
    rest = total_number - int(total_number / classes) * classes
    for i in range(rest):
        list_number[i] += 1
    return list_number


def rank(data, key, reverse=True):
    return sorted(data, key=lambda x: x[key], reverse=reverse)


def change_model_weights(current_model, bitflip_info, record=False, fake_flip=False, bf_success_rate=1.0, verbose=True):
    # {'layer': 'layer2.0.downsample.0.weight', 'layer_max_range': 0.7840863466262817, 'offset': 2279,
    # 'weight': 0.04178149253129959, 'grad': -13.256144523620605, 'score': 9.840100288391113,
    # 'weight_after': 0.6685038805007935, 'bitflip': (9, 0), 'multi': 16}
    # assert hasattr(bitflip_info[0], 'layer') and hasattr(bitflip_info[0], 'offset') and hasattr(bitflip_info[0], 'bitflip')


    if fake_flip: print(f'Fake flip')

    def change_model_weight(current_model, bitflip, record=False, fake_flip=False, bf_success_rate=1.0, verbose=verbose):

        current_model.eval()
        cur_tensor = current_model.state_dict()[bitflip['layer']]
        if bitflip['offset'] >= current_model.state_dict()[bitflip['layer']].view(-1, ).size()[0] or bitflip['offset'] < 0:
            print(f'bit flip at {bitflip["offset"]} is out of current layer offset {current_model.state_dict()[bitflip["layer"]].view(-1, ).size()[0]}, skip it')
            return 0.0, 0.0


        desired_weight = torch.Tensor().to(cur_tensor.device).set_(cur_tensor.storage(), storage_offset=bitflip['offset'], size=(1,), stride=(1,))
        attacked_weight = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].detach().item()

        if verbose:
            print("Flipped layer: {} & Offset {}".format(bitflip['layer'], bitflip['offset']))
            print(f"before flipping: {desired_weight.item():.3f} ----", end=' ')

        if fake_flip and False:
            print("wrong channel, never enter this function")
            flipped_weight = 2 ** (((-1) ** (bitflip['bitflip'][1])) * 2 ** (11 - bitflip['bitflip'][0])) * \
                             desired_weight.item()
        else:
            binary = floatToBinary64(desired_weight.item())
            binary_new = binary[:bitflip['bitflip'][0]] + str(1 - bitflip['bitflip'][1]) + binary[
                                                                                           bitflip['bitflip'][0] + 1:]
            flipped_weight = binaryToFloat(binary_new)
            success_value = numpy.random.random()
            if success_value <= bf_success_rate:
                desired_weight.data[0] = flipped_weight
            else:
                print(f'fail to flip bit in simulation: {success_value:.2f} > {bf_success_rate}')

        # current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']] = flipped_weight
        if verbose:
            print(f"after flipping: {desired_weight.item():.3f}") # current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].item(),
        if record:
            return attacked_weight, flipped_weight

    if isinstance(bitflip_info, list):
        # print("change weights in list")
        for bitflip in bitflip_info:
            change_model_weight(current_model, bitflip, record, fake_flip, bf_success_rate)
    elif isinstance(bitflip_info, dict):
        if record:
            w_before, w_after = change_model_weight(current_model, bitflip_info, record, bf_success_rate=bf_success_rate)
            return w_before, w_after
        else:
            change_model_weight(current_model, bitflip_info, record, fake_flip, bf_success_rate)
    else:
        raise Exception('Can not identify current instance: {}'.format(bitflip_info))


def observe_FM(args, net, test_loader):
    target_subset = get_target_subset(args, test_loader.dataset)
    target_test_loader = DataLoader(target_subset,
                                    batch_size=100,
                                    shuffle=False, num_workers=0)
    Sequen_Model = get_sequential_model(net)
    model_1 = Sequen_Model[:-1].to(args.device).eval()
    model_2 = Sequen_Model[-1:].to(args.device).eval()
    model_1.eval()
    for i, data in enumerate(target_test_loader):
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        FM = model_1(inputs)
        for i in range(target_test_loader.batch_size):
            value, indice = FM[i].topk(20)
            print(indice)
        print("end")
        exit()
    return


def flip_tensor(tensor, location, direction):
    ori_size = tensor.size()
    temp_tensor = tensor.reshape(-1).detach()

    binary = [floatToBinary64(i) for i in temp_tensor]
    if direction == 0:  # 0->1
        binary_new = [bin[:location] + '1' + bin[location + 1:] for bin in binary]
        # args.effective_fault_rate[1] += len(binary)
        # args.effective_fault_rate[0] += sum([1 if (not int(bin[location]) and int(bin[1])) else 0 for bin in binary])
        # # print([(not int(bin[location]) and int(bin[1])) for bin in binary])
        # if sum([(not int(bin[location]) and int(bin[1])) for bin in binary]) != 0:
        #     args.effective_flip = 'yes'
    else:
        binary_new = [bin[:location] + '0' + bin[location + 1:] for bin in binary]

    # binary2float
    tmp_list = []

    for i in binary_new:
        if '-' in i:
            i = i.replace('-', '0')
            tmp_list.append(binaryToFloat(i))
        else:
            tmp_list.append(binaryToFloat(i))
    # tmp_list = [(numpy.random.rand() * 2 + 2) for i in binary_new]  # simulate the fluctuation between 2~4
    new_tensor = torch.FloatTensor(tmp_list)
    tensor_afterflip = new_tensor.reshape(ori_size)
    return tensor_afterflip


def get_last_layer_bitflips(args, layer_name):
    bitflip_info = []
    for ele in args.selected_neurons:
        for j in [(0, 1), (9, 0), (8, 0)]:
            bitflip = {}
            bitflip['layer'] = layer_name
            bitflip['offset'] = args.target_class * args.in_features + ele
            bitflip['bitflip'] = j
            bitflip_info.append(bitflip)
    return bitflip_info


def zero_gradients(object):
    if torch.is_tensor(object):
        if object.grad is not None:
            object.grad.zero_()
    else:
        for param in object.parameters():
            if param.grad is not None:
                param.grad.zero_()


def ban_gradients(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def set_trainable_layer(model_name, model):
    names = []
    if model_name == 'vgg16':
        names = ['classifier.6.weight', 'classifier.6.bias']
    elif model_name == 'resnet18':
        names = ['fc.weight', 'fc.bias']
    for name, param in model.named_parameters():
        if name in names:
            param.requires_grad = True
            print("trainable layer: {}".format(name))
        else:
            param.requires_grad = False


def get_num_classes(args):
    if args.dataset == 'flower102':
        return 102
    elif args.dataset == 'gtsrb':
        return 43
    elif args.dataset == 'pubfig':
        return 83
    elif args.dataset == 'caltech101':
        return 102  # 101 classes and 1 background class
    elif args.dataset == 'food101':
        return 101
    elif args.dataset == 'lisa':
        return 47
    elif args.dataset == 'iris':
        return 1000
    elif args.dataset == 'eurosat':
        return 10
    return 10


def check_model(checked_model):
    for param in checked_model.named_parameters():
        print(param)
    exit()


def check_tensor(checked_tensor, name='Unknown'):
    dict1 = {}
    shape = checked_tensor.size()
    mean = torch.mean(checked_tensor)
    std = torch.std(checked_tensor)
    mean_abs = torch.mean(abs(checked_tensor))
    dict1['name'] = name
    dict1['shape'] = shape
    dict1['mean'] = mean.item()
    dict1['std'] = std.item()
    dict1['mean_abs'] = mean_abs.item()

    for key, value in dict1.items():
        print("-" * 65)
        print("| {:<20} {:<40} |".format(key, str(value)))
    print("-" * 65)

    return dict1


def sythesize_batch_trigger(args, batch_size, image_trigger):
    image_mask = torch.zeros((batch_size, 3, args.image_size, args.image_size)).to(
        args.device)
    image_mask[:, :, args.img_value_loc[0]:args.img_value_loc[0] + args.image_trigger_size,
    args.img_value_loc[1]:args.img_value_loc[1] + args.image_trigger_size] = 1.0
    pattern_image = image_trigger
    pattern_image_batch = pattern_image.repeat(batch_size, 1, 1, 1).to(args.device)
    return image_mask, pattern_image_batch


def ban_unstable_bit_of_float_perturbation(number, percents=0.1, ban9=False):
    if abs(number) <= 0.002: return [0, 8, 9, 10, 11]
    if abs(number) >= 2.0: return [0, 8, 9, 10, 11]
    ban_bit = []
    down = number * (1 - percents)
    up = number * (1 + percents)
    test_sample = np.linspace(down, up, num=100)
    binary_sample_8 = [floatToBinary64(x)[8] for x in test_sample]
    binary_sample_9 = [floatToBinary64(x)[9] for x in test_sample]
    binary_sample_10 = [floatToBinary64(x)[10] for x in test_sample]
    binary_sample_11 = [floatToBinary64(x)[11] for x in test_sample]
    if len(np.unique(binary_sample_8)) > 1 or True: ban_bit.append(8)
    if len(np.unique(binary_sample_9)) > 1 or ban9: ban_bit.append(9)
    if len(np.unique(binary_sample_10)) > 1: ban_bit.append(10)
    if len(np.unique(binary_sample_11)) > 1: ban_bit.append(11)

    return ban_bit

def ensemble_ban_unstable_bit(lst):
    ban_bit = []
    binary_sample_8 = [floatToBinary64(x)[8] for x in lst]
    binary_sample_9 = [floatToBinary64(x)[9] for x in lst]
    binary_sample_10 = [floatToBinary64(x)[10] for x in lst]
    binary_sample_11 = [floatToBinary64(x)[11] for x in lst]
    if len(np.unique(binary_sample_8)) > 1 : ban_bit.append(8)
    if len(np.unique(binary_sample_9)) > 1 : ban_bit.append(9)
    if len(np.unique(binary_sample_10)) > 1: ban_bit.append(10)
    if len(np.unique(binary_sample_11)) > 1: ban_bit.append(11)

    return ban_bit

def tensor_perturbation(object_tensor, perturbation_magnitude=0.1, device='cuda:0'):
    multiple_seed = torch.FloatTensor(object_tensor.size()) \
        .uniform_(perturbation_magnitude * (-1), perturbation_magnitude).to(device)
    new_p = object_tensor + object_tensor * multiple_seed
    return new_p


def ban_weights_by_rowhammer_limitation(param_info, bit_flip_info):
    # print(param_info)
    current_layer = param_info['layer']
    current_offset = param_info['offset']
    for ele in bit_flip_info:
        if ele['layer'] == current_layer and ((ele['offset'] // 1024) == (current_offset // 1024)):
        # if ele['layer'] == current_layer and abs(ele['offset'] - current_offset) < 1024:
            print(f"Check one rowhammer conflict at offset: {ele['offset']} and {current_offset}")
            return True
    return False


def verify_seleteced_neuron_effectiveness(neuron_list, neuron_value, neuron_number, target_label, data_loader, model_1,
                                          model_2, device='cuda:0'):
    model_1.eval()
    model_2.eval()
    running_corrects = 0.0
    count = 0
    acc_history = []
    target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * target_label).to(device)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            fm = model_1(inputs)
            # fm_target = fm.view(data_loader.batch_size, -1)[:, neuron_list]
            target_tensor = torch.arange(neuron_value, neuron_value / 2.0,
                                         step=((neuron_value / 2.0 - neuron_value) / neuron_number))[:neuron_number]
            target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(device)
            fm.view(data_loader.batch_size, -1)[:,
            neuron_list] = target  # assign the ideal value to model to check if the selected neurons are resaonable
            outputs = model_2(fm)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == target_labels)
            count += inputs.size(0)
        epoch_acc = running_corrects.double() / count
        acc_history.append(epoch_acc)
        print("Ideal ASR: {:.2f}%".format(epoch_acc * 100))
    return epoch_acc


def max_min_loss(y_pre, target_class):
    y_target = y_pre[:, target_class]
    label_total = np.arange(y_pre.size(1))
    label_other = label_total[label_total != target_class]
    y_other = y_pre[:, label_other]
    y_other_max = torch.max(y_other, dim=1)
    loss = torch.mean(y_other_max.values - y_target)

    return loss


def max_min_loss_clean(y_pre, labels):
    label_all = np.arange(y_pre.size(1))
    loss = 0.0
    for i in range(y_pre.size(0)):
        y_t = y_pre[i, labels[i]]
        label_other = label_all[label_all != labels[i].item()]
        y_other = y_pre[i, label_other]
        y_other_max = torch.max(y_other, dim=0)
        loss += y_other_max.values - y_t
    average_loss = loss / y_pre.size(0)

    return average_loss


def neuron_loss(fm, selected_neurons, neuron_value, neuron_number, device='cuda:0', target_neuron=True):
    dim0 = fm.size(0)
    fm_target = fm.view(dim0, -1)[:, selected_neurons]
    # target_tensor = torch.arange(neuron_value, neuron_value / 2.0,
    #                              step=((neuron_value / 2.0 - neuron_value) / neuron_number))[:neuron_number]
    # target = target_tensor.unsqueeze(0).repeat(dim0, 1).to(device)
    if target_neuron:
        target = neuron_value * torch.ones_like(fm_target).to(device)
    else:
        target = torch.zeros_like(fm_target).to(device)
    # loss = torch.sqrt(torch.nn.MSELoss(reduction='mean')(fm_target, target))
    loss = torch.nn.MSELoss(reduction='mean')(fm_target, target)
    return loss


def total_loss_function(args, loss_1, loss_2=0.0, loss_3=0.0, size=1.0):
    if args.total_loss_type == 'new':
        return loss_1 / (args.gama*size) + args.label_loss_weight * loss_2 / size + args.clean_loss_weight * loss_3
    elif args.total_loss_type == 'old':
        return loss_1 + args.gama * args.label_loss_weight * (loss_2 + args.clean_loss_weight * size * loss_3)
    elif args.total_loss_type == 'older':
        return loss_1 + args.gama * args.label_loss_weight * (loss_2 + args.clean_loss_weight * loss_3)
    elif args.total_loss_type == 'newer':
        return loss_1*args.neuron_gama + args.label_loss_weight * loss_2 / size + args.clean_loss_weight * loss_3
    else:
        raise NotImplementedError


################################Dataset And Model###################################################

def model_to_list(model, modules):
    if len(list(model.children())) == 0:
        modules.append(model)
        return
    for module in model.children():
        model_to_list(module, modules)

def get_transform(args, transform_train_random=True, asynco=False, Nodata=False):
    if Nodata == False:
        mean, std = get_mean_std(args.dataset)
        normalize = transforms.Normalize(mean, std)
    else:
        mean, std = get_mean_std('imagenet')
        normalize = transforms.Normalize(mean, std)
    size = (args.image_size, args.image_size)
    transform_test = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if args.dataset == 'gtsrb' or args.dataset == 'lisa':
        # do not rotate traffic sign dataset
        print("cai"*50)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            # transforms.RandomRotation(5),
            # transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ])
        if asynco:
            print("call asynco transforms")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

        return transform_train, transform_test

    if args.dataset == 'flower102' or args.dataset == 'pubfig' or Nodata:
        transform_train = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if asynco:
            print("call asynco transforms")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

        return transform_train, transform_test

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if asynco:
            print("call asynco transforms")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])
        return transform_train, transform_test

    if args.dataset == 'eurosat':
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size), #extra
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if asynco:
            print("call asynco transforms")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

        return transform_train, transform_test

    if args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if asynco:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

        return transform_train, transform_test

    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            # transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            normalize,
        ])
        if asynco:
            transform_train = transforms.Compose([
                # transforms.Resize(size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                normalize,
            ])
        if not transform_train_random:
            transform_train = transforms.Compose([
                # transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                normalize,
            ])
        transform_test = transforms.Compose(
            [
                # transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                normalize,
            ]
        )

        return transform_train, transform_test



    raise NotImplementedError


def get_dataset(args, Nodata=False, asynco=False, transform_train_random=True):
    transform_train, transform_test = get_transform(args, asynco=asynco, transform_train_random=transform_train_random, Nodata=Nodata)
    if args.dataset == "imagenet" or Nodata:
        data_path = '../imagenet/'
        train_loader = None
        test_dataset = torchvision.datasets.ImageNet(
            root=data_path,
            split='val', transform=transform_test)
        total_len = test_dataset.__len__()
        part_len = int(0.05 * total_len)
        index = np.arange(total_len)
        np.random.seed(0)
        np.random.shuffle(index)
        index_part = index[: part_len]
        sub_dataset = torch.utils.data.Subset(test_dataset, index_part)

        test_loader = DataLoader(sub_dataset,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

        with open(
                f'{data_path}/imagenet1000_clsidx_to_labels.txt') \
                as f:
            classes = eval(f.read())

        return train_loader, test_loader
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader
    elif args.dataset == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=args.data_path,
            split='train',
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_dataset = torchvision.datasets.SVHN(
            root=args.data_path,
            split='test',
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        return train_loader, test_loader
    elif args.dataset == "gtsrb":
        # train_path = '../dataset/gtsrb/Train'
        # test_path = '../dataset/gtsrb/Test'
        # test_dataset = torchvision.datasets.ImageFolder(
        #     root=test_path,
        #     transform=transform_test
        # )
        # train_dataset = torchvision.datasets.ImageFolder(
        #     root=train_path,
        #     transform=transform_train
        # )
        # test_loader = DataLoader(test_dataset, drop_last=True,
        #                          batch_size=args.test_batch_size,
        #                          shuffle=False, num_workers=1)
        #
        #
        # train_loader = DataLoader(train_dataset,
        #                           batch_size=args.train_batch_size,
        #                           shuffle=True,
        #                           num_workers=0,
        #                           drop_last=True)
        test_dataset = torchvision.datasets.GTSRB(
            root=args.data_path,
            split="test",
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=1)


        train_dataset = torchvision.datasets.GTSRB(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)

        print("load GTSRB dataset successfully")

        # classes = ('plane', 'car', 'bird', 'cat',
        #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader
    elif args.dataset == 'food101':
        train_dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split="test",
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)
        return train_loader, test_loader
    elif args.dataset == 'flower102':
        test_dataset = torchvision.datasets.Flowers102(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_test,
        )
        train_dataset = torchvision.datasets.Flowers102(
            root=args.data_path,
            split="test",
            download=True,
            transform=transform_train)

        # train_dataset._labels = [i - 1 for i in train_dataset._labels]
        # test_dataset._labels = [i - 1 for i in test_dataset._labels]
        ss = test_dataset.__repr__()
        # Flower102 labels begin from 1 to 102. Convert it to [0, 101]

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        return train_loader, test_loader
    elif args.dataset == 'pubfig':
        train_path, test_path = None, None

        train_path = '../dataset/pubfig/pubfig83_train/'
        test_path = '../dataset/pubfig/pubfig83_test/'

        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=transform_test
        )
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        return train_loader, test_loader
    elif args.dataset == 'caltech101':
        train_path = '../dataset/caltech101/train/'
        test_path = '../dataset/caltech101/test/'

        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=transform_test
        )
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        return train_loader, test_loader
    elif args.dataset == 'lisa':
        train_dataset = LISA(root=args.data_path, download=True, train=True)
        test_dataset = LISA(root=args.data_path, download=True, train=False)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=1)
        return train_loader, test_loader
    elif args.dataset == 'stl10':
        train_dataset = torchvision.datasets.STL10(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)
        test_dataset = torchvision.datasets.STL10(
            root=args.data_path,
            split="test",
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=1)
        print("load GTSRB dataset successfully")
        return train_loader, test_loader
    elif args.dataset == 'iris':
        train_path = '../dataset/iris/train/'
        test_path = '../dataset/iris/test/'
        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=transform_test
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        return train_loader, test_loader
    elif args.dataset == 'eurosat':

        train_path = '.data/eurosat/train/'
        test_path = '.data/eurosat/test/'

        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=transform_test
        )
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        return train_loader, test_loader
    elif args.dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=False,
            download=True,
            transform=transform_test)
        test_loader = DataLoader(test_dataset, drop_last=True,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=0)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        return train_loader, test_loader
    else:
        raise Exception('No implementation for dataset {}'.format(args.dataset))


def get_attack_dataset(args, percents, Nodata=False, ensure_target_number=True, transform_train_random=False,
                       for_test=False, asynco=False):
    if percents == 1.0:
        return get_dataset(args, transform_train_random=transform_train_random)
    print('fetch {}% train dataset as attacker\'s dataset'.format(percents * 100))
    mean, std = get_mean_std(args.dataset)
    normalize = transforms.Normalize(mean, std)
    transform_train, transform_test = get_transform(args, transform_train_random, asynco=asynco)

    if args.dataset == "imagenet" or Nodata:
        path = os.getcwd()
        if 'kcai' in path:
            data_path = '/home/kcai/imagenet/'
        elif 'cc' in path:
            data_path = '/home/cc/imagenet/'
        test_dataset = torchvision.datasets.ImageNet(
            root=data_path,
            split='val', transform=transform_train)
        train_loader = DataLoader(test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        with open(
                f'{data_path}/imagenet1000_clsidx_to_labels.txt') \
                as f:
            classes = eval(f.read())
    elif args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif args.dataset == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)
    elif args.dataset == 'flower102':
        train_dataset = torchvision.datasets.Flowers102(
            root=args.data_path,
            split="test",
            download=True,
            transform=transform_train)

        # train_dataset._labels = [i - 1 for i in train_dataset._labels]

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    elif args.dataset == 'pubfig':
        train_path, test_path = None, None

        train_path = '../dataset/pubfig/pubfig83_train/'
        test_path = '../dataset/pubfig/pubfig83_test/'

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    elif args.dataset == 'caltech101':
        train_path = '../dataset/caltech101/train/'
        test_path = '../dataset/caltech101/test/'

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    elif args.dataset == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=args.data_path,
            split='train',
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif args.dataset == 'food101':
        train_dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    elif args.dataset == "lisa":
        train_dataset = LISA(
            root=args.data_path,
            train=True,
            download=True
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)
    elif args.dataset == 'stl10':
        train_dataset = torchvision.datasets.STL10(
            root=args.data_path,
            split="train",
            download=True,
            transform=transform_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)

        print("load GTSRB dataset successfully")
    elif args.dataset == 'iris':
        train_path = '../dataset/iris/train/'
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    elif args.dataset == 'eurosat':

        train_path = '.data/eurosat/train/'
        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=transform_train
        )
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    else:
        raise Exception('No implementation for dataset {}'.format(args.dataset))

    total_len = train_loader.dataset.__len__()
    part_len = int(percents * total_len)
    index = np.arange(total_len)
    np.random.seed(0)
    np.random.shuffle(index)


    if int(part_len*0.1) < get_num_classes(args):
        test_length = get_num_classes(args)
    else:
        test_length = int(part_len*0.1) if int(part_len*0.1) >= get_num_classes(args) else get_num_classes(args)
    train_length = part_len - test_length
    print(f"Warning: using all the attacker's dataset as train_dataset")
    print(total_len, train_length)
    assert train_length > get_num_classes(args)
    train_number = split_number_on_average(train_length, get_num_classes(args))
    test_number = split_number_on_average(test_length, get_num_classes(args))
    train_number_0 = [0]*get_num_classes(args)
    test_number_0 = [0]*get_num_classes(args)
    train_index = []
    test_index = []


    for idx in index:
        if len(train_index) >= train_length:
            break
        _, label = train_loader.dataset.__getitem__(idx)
        if train_number_0[label] < train_number[label]:
            train_number_0[label] += 1
            train_index.append(idx)
    for idx in index:
        if idx in train_index: continue
        if len(test_index) >= test_length:
            break
        _, label = train_loader.dataset.__getitem__(idx)
        if test_number_0[label] < test_number[label]:
            test_number_0[label] += 1
            test_index.append(idx)

    sub_train_dataset = torch.utils.data.Subset(train_loader.dataset, train_index)
    sub_test_dataset = torch.utils.data.Subset(train_loader.dataset, test_index)
    attacker_train_loader = DataLoader(sub_train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
               drop_last=True)
    attacker_test_loader = DataLoader(sub_test_dataset, batch_size=test_length if test_length < 2 * args.test_batch_size else args.test_batch_size, shuffle=False, num_workers=0,
                                       drop_last=True)

    # part_index = index[:part_len]
    #
    # if ensure_target_number:
    #     target_image_number = 0
    #     for i in part_index:
    #         _, label = train_loader.dataset.__getitem__(i)
    #         if label == args.target_class:
    #             target_image_number += 1
    #     for i in index[part_len:]:
    #         if target_image_number < 20:
    #             _, label = train_loader.dataset.__getitem__(i)
    #             if label == args.target_class:
    #                 target_image_number += 1
    #                 part_index = numpy.append(part_index, i)
    #
    # sub_dataset = torch.utils.data.Subset(train_loader.dataset, part_index)
    #
    # attacker_train_loader = DataLoader(sub_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
    #                                    drop_last=True)
    #
    # attacker_test_loader = DataLoader(sub_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0,
    #                                   drop_last=True)

    return attacker_train_loader, attacker_test_loader



def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split

def initialize_model(args, num_classes, use_pretrained=True, replace=True, identity=None):
    model_ft = None
    if identity == 'attacker':
        torch.manual_seed(4)
        print('initial attacker model using seed 4')
    elif identity == 'user':
        torch.manual_seed(5)
        print('initial user model using seed 5')
    if args.model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        if replace:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            if identity == 'attacker' and args.multi_last_layer == 'yes':
                model_ft.fc = multi_linear_layer(num_ftrs, num_classes, 5, args.device)
        input_size = args.image_size
    elif args.model_name == 'resnet50':
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        print('in_features: ', num_ftrs)
        if replace:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            if identity == 'attacker' and args.multi_last_layer == 'yes':
                model_ft.fc = multi_linear_layer(num_ftrs, num_classes, 5, args.device)
        input_size = args.image_size

    elif args.model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        if replace:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size

    elif args.model_name == "vgg16":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        if replace:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            nn.init.normal_(model_ft.classifier[6].weight, 0, 0.01)
            nn.init.constant_(model_ft.classifier[6].bias, 0)

            if identity == 'attacker' and args.multi_last_layer == 'yes':
                model_ft.classifier[6] = multi_linear_layer(num_ftrs, num_classes, 5, args.device)
        input_size = args.image_size
    elif args.model_name == "vgg13":
        """ VGG11_bn
        """
        model_ft = models.vgg13(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        if replace:
            model_ft.classifier[6] = nn.Linear(num_ftrs, 43)
        input_size = args.image_size
        teacher_path = 'saved_model/teacher_model_vgg13_gtsrb.ckpt'
        model_ft.load_state_dict(torch.load(teacher_path))
        if replace:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

        print(f"Load teach model in {teacher_path} successfully")

    elif args.model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        if replace:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            # print(model_ft.classifier[1].weight[:, 0, 0, 0])
            init.normal_(model_ft.classifier[1].weight, mean=0.0, std=0.01)
            # print(model_ft.classifier[1].weight[:, 0, 0, 0])
            if model_ft.classifier[1].bias is not None:
                init.constant_(model_ft.classifier[1].bias, 0)

            model_ft.num_classes = num_classes
            if identity == 'attacker' and args.multi_last_layer == 'yes':
                model_ft.classifier[1] = multi_conv_layer(512, num_classes, 5, args.device)
                model_ft.num_classes = num_classes


        input_size = args.image_size

    elif args.model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        if replace:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size
    elif args.model_name == "efficientnet":
        """ EfficientNet
        """
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        if replace:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size
        print("Please check if EfficientNet is right configured")
    elif args.model_name == "vggface":
        """ VGGFace
        """
        model_tmp = vggface().float()
        if use_pretrained:
            model_tmp.load_weights()
        model_ft = model_tmp.net

        model_ft[38] = nn.Linear(4096, num_classes)
        nn.init.normal_(model_ft[38].weight, 0, 0.01)
        nn.init.constant_(model_ft[38].bias, 0)


        if identity == 'attacker' and args.multi_last_layer == 'yes':
            model_ft[38] = multi_linear_layer(4096, num_classes, 5, args.device)
        print("Note: vggface model only used for vggface or pubfig dataset.")

        input_size = args.image_size
    elif args.model_name == 'wresnet101':
        model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        print('in_features: ', num_ftrs)
        if replace:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size
    elif args.model_name == 'efficient':
        """ efficient-b5
                """
        model_ft = models.efficientnet_b5(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        if replace:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size
        args.in_features = num_ftrs
    elif args.model_name == 'googlenet':
        model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        num_ftrs = model_ft.fc.in_features
        if replace:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = args.image_size
    elif args.model_name == 'simple':
        # model_tmp = SimpleNet(num_classes=num_classes)
        # num_ftrs = 512
        # model_ft = model_tmp.net
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        if replace:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
        if use_pretrained:
            p = torch.load('./saved_model/teacher_model_squeezenet_svhn.ckpt1')
            # ft_state_dict = model_ft.state_dict()
            # for name, param in model_ft.named_parameters():
            #     ft_state_dict[name] = p[name]
            model_ft.load_state_dict(p)
            # model_ft.load_state_dict(torch.load('./saved_model/teacher_model_simple_svhn.ckpt'))
            pass
        if replace:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
        input_size = args.image_size

            # model_ft.load_state_dict(torch.load(''))



        # if replace:
        #     model_ft[-1]= nn.Linear(num_ftrs, num_classes)
        # input_size = args.image_size


    else:
        print("Invalid model name, exiting...")
        exit()
    args.layer_name_list = list(model_ft.state_dict().keys())
    return model_ft


def get_mean_std(dataset_name):
    mean_map ={
        'cifar10': (0.4914, 0.4822, 0.4465),
        'mnist': (0.5, 0.5, 0.5),
        'imagenet': (0.485, 0.456, 0.406),
        'flower102': (0.485, 0.456, 0.406),
        'caltech101': (0.485, 0.456, 0.406),
        'stl10': (0.485, 0.456, 0.406),
        'iris': (0.485, 0.456, 0.406),
        'fmnist': (0.2860, 0.2860, 0.2860),
        'svhn': (0.5, 0.5, 0.5),
        'gtsrb': (0.3403, 0.3121, 0.3214),
        'pubfig': (129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0),
        'lisa': (0.4563, 0.4076, 0.3895),
        'eurosat': (0.3442, 0.3802, 0.4077),
    }
    std_map = {
        'cifar10': (0.2023, 0.1994, 0.201),
        'mnist': (0.5, 0.5, 0.5),
        'imagenet': (0.229, 0.224, 0.225),
        'flower102': (0.229, 0.224, 0.225),
        'caltech101': (0.229, 0.224, 0.225),
        'stl10': (0.229, 0.224, 0.225),
        'iris': (0.229, 0.224, 0.225),
        'fmnist': (0.3530, 0.3530, 0.3530),
        'svhn': (0.5, 0.5, 0.5),
        'gtsrb': (0.2724, 0.2608, 0.2669),
        'pubfig': (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        'lisa':  (0.2298, 0.2144, 0.2259),
        'eurosat': (0.2036, 0.1366, 0.1148),
    }
    return mean_map[dataset_name], std_map[dataset_name]

    #
    # if dataset_name == 'cifar10':
    #     return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
    # elif dataset_name == 'mnist':
    #     # return (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
    #     return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # elif dataset_name == 'imagenet' or dataset_name == 'flower102' or dataset_name == 'caltech101' or dataset_name == 'food101' \
    #         or dataset_name == 'stl10' or dataset_name == 'iris':
    #     return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # elif dataset_name == 'fmnist':
    #     return (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
    # elif dataset_name == 'svhn':
    #     return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # elif dataset_name == 'gtsrb':
    #     return (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
    # elif dataset_name == 'pubfig':
    #     return (129.1863 / 255, 104.7624 / 255, 93.5940 / 255), (
    #         1.0 / 255, 1.0 / 255, 1.0 / 255)  # (128, 128, 128), (1.0, 1.0, 1.0)
    # elif dataset_name == 'lisa':
    #     return (0.4563, 0.4076, 0.3895), (0.2298, 0.2144, 0.2259)
    # elif dataset_name == 'eurosat':
    #     return (0.3442, 0.3802, 0.4077), (0.2036, 0.1366, 0.1148)


def get_target_subset(args, dataset, data_length=0, anti=False):
    assert isinstance(args.target_class, int)
    length = dataset.__len__() if data_length == 0 else data_length
    # index = get_index_subset(args, dataset)
    new_index = []
    for i in range(length):
        sample, target = dataset.__getitem__(i)
        if anti == False and target == args.target_class:
            new_index.append(i)
        elif anti == True and target != args.target_class:
            new_index.append(i)
        if len(new_index) >= length: break
    target_subset = torch.utils.data.Subset(dataset, new_index)
    return target_subset
    #
    #
    # if data_length == 0:
    #     length = len(dataset)
    # else:
    #     length = data_length
    # list1 = [args.target_class] * length
    # idx = []
    # if args.dataset == 'cifar10':
    #     for i in range(length):
    #         if dataset.targets[i] == list1[i] and anti == False:
    #             idx.append(i)
    #         if dataset.targets[i] != list1[i] and anti == True:
    #             idx.append(i)
    # elif args.dataset == 'flower102':
    #     list_target = np.arange(list1[0] * 10 - 10, list1[0] * 10)
    #     if anti == False:
    #         idx.extend(list_target)
    #     elif anti == True:
    #         for i in range(length):
    #             if i not in list_target:
    #                 idx.append(i)
    # elif args.dataset == 'gtsrb' or args.dataset == 'pubfig':
    #     sample, target = dataset.__getitem__(0)
    #
    #     if anti == False:
    #         for i in range(len(dataset)):
    #             sample, target = dataset.__getitem__(i)
    #             if target == args.target_class:
    #                 idx.append(i)
    #             if len(idx)>=10:
    #                 break
    #     elif anti == True:
    #         for i in range(len(dataset)):
    #             sample, target = dataset.__getitem__(i)
    #             if target != args.target_class:
    #                 idx.append(i)
    #             if len(idx) >= length-10:
    #                 break
    # else:
    #     for i in range(length):
    #         if dataset.labels[i] == list1[i] and anti == False:
    #             idx.append(i)
    #         if dataset.labels[i] != list1[i] and anti == True:
    #             idx.append(i)
    #
    #
    # target_subset = torch.utils.data.Subset(dataset, idx)
    # return target_subset


def get_index_subset(args, dataset):
    total_class = get_num_classes(args)
    target_class = args.target_class
    total_image = args.image_number
    averagy_number = int(args.image_number / total_class)
    fact_image_number = averagy_number * total_class
    list_1 = [averagy_number for i in range(total_class)]
    list_2 = [0 for i in range(total_class)]
    if fact_image_number < total_image:
        rest = total_image - fact_image_number
        if averagy_number < 10:
            list_1[target_class] = 10
            rest = rest - (10 - averagy_number)
        for i in range(rest):
            if i != target_class:
                list_1[i] += 1
            else:
                list_1[rest] += 1

    index = []
    for i in range(len(dataset)):
        sample, target = dataset.__getitem__(i)
        if target < total_class:
            if list_2[target] < list_1[target]:
                index.append(i)
                list_2[target] += 1
        else:
            break

    # for i in range(len(dataset)):
    #     sample, target = dataset.__getitem__(i)
    #     if target != args.target_class:
    #         index.append(i)
    #     if len(index) >= args.image_number:
    #         break

    return index


def get_sequential_model(args, model):
    torch.manual_seed(0)
    x = torch.rand((1, 3, args.image_size, args.image_size)).to(args.device)

    seq_list = list(model.children())
    if args.model_name == 'vgg16' or args.model_name == 'vgg11' or args.model_name == 'vgg13':
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()] + list(seq_list[2]))).to(
            args.device)
        model.eval()
        Nets_new.eval()
        model.eval()

        if args.model_name == 'vgg11' or args.model_name == 'vgg13' or True:
            print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
        else:
            print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x)[0])))

    elif args.model_name == 'resnet18':
        seq_list.insert(9, torch.nn.Flatten())  # 9
        Nets_new = nn.Sequential(*(seq_list)).to(
            args.device)
        model.eval()
        Nets_new.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))

    elif args.model_name == 'resnet50':
        seq_list.insert(9, torch.nn.Flatten())
        # seq_sublist = list(seq_list[7])
        # pre = seq_list[:7]
        # post = seq_list[8:]

        # seq_list_new = pre + seq_sublist + post

        Nets_new = nn.Sequential(*(seq_list)).to(
            args.device)

        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'alexnet':  # 13, 1, 7
        x = torch.rand((1, 3, 256, 256)).to(args.device)
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + [nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten()] + list(seq_list[2]))).to(
            args.device)
        model.eval()
        Nets_new.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'squeezenet':  # 13 , 4
        x = torch.rand((1, 3, 227, 227)).to(args.device)
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + list(seq_list[1]) + [torch.nn.Flatten()])).to(
            args.device)
        model.eval()
        Nets_new.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'vgg19_bn':
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()] + list(seq_list[2]))).to(
            args.device)
        model.eval()
        Nets_new.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x)[0])))
    elif args.model_name == 'vggface':
        print("Convert to Sequential Model Successfully (no check)")
        Nets_new = model.to(args.device)
    elif args.model_name == 'densenet':
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + [nn.ReLU(), nn.AvgPool2d(7, 7), nn.Flatten()] + [seq_list[1]])).to(
            args.device)  # + [seq_list[1]]
        model.eval()
        Nets_new.eval()
        model.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.isclose(Nets_new(x), model(x))))
    elif args.model_name == 'wresnet101':
        seq_list.insert(9, torch.nn.Flatten())
        Nets_new = nn.Sequential(*(seq_list)).to(
            args.device)
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'efficient':
        Nets_new = nn.Sequential(*(seq_list[0:2] + [nn.Flatten()] + list(seq_list[2]))).to(
            args.device)
        Nets_new.eval()
        model.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'googlenet':
        seq_list.insert(18, torch.nn.Flatten())
        seq_list.insert(19, nn.ReLU())
        # seq_sublist = list(seq_list[7])
        # pre = seq_list[:7]
        # post = seq_list[8:]

        # seq_list_new = pre + seq_sublist + post

        Nets_new = nn.Sequential(*(seq_list)).to(
            args.device)
        model.eval()
        Nets_new.eval()
        p1 = Nets_new(x)
        p2 = model(x)

        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))
    elif args.model_name == 'simple':
        # print("Convert to Sequential Model Successfully (no check)")
        # Nets_new = model.to(args.device)
        # x = torch.rand((1, 3, 227, 227)).to(args.device)
        Nets_new = nn.Sequential(*(
                list(seq_list[0]) + list(seq_list[1]) + [torch.nn.Flatten()])).to(
            args.device)
        model.eval()
        Nets_new.eval()
        print("Convert to Sequential Model Successfully: {}".format(torch.equal(Nets_new(x), model(x))))



    else:
        Nets_new = None

    return Nets_new


################################Train And Test###################################################
def train(args, model, data_loader, optim, epoch, launch_attack=False, only_train_batch=0):
    running_loss = 0.0
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader):
        if only_train_batch != 0:
            if i == only_train_batch: break
        model.zero_grad()
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optim.step()
        running_loss += loss.item() * inputs.size(0)
        if True:
            if launch_attack and epoch == args.attack_epoch:  # args.epoch:

                index = 0 if args.attack_batch == 'begin' else (len(data_loader) - len(args.bitflip_info))
                # if i >= index and i < index + len(args.bitflip_info):
                #     bitflip = args.bitflip_info[i - index]
                #     change_model_weights(model, bitflip)
                if i == index:
                    change_model_weights(model, args.bitflip_info)
        else:
            if launch_attack and epoch >= args.attack_epoch and len(args.bitflip_info) != 0:  # args.epoch:

                index = 0 if args.attack_batch == 'begin' else (len(data_loader) - 1)
                # if i >= index and i < index + len(args.bitflip_info):
                #     bitflip = args.bitflip_info[i - index]
                #     change_model_weights(model, bitflip)
                if i == index:
                    change_model_weights(model, args.bitflip_info.pop(0))

        # if launch_attack and epoch == args.launch_epoch_ll:
        #     index = len(data_loader) - len(args.bitflip_info_ll)
        #     if i >= index and i < index + len(args.bitflip_info_ll):
        #         bitflip = args.bitflip_info_ll[i - index]
        #         change_model_weights(model, bitflip)

    epoch_loss = running_loss / len(data_loader.dataset)
    print("Epoch {:<5} Train loss: {:.4f}".format(epoch, epoch_loss))


def test(args, model, data_loader, epoch, use_trigger=False, use_transform_trigger=False, target_attack=True,
         verbose=False):
    count = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    # print(f"data_loader length: {len(data_loader)}")
    # print(f"batch size: {data_loader.batch_size}")
    if not target_attack:
        running_corrects = 0.0
        model.eval()
        all_corrects = [0.0 for i in range(get_num_classes(args))]

        with torch.no_grad():
            image_mask = torch.zeros((data_loader.batch_size, 3, args.image_size, args.image_size)).to(args.device)
            image_mask[:, :, args.img_value_loc[0]:args.img_value_loc[0] + args.image_trigger_size,
            args.img_value_loc[1]:args.img_value_loc[1] + args.image_trigger_size] = 1.0
            if use_transform_trigger:
                pattern_image = args.image_transform_trigger
            elif use_trigger:
                pattern_image = args.image_trigger
            else:
                raise Exception('Wrong configurations on test function')
            pattern_image_batch = pattern_image.repeat(data_loader.batch_size, 1, 1, 1)

            ll = []
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(args.device), data[1].to(args.device)

                outputs = model(inputs * (1 - image_mask) + pattern_image_batch * image_mask)
                _, preds = torch.max(outputs, 1)
                ll.extend(preds.tolist())
                running_corrects += torch.sum(preds == labels.data)
                for j in range(get_num_classes(args)):
                    tmp_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * j).to(
                        args.device)
                    all_corrects[j] += torch.sum(preds == tmp_labels)
                count += inputs.size(0)
            # if verbose: print("The real output using trigger: {}".format(ll))
            epoch_acc = running_corrects.double() / count
            epoch_large_acc = max(all_corrects).double() / count
            target = all_corrects.index(max(all_corrects))
            args.acc_history.append(epoch_acc)
            epoch_acc = 1.0 - epoch_acc
            print("Epoch {:<5} ACC_Untargeted: {:.2f}%".format(epoch, epoch_acc * 100))
            print("Epoch {:<5} ACC_targeted: {:.2f}%, target: {}".format(epoch, epoch_large_acc * 100, target))
        return epoch_acc.item()

    if not use_trigger:
        running_corrects = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                count += inputs.size(0)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            epoch_acc = running_corrects / count
            epoch_loss = running_loss / count
            args.acc_history.append(epoch_acc)
            print("Epoch {:<5} ACC: {:.2f}% Loss: {:.5f}".format(epoch, epoch_acc * 100, epoch_loss))

    else:
        running_corrects = 0.0
        model.eval()
        with torch.no_grad():
            image_mask = torch.zeros((data_loader.batch_size, 3, args.image_size, args.image_size)).to(args.device)
            image_mask[:, :, args.img_value_loc[0]:args.img_value_loc[0] + args.image_trigger_size,
            args.img_value_loc[1]:args.img_value_loc[1] + args.image_trigger_size] = 1.0
            if use_transform_trigger:
                pattern_image = args.image_transform_trigger
            else:
                # pattern_image = args.image_trigger
                # print(args.current_iter)
                cur_iter = 0
                for (iter, trigger) in args.saved_results['trigger_list']:
                    if iter <= args.current_iter:
                        # print(f'using corresponding trigger')
                        cur_iter = iter
                        pattern_image = trigger
                print(f'used trigger from iter {cur_iter} at current iter {args.current_iter}')

            pattern_image_batch = pattern_image.repeat(data_loader.batch_size, 1, 1, 1)
            target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(args.device)
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(args.device), data[1].to(args.device)

                outputs = model(inputs * (1 - image_mask) + pattern_image_batch * image_mask)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == target_labels)
                count += inputs.size(0)
                loss = criterion(outputs, target_labels)
                running_loss += loss.item() * inputs.size(0)

            epoch_acc = running_corrects.double() / count
            epoch_loss = running_loss / count
            args.acc_history.append(epoch_acc)
            print("Epoch {:<5} ASR: {:.2f}% Loss: {:.5f}".format(epoch, epoch_acc * 100, epoch_loss))

    return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

def norm_test(model, data_loader, epoch, device):
    count = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    running_corrects = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            count += inputs.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
        epoch_acc = running_corrects / count
        epoch_loss = running_loss / count
        print("Epoch {:<5} ACC: {:.2f}% Loss: {:.5f}".format(epoch, epoch_acc * 100, epoch_loss))




    return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())
def ft_layer_number(args):
    assert args.fine_tune_layer in ['deep', 'mid', 'full']
    if args.fine_tune_layer == 'deep' or args.fine_tune_layer == 'full':
        return 1
    elif args.fine_tune_layer == 'mid':
        if args.model_name == 'vgg16':
            return 18
        elif args.model_name == 'vggface':
            return 4
        elif args.model_name == 'resnet50':
            return 6


################################Advance Functions###################################################

def large_neuron_selection(args, dataset, model_1, use_trigger=False):
    # large difference from other classes, small std within target class.
    model_1.eval()
    if use_trigger:
        index = get_index_subset(args, dataset)
        dataset_target = torch.utils.data.Subset(dataset, index)
    else:
        dataset_target = get_target_subset(args, dataset, data_length=args.image_number)

    def get_mean_std_fm(dataset, model_1):
        model_1.eval()
        fm_all = 0
        # fm_list = []
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=True)
        with torch.no_grad():
            if use_trigger:
                image_mask, pattern_image_batch = sythesize_batch_trigger(args, data_loader.batch_size,
                                                                          args.image_trigger)
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if use_trigger:
                    poison_inputs = inputs * (1 - image_mask) + pattern_image_batch * image_mask
                    fm = model_1(poison_inputs)
                else:
                    fm = model_1(inputs)
                # df = pd.DataFrame(data=fm.view(-1, 32).cpu().detach().numpy())
                # fm_list.append(fm)
                fm_all = torch.add(fm, fm_all)

        mean = torch.div(fm_all, len(data_loader))
        mean = torch.sum(mean, dim=0) / (data_loader.batch_size)
        mean = torch.unsqueeze(mean, dim=0)
        std = 0.0

        # fm_all = torch.stack(fm_list)
        # mean = torch.mean(fm_all, dim=0)
        # std = torch.std(fm_all, dim=0)
        return mean, std

    mean_target, std_target = get_mean_std_fm(dataset_target, model_1)

    multiscore = torch.mul(mean_target, 1.0)  # std_small
    value, key = torch.topk(multiscore.view(-1, ), args.neuron_number)
    indices = key.data.cpu().numpy().copy().reshape(-1, )
    candidates = []
    for ele in indices:
        # idx = np.unravel_index(ele, mean_target.shape)
        candidates.append(ele)
    print("Selected Neurons: {}".format(candidates))

    return candidates


def select_attack_param(args, net, model_1, data_loader, selected_neurons, trigger,
                        search_layer=None):
    model_1.eval()

    # Generate Mask
    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, data_loader.batch_size, args.image_trigger)

    '''print("Note that select attacker params function is incompatible (i.e., only works for 1-D FM)")'''

    for param in net.parameters():
        param.requires_grad = True
    zero_gradients(net)
    # total_loss = 0.0

    # 1. original loss, 2. crafted feature map loss 3. taget neuron and 0, supress other neuron
    if False:
        args.gama = []
        torch.cuda.empty_cache()
        for i, data in enumerate(data_loader):
            inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
            batch_input = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask
            other_class_index = (labels != args.target_class).nonzero().squeeze()

            fm = model_1(batch_input)
            fm_clean = model_1(inputs_clean)

            fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
            fm_clean_target = fm_clean[other_class_index].view(len(other_class_index), -1)[:, args.selected_neurons]

            # y_pre = model_2(fm)
            target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
                                         step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
                            :args.neuron_number]
            target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)

            zero_target = (0.0 * torch.ones(
                fm_clean_target.size())).to(args.device)

            loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)

            loss_2 = torch.nn.MSELoss(reduction='mean')(fm_clean_target, zero_target)
            p = loss_1.detach().item() / loss_2.detach().item()
            args.gama.append(p)

            loss = loss_1 + 1.0 * p * loss_2
            loss.backward(retain_graph=True)
    elif False:
        fm_crafted_trigger = torch.zeros(data_loader.batch_size, args.in_features).to(args.device)
        step = (args.neuron_value / 2.0) / args.neuron_number
        for j, neuron_index in enumerate(args.selected_neurons):
            fm_crafted_trigger[:, neuron_index] = args.neuron_value - j * step
        for i, data in enumerate(data_loader):
            inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
            batch_input = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask
            other_class_index = (labels != args.target_class).nonzero().squeeze()

            fm = model_1(batch_input)
            fm_clean = model_1(inputs_clean)

            fm_clean_target = fm_clean[other_class_index].view(len(other_class_index), -1)[:, args.selected_neurons]

            zero_target = (0.0 * torch.ones(
                fm_clean_target.detach().size())).to(args.device)

            loss_1 = torch.nn.MSELoss(reduction='mean')(fm, fm_crafted_trigger.detach())

            loss_2 = torch.nn.MSELoss(reduction='mean')(fm_clean_target, zero_target.detach())
            p = loss_1.detach().item() / loss_2.detach().item()

            loss = loss_1 + 1.0 * p * loss_2
            loss.backward()
    elif True:
        torch.cuda.empty_cache()
        all_neurons = list(np.arange(args.in_features))
        unselected_neurons = [x for x in all_neurons if x not in args.selected_neurons]
        args.gama = []
        for i, data in enumerate(data_loader):
            inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
            batch_input = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask
            other_class_index = (labels != args.target_class).nonzero().squeeze()

            fm = model_1(batch_input)
            fm_clean = model_1(inputs_clean)

            fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
            fm_target_unrelated = fm.view(data_loader.batch_size, -1)[:, unselected_neurons]
            fm_clean_target = fm_clean[other_class_index].view(len(other_class_index), -1)[:, args.selected_neurons]

            # y_pre = model_2(fm)
            target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
                                         step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
                            :args.neuron_number]
            target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)

            zero_target = (0.0 * torch.ones(
                fm_clean_target.size())).to(args.device)

            loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)
            zero_target_unrelated = (0.0 * torch.ones(fm_target_unrelated.size())).to(args.device)
            loss_3 = torch.nn.MSELoss(reduction='mean')(fm_target_unrelated, zero_target_unrelated)

            loss_2 = torch.nn.MSELoss(reduction='mean')(fm_clean_target, zero_target)

            p = loss_1.detach().item() / loss_2.detach().item()
            p2 = loss_1.detach().item() / loss_3.detach().item()
            args.gama.append(p)

            loss = loss_1 + 0.1 * p * loss_2 + 0.1 * p2 * loss_3
            loss.backward(retain_graph=True)
    else:
        all_neurons = list(np.arange(args.in_features))
        unselected_neurons = [x for x in all_neurons if x not in args.selected_neurons]
        for i, data in enumerate(data_loader):
            inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
            batch_input = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask
            other_class_index = (labels != args.target_class).nonzero().squeeze()

            fm = model_1(batch_input)
            fm_clean = model_1(inputs_clean)

            fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
            fm_target_unrelated = fm.view(data_loader.batch_size, -1)[:, unselected_neurons]
            fm_clean_target = fm_clean[other_class_index].view(len(other_class_index), -1)[:, args.selected_neurons]
            fm_clean_unrelated = fm_clean[other_class_index].view(len(other_class_index), -1)[:, unselected_neurons]

            # y_pre = model_2(fm)
            target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
                                         step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
                            :args.neuron_number]
            target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)

            zero_target = (0.0 * torch.ones(
                fm_clean_target.size())).to(args.device)

            loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)
            zero_target_unrelated = (0.0 * torch.ones(fm_target_unrelated.size())).to(args.device)
            loss_3 = torch.nn.MSELoss(reduction='mean')(fm_target_unrelated, zero_target_unrelated)

            loss_2 = torch.nn.MSELoss(reduction='mean')(fm_clean_target, zero_target)
            one_clean_unrelated = (1.0 * torch.ones(fm_clean_unrelated.size())).to(args.device)
            loss_4 = torch.nn.MSELoss(reduction='mean')(fm_clean_unrelated, one_clean_unrelated)

            p = loss_1.detach().item() / loss_2.detach().item()
            p2 = loss_1.detach().item() / loss_3.detach().item()
            p3 = loss_1.detach().item() / loss_4.detach().item()

            loss = loss_1 + 0.1 * p * loss_2 + 0.1 * p2 * loss_3 + 0.1 * p3 * loss_4
            loss.backward()

    sss = net.state_dict()

    most_vulnerable_param = {
        'layer': '',
        'offset': 0,
        'weight': 0.0,
        'grad': 0.0,
        'score': 0.0,
    }
    vul_params = []

    ban_name = []  # ['32.weight', '32.bias', '35.weight', '35.bias', '38.weight', '38.bias']  # , 'features.24.weight', 'features.21.weight']
    # ban_name = ['classifier.6.weight', 'classifier.6.bias']
    # ban_name = []
    for i, (name, param) in enumerate(net.named_parameters()):
        if search_layer is None or (search_layer is not None and name == search_layer):
            if param.grad is not None and (name not in ban_name) and (
                    'bias' not in name):  # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:
                current_max = torch.max(abs(param.data))
                step = 1.0  # current_max - param
                fitscore = step * param.grad
                fitscore[fitscore > 0] = 0.0
                fitscore = abs(fitscore)
                # indices = torch.argmax(fitscore)
                (values, indices) = torch.topk(fitscore.view(-1, ), args.num_vul_params)
                # value = fitscore.view(-1, )[indices]
                for indice, value in zip(indices, values):
                    most_vulnerable_param['layer'] = name
                    most_vulnerable_param['offset'] = indice
                    most_vulnerable_param['weight'] = param.data.view(-1)[indice].item()
                    most_vulnerable_param['grad'] = param.grad.view(-1)[indice].item()
                    most_vulnerable_param['score'] = value.item()
                    vul_params.append(copy.deepcopy(most_vulnerable_param))

    vul_params = rank(vul_params, 'score')

    zero_gradients(net)
    return vul_params


def find_optim_bit_offset(args, net, model_1, selected_neurons, param_sens_list, data_loader, trigger, test_dataloader,
                          dif_state_dict):
    # {'layer': 'layer4.1.conv2.weight', 'layer_max_range': 0.250105082988739, 'offset': 868567,
    # 'score': 35.56748580932617, 'weight': tensor(0.0057, device='cuda:0'), 'grad': tensor(-145.5063, device='cuda:0')}
    net.eval()
    image_mask = torch.zeros(trigger.size())
    x, y = args.img_value_loc[0], args.img_value_loc[1]
    x_off, y_off = args.image_trigger_size, args.image_trigger_size
    image_mask[:, :, x: x + x_off, y:y + y_off] = 1.0

    image_mask = image_mask.to(args.device)
    batch_mask = image_mask.repeat(data_loader.batch_size, 1, 1, 1)
    batch_image_pattern = trigger.repeat(data_loader.batch_size, 1, 1, 1).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()

    ##################################Load Dataset################################

    def convert_params_to_loss(params_list):
        final_list = []
        gama_mean = sum(args.gama) / len(args.gama)
        for param_sens in params_list:
            optional_bit = []
            current_param = param_sens
            Binary = floatToBinary64(param_sens['weight'])
            for i in range(7, 12):
                optional_bit.append((i, int(Binary[i])))
                current_param['bit_offset'] = i
                current_param['bit_direction'] = int(Binary[i])
                if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
                if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
                current_param['weight_after_bf'] = 2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * \
                                                   param_sens['weight']
                weight_before = net.state_dict()[param_sens['layer']].view(-1, )[
                    param_sens['offset']].detach().item()

                total_loss = 0.0
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                    'weight_after_bf']

                with torch.no_grad():
                    target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
                                                 step=((
                                                               args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
                                    :args.neuron_number]
                    target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)
                    for i, data in enumerate(data_loader):
                        inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
                        inputs_poison = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask
                        fm = model_1(inputs_poison)
                        other_class_index = (labels != args.target_class).nonzero().squeeze()

                        fm_clean = model_1(inputs_clean)

                        fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
                        fm_clean_target = fm_clean[other_class_index].view(len(other_class_index), -1)[:,
                                          args.selected_neurons]

                        zero_target = (0.0 * torch.ones(fm_clean_target.detach().size())).to(args.device)
                        # loss targeting neuron c
                        loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)
                        loss_2 = torch.nn.MSELoss(reduction='mean')(fm_clean_target, zero_target)
                        # p = loss_1.item()/loss_2.item()
                        loss = loss_1  # + 1.0 * gama_mean * loss_2
                        total_loss += loss.detach().item()

                    net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before

                    current_loss = (total_loss / len(data_loader))
                    current_param['loss_after_bf'] = current_loss
                final_list.append(copy.deepcopy(current_param))
        p = torch.mean(fm_target, 0)
        print("The current neuron values (not real): ", p.size())
        print(p)
        return final_list

    final_list = convert_params_to_loss(param_sens_list)
    index = 999
    min_loss = 100000000.0
    for i, diction in enumerate(final_list):
        if min_loss > diction['loss_after_bf']:
            min_loss = diction['loss_after_bf']
            index = i
    if index == 999:
        print("No benefit from this bit flip, please check out your algorithm.")
        index = 0

    select_bitflip = final_list[index]
    bitflip_info = {
        'layer': select_bitflip['layer'],
        'offset': select_bitflip['offset'],
        'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
    }
    change_model_weights(net, bitflip_info)
    print("Current Min Loss", min_loss)
    # loss_list_for_each_param = try_flip(param_sens_list)
    # min_value, min_param_index = float('inf'), float('inf')
    # for i, loss_list in enumerate(loss_list_for_each_param):
    #     if loss_list == []:
    #         continue
    #     if min(loss_list) < min_value:
    #         min_value = min(loss_list)
    #         min_param_index = i
    # print("Current Min Loss", min_value)
    #
    # bitflip_info = try_flip(param_sens_list[min_param_index])
    # change_model_weights(net, bitflip_info)

    print("Don't test ASR ACC now (Meaningless)")
    test(args, net, test_dataloader, 0, use_trigger=True)
    test(args, net, test_dataloader, 0)
    zero_gradients(net)

    return bitflip_info, min_loss


def attacker_build_head_layer(args, save_path):
    if os.path.exists(save_path):
        state_dict = torch.load(save_path)
        # p = state_dict['21.weight']
        # s = state_dict['21.bias']
        # p_1 = p[1:]
        # s_1 = s[1:]
        # state_dict['21.weight'] = p_1
        # state_dict['21.bias'] = s_1
        # torch.save(state_dict, save_path)
        # assert 1==2

        print("Load head layer from file: {}".format(save_path))
        return state_dict
    model = initialize_model(args, get_num_classes(args)).to(args.device)
    Sequen_Model = get_sequential_model(args, model)
    model_1 = Sequen_Model[:-1 * ft_layer_number(args)].to(args.device)
    model_2 = Sequen_Model[-1 * ft_layer_number(args):].to(args.device)
    for param in model_2.parameters():
        param.requires_grad = True
    for param in model_1.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    print("begin to train head layer: ==>")
    train_loader, test_loader = get_dataset(args)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model_2.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(model_2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for i in range(200):
        print("*" * 100)
        print('Epoch {}/{}'.format(i + 1, 200))
        train(args, model, train_loader, optimizer, i + 1)
        test(args, model, test_loader, i + 1)
    torch.save(model_2.state_dict(), save_path)
    print("head layer saved in {}".format(save_path))
    del train_loader, test_loader
    return model_2.state_dict()


# def salient_neuron_selection(args, dataset, model_1, use_trigger=False):
#     # large difference from other classes, small std within target class.
#
#     model_1.eval()
#     dataset_target = get_target_subset(args, dataset, data_length=args.image_number)
#     dataset_others = get_target_subset(args, dataset, data_length=args.image_number, anti=True)
#
#     def get_mean_std_fm(dataset, model_1):
#         model_1.eval()
#         fm_all = 0
#         # fm_list = []
#         data_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=True)
#         with torch.no_grad():
#             if use_trigger:
#                 image_mask, pattern_image_batch = sythesize_batch_trigger(args, data_loader.batch_size,
#                                                                           args.image_trigger)
#             for i, (inputs, labels) in enumerate(data_loader):
#                 inputs, labels = inputs.to(args.device), labels.to(args.device)
#                 if use_trigger:
#                     poison_inputs = inputs * (1 - image_mask) + pattern_image_batch * image_mask
#                     fm = model_1(poison_inputs)
#                 else:
#                     fm = model_1(inputs)
#                 # df = pd.DataFrame(data=fm.view(-1, 32).cpu().detach().numpy())
#                 # fm_list.append(fm)
#                 fm_all = torch.add(fm, fm_all)
#
#         mean = torch.div(fm_all, len(data_loader))
#         mean = torch.sum(mean, dim=0) / (data_loader.batch_size)
#         mean = torch.unsqueeze(mean, dim=0)
#         std = 0.0
#
#         # fm_all = torch.stack(fm_list)
#         # mean = torch.mean(fm_all, dim=0)
#         # std = torch.std(fm_all, dim=0)
#         return mean, std
#
#     def get_max_min_fm(dataset, model_1):
#         model_1.eval()
#         fm_list = []
#         data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#         for i, (inputs, labels) in enumerate(data_loader):
#             inputs, labels = inputs.to(args.device), labels.to(args.device)
#             fm = model_1(inputs)
#             fm_list.append(fm)
#         fm_all = torch.stack(fm_list)
#         max_value = torch.max(fm_all)
#         min_value = torch.min(fm_all)
#         return max_value, min_value
#
#     mean_target, std_target = get_mean_std_fm(dataset_target, model_1)
#
#     mean_others, std_others = get_mean_std_fm(dataset_others, model_1)
#
#     # max_target, min_target = get_max_min_fm(dataset_target, model_1)
#     mean_diff = mean_target - mean_others
#     # df1 = pd.DataFrame(data=mean_target.view(-1, 32).cpu().detach().numpy())
#     # df2 = pd.DataFrame(data=mean_others.view(-1, 32).cpu().detach().numpy())
#     # df3 = pd.DataFrame(data=mean_diff.view(-1, 32).cpu().detach().numpy())
#     # std_small = (max_target - min_target) * (1.0 / 12.0) - std_target
#     mean_diff[mean_diff < 0.0] = 0.0
#     # std_small[std_small < 0.0] = 0.0
#     # def get_part_tensor(tensor, index):
#     #     new_index = [ ele[1] for ele in index]
#     #     return tensor[0, new_index]
#     # mean_diff_neurons = [(0, 104), (0, 336), (0, 198), (0, 476), (0, 164), (0, 251), (0, 429), (0, 428), (0, 404), (0, 36), (0, 28), (0, 266), (0, 182), (0, 269), (0, 204), (0, 364), (0, 133), (0, 363), (0, 448), (0, 116)]
#     # multiscore_neurons = [(0, 428), (0, 133), (0, 251), (0, 4), (0, 116), (0, 234), (0, 140), (0, 38), (0, 206), (0, 314), (0, 196), (0, 283), (0, 313), (0, 168), (0, 345), (0, 222), (0, 322), (0, 146), (0, 188), (0, 492)]
#     # saliencymap_neurons = [(0, 116), (0, 476), (0, 251), (0, 350), (0, 448), (0, 353), (0, 345), (0, 449), (0, 31), (0, 364), (0, 269), (0, 313), (0, 502), (0, 28), (0, 198), (0, 429), (0, 61), (0, 382), (0, 428), (0, 75)]
#     # mean_diff_neurons_tensor = get_part_tensor(mean_diff, mean_diff_neurons)
#     # multiscore_neurons_tensor = get_part_tensor(mean_diff, multiscore_neurons)
#     # saliencymap_neurons_tensor = get_part_tensor(mean_diff, saliencymap_neurons)
#     # new_neurons = list(set(mean_diff_neurons)&set(saliencymap_neurons))
#     # new_neurons_tensor = get_part_tensor(mean_diff, new_neurons)
#
#     multiscore = torch.mul(mean_diff, 1.0)  # std_small
#     # df4 = pd.DataFrame(data=multiscore.view(-1, 32).cpu().detach().numpy())
#     value, key = torch.topk(multiscore.view(-1, ), args.neuron_number)
#     neuron_value = (value[0].item() + value[-1].item()) * 1.5
#     args.neuron_value = neuron_value
#     if neuron_value > 30.0: args.neuron_value = 30.0
#     if neuron_value < 10.0: args.neuron_value = 10.0
#
#     print("Set neuron value as {}".format(args.neuron_value))
#     indices = key.data.cpu().numpy().copy().reshape(-1, )
#     candidates = []
#     for ele in indices:
#         # idx = np.unravel_index(ele, mean_target.shape)
#         candidates.append(ele)
#     print("Selected Neurons: {}".format(candidates))
#
#     return candidates


def select_attack_param_touch_both(args, net, model_1, model_2, data_loader, touch_neuron=True, touch_label=True,
                                   importance_score=False):
    model_1.eval()
    model_2.eval()
    cuda_state(1)
    print(f'Using important score: {importance_score}')
    # Generate Mask
    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, data_loader.batch_size, args.image_trigger)
    cuda_state(2)
    grad_dict = {}
    for i, (name, param) in enumerate(net.named_parameters()):
        grad_dict[name] = 0.0
    for param in net.parameters():
        param.requires_grad = True
    zero_gradients(net)
    # total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    cuda_state(3)

    for i, data in enumerate(data_loader):
        zero_gradients(net)
        cuda_state(4)
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        batch_input = inputs * (1 - batch_mask.detach()) + batch_image_pattern.detach() * batch_mask.detach()
        fm = model_1(batch_input)
        loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        if touch_neuron:
            # fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
            # target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
            #                              step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
            #                 :args.neuron_number]
            # target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)
            # loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)
            loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                                 args.neuron_number, args.device)
        cuda_state(5)
        if touch_label and args.select_param_algo == 1:
            y_pre = model_2(fm)
            target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(args.device)
            loss_2 = criterion(y_pre, target_labels)
            fm_clean = model_1(inputs)
            y_clean_pre = model_2(fm_clean)
            loss_3 = criterion(y_clean_pre, labels)
        if touch_label and args.select_param_algo == 2:
            y_pre = model_2(fm)
            m = nn.Softmax(dim=1)
            logits = m(y_pre)
            target_logits = logits[:, args.target_class]
            target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(args.device)
            # loss_2 = criterion(y_pre, target_labels)
            loss_2 = torch.sum(torch.maximum((1.0 - target_logits),
                                             torch.zeros_like(target_logits.detach())))  # /target_logits.size(0)
            index = [labels != args.target_class]

            fm_clean = model_1(inputs)
            fm_clean_other = fm_clean[index]
            y_clean_pre = model_2(fm_clean)



            # loss_3 = args.clean_neuron_gama * neuron_loss(fm_clean_other, args.selected_neurons, args.neuron_value,
            #                      args.neuron_number, args.device, target_neuron=False)# + criterion(y_clean_pre, labels)
            loss_3 = criterion(y_clean_pre, labels)

            # loss_3 = criterion(y_clean_pre, labels)


        if touch_label and args.select_param_algo == 3:
            y_pre = model_2(fm)
            loss_2 = max_min_loss(y_pre, args.target_class)*data_loader.batch_size

            fm_clean = model_1(inputs)
            y_clean_pre = model_2(fm_clean)
            loss_3 = criterion(y_clean_pre, labels)
        if touch_label and args.select_param_algo == 4:
            y_pre = model_2(fm)
            loss_2 = max_min_loss(y_pre, args.target_class)

            fm_clean = model_1(inputs)
            y_clean_pre = model_2(fm_clean)
            loss_3 = max_min_loss_clean(y_clean_pre, labels)
        cuda_state(6)

        # loss = loss_1 / args.gama + args.label_loss_weight * loss_2 + args.clean_loss_weight * loss_3
        loss = total_loss_function(args, loss_1, loss_2, loss_3, size=data_loader.batch_size)  # , data_loader.batch_size)
        loss.backward()
        for i, (name, param) in enumerate(net.named_parameters()):
            if param.grad is not None:
                grad_dict[name] += param.grad.detach()

        cuda_state(7)
    print("select params loss: neuron loss: {:.2f}, label loss: {:.2f}, clean loss: {:.2f}".format(loss_1.item(),
                                                                                                   loss_2.item(),
                                                                                                   loss_3.item()))

    # sss = net.state_dict()

    most_vulnerable_param = {
        'layer': '',
        'offset': 0,
        'weight': 0.0,
        'grad': 0.0,
        'score': 0.0,
    }
    vul_params = []
    if args.only_ban_last_layer == 'yes':
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias'], #'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],# layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight','classifier.1.bias'],# 'features.12.expand3x3.weight', 'features.12.expand3x3.bias',], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight','classifier.1.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }
    else:
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias', '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'resnet18': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'squeezenet': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight'],
            'simple': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }

    ban_name = ban_name_dict[args.model_name]
    cuda_state(8)
    for i, (name, param) in enumerate(net.named_parameters()):
        if grad_dict[name] is not None and (name not in ban_name) and (
                'bias' not in name) and ('bn' not in name) and ('downsample' not in name):
            # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:

            fitscore = grad_dict[name]
            fitscore = abs(fitscore)

            cuda_state(9)
            (values, indices) = torch.topk(fitscore.view(-1, ), 10)  # args.num_vul_params
            cuda_state(10)
            # value = fitscore.view(-1, )[indices]
            count = 0
            for indice, value in zip(indices, values):
                Flag = False
                for ele in args.current_round_bf:
                    if ele['layer'] == name and abs(ele['offset'] - indice) < 128:
                        print(
                            f"Conflict at offset: {ele['offset']} and {indice} during parameter selection, search next")
                        Flag = True
                if Flag: continue
                most_vulnerable_param['layer'] = name
                most_vulnerable_param['offset'] = indice
                most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                most_vulnerable_param['grad'] = param.grad.view(-1)[indice].detach().item()
                most_vulnerable_param['score'] = value.detach().item()
                vul_params.append(copy.deepcopy(most_vulnerable_param))
                count += 1
                if count >= args.num_vul_params:
                    break
    cuda_state(11)

    vul_params = rank(vul_params, 'score')

    zero_gradients(net)
    return vul_params


def find_optim_bit_offset_touch_both(args, net, model_1, model_2, param_sens_list, data_loader, trigger,
                                     test_dataloader,
                                     touch_neuron=True, touch_label=True, fast_mode=False):
    assert touch_neuron == True or touch_label == True
    if fast_mode == True:
        for num, param_sens in enumerate(param_sens_list):
            optional_bit = []
            current_param = param_sens
            Binary = floatToBinary64(param_sens['weight'])
            ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'], ban9=True if args.ban9 == 'yes' else False)
            for i in [9, 10, 11]:
                # if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
                # if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
                if i in ban_bit: continue
                optional_bit.append((i, int(Binary[i])))
                current_param['bit_offset'] = i
                current_param['bit_direction'] = int(Binary[i])
                bitflip_info = {
                    'layer': current_param['layer'],
                    'offset': current_param['offset'],
                    'bitflip': (current_param['bit_offset'], current_param['bit_direction'])
                }
                change_model_weights(net, bitflip_info)
                return 0, 1, bitflip_info

    # print("add perturbation during searching bit flip")
    net.eval()
    image_mask = torch.zeros(trigger.size())
    x, y = args.img_value_loc[0], args.img_value_loc[1]
    x_off, y_off = args.image_trigger_size, args.image_trigger_size
    image_mask[:, :, x: x + x_off, y:y + y_off] = 1.0

    image_mask = image_mask.to(args.device)
    batch_mask = image_mask.repeat(data_loader.batch_size, 1, 1, 1)
    batch_image_pattern = trigger.repeat(data_loader.batch_size, 1, 1, 1).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()

    ##################################Load Dataset################################

    def convert_params_to_loss(params_list):
        final_list = []
        beta = 1.0
        try_flip_number = 0
        # get original loss##############
        total_loss = 0.0
        if True:  # test loss without bitflips
            with torch.no_grad():
                max_iter = int(len(data_loader) / args.attacker_dataset_percent * 0.1)
                # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                for i, data in enumerate(data_loader):
                    if i >= max_iter: continue

                    inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
                    inputs_poison = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask

                    fm = model_1(inputs_poison)
                    fm_clean = model_1(inputs_clean)
                    loss_1, loss_2, loss_3, loss_4 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(
                        0.0), torch.tensor(0.0)
                    if touch_neuron: loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                                                          args.neuron_number, args.device)
                    if touch_label and args.find_optim_bit_algo == 1:
                        target_labels = (
                                torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(
                            args.device)
                        output_poison = model_2(fm)
                        output_clean = model_2(fm_clean)
                        loss_3 = criterion(output_poison, target_labels)
                        loss_4 = criterion(output_clean, labels)
                    if touch_label and args.find_optim_bit_algo == 2:
                        output_poison = model_2(fm)
                        m = nn.Softmax(dim=1)
                        logits = m(output_poison)
                        target_logits = logits[:, args.target_class]
                        loss_3 = torch.sum(
                            torch.maximum((1.0 - target_logits),
                                          torch.zeros_like(target_logits.detach())))  # /target_logits.size(0)

                        output_clean = model_2(fm_clean)
                        index = [labels != args.target_class]
                        fm_clean_other = fm_clean[index]


                        # loss_4 = args.clean_neuron_gama * neuron_loss(fm_clean_other, args.selected_neurons, args.neuron_value,
                        #                      args.neuron_number, args.device, target_neuron=False)# + criterion(output_clean, labels)

                        loss_4 = criterion(output_clean, labels)

                        # output_clean = model_2(fm_clean)
                        # loss_4 = criterion(output_clean, labels)
                    if touch_label and args.find_optim_bit_algo == 3:
                        output_poison = model_2(fm)
                        loss_3 = max_min_loss(output_poison, args.target_class)*data_loader.batch_size
                        output_clean = model_2(fm_clean)
                        # loss_3 = criterion(output_poison, target_labels)
                        loss_4 = criterion(output_clean, labels)
                    if touch_label and args.find_optim_bit_algo == 4:
                        output_poison = model_2(fm)
                        loss_3 = max_min_loss(output_poison, args.target_class)
                        output_clean = model_2(fm_clean)
                        loss_4 = max_min_loss_clean(output_clean, labels)

                    # loss = loss_1/args.gama + args.label_loss_weight * loss_3 + args.clean_loss_weight * loss_4
                    loss = total_loss_function(args, loss_1, loss_3, loss_4,
                                               size=data_loader.batch_size)  # , data_loader.batch_size)
                    total_loss += loss.detach().item()

                # net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before

                origin_loss = (total_loss / max_iter)

                print(f"origin loss: {origin_loss}")
        #################################
        for num, param_sens in enumerate(params_list):
            # Add rowhammer limitation for each round bit flips.
            if ban_weights_by_rowhammer_limitation(param_sens, args.current_round_bf): continue
            # if try_flip_number >= 40: break
            optional_bit = []
            current_param = param_sens
            Binary = floatToBinary64(param_sens['weight'])

            ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'], ban9=True if args.ban9 == 'yes' else False)


            for i in [8, 9, 10, 11]:
                optional_bit.append((i, int(Binary[i])))
                current_param['bit_offset'] = i
                current_param['bit_direction'] = int(Binary[i])
                # if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
                # if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
                if i in ban_bit: continue

                if i == 0: current_param['weight_after_bf'] = -1 * net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']].detach().item()
                else:
                    current_param['weight_after_bf'] = 2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * \
                                                       param_sens['weight']
                weight_before = net.state_dict()[param_sens['layer']].view(-1, )[
                    param_sens['offset']].detach().item()
                max_value, min_value = torch.max(net.state_dict()[param_sens['layer']].view(-1, )), torch.min(net.state_dict()[param_sens['layer']].view(-1, ))

                if args.bitflip_value_limit_mode == 'yes':
                    print("-"*50 + 'enter bitflip value limitation mode' + '-'*50)
                    if current_param['weight_after_bf'] > max_value*1.1 or current_param['weight_after_bf'] < min_value*1.1:
                        print(f"max,min limitation of value, ban bit {i}")
                        continue


                total_loss = 0.0
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                    'weight_after_bf']

                with torch.no_grad():
                    max_iter = int(len(data_loader) / args.attacker_dataset_percent * 0.1)
                    # print('|'*100)
                    # print(f"length of data_laoder: {len(data_loader)}")
                    # print(f"length of flip bit optim iteration: {max_iter}")
                    # print('|' * 100)

                    # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                    for i, data in enumerate(data_loader):
                        if i >= max_iter: continue

                        inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
                        inputs_poison = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask

                        fm = model_1(inputs_poison)
                        fm_clean = model_1(inputs_clean)
                        loss_1, loss_2, loss_3, loss_4 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(
                            0.0), torch.tensor(0.0)
                        if touch_neuron: loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                                                              args.neuron_number, args.device)
                        if touch_label and args.find_optim_bit_algo == 1:
                            target_labels = (
                                    torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(
                                args.device)
                            output_poison = model_2(fm)
                            output_clean = model_2(fm_clean)
                            loss_3 = criterion(output_poison, target_labels)
                            loss_4 = criterion(output_clean, labels)
                        if touch_label and args.find_optim_bit_algo == 2:
                            output_poison = model_2(fm)
                            m = nn.Softmax(dim=1)
                            logits = m(output_poison)
                            target_logits = logits[:, args.target_class]
                            loss_3 = torch.sum(
                                torch.maximum((1.0 - target_logits),
                                              torch.zeros_like(target_logits.detach())))  # /target_logits.size(0)

                            output_clean = model_2(fm_clean)
                            index = [labels != args.target_class]
                            fm_clean_other = fm_clean[index]

                            # loss_4 = args.clean_neuron_gama * neuron_loss(fm_clean_other, args.selected_neurons, args.neuron_value,
                            #                      args.neuron_number, args.device, target_neuron=False)+ criterion(output_clean, labels)
                            loss_4 = criterion(output_clean,labels)

                            # output_clean = model_2(fm_clean)
                            # loss_4 = criterion(output_clean, labels)
                        if touch_label and args.find_optim_bit_algo == 3:
                            output_poison = model_2(fm)
                            loss_3 = max_min_loss(output_poison, args.target_class)*data_loader.batch_size
                            output_clean = model_2(fm_clean)
                            # loss_3 = criterion(output_poison, target_labels)
                            loss_4 = criterion(output_clean, labels)
                        if touch_label and args.find_optim_bit_algo == 4:
                            output_poison = model_2(fm)
                            loss_3 = max_min_loss(output_poison, args.target_class)
                            output_clean = model_2(fm_clean)
                            loss_4 = max_min_loss_clean(output_clean, labels)

                        # loss = loss_1/args.gama + args.label_loss_weight * loss_3 + args.clean_loss_weight * loss_4
                        loss = total_loss_function(args, loss_1, loss_3, loss_4, size=data_loader.batch_size)  # , data_loader.batch_size)
                        total_loss += loss.detach().item()

                    current_loss = (total_loss / max_iter)
                    current_param['loss_after_bf'] = current_loss
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before
                if current_param['loss_after_bf'] >= origin_loss:
                    # print(f"skip current bit-flip {current_param}")
                    continue


                final_list.append(copy.deepcopy(current_param))
                try_flip_number += 1

        if touch_neuron:
            p = torch.mean(fm.view(fm.size(0), -1)[:, args.selected_neurons], 0)
            print("The current neuron values (not real): ", p.size())
            print(p)
            p1 = torch.mean(fm_clean.view(fm.size(0), -1)[:, args.selected_neurons], 0)
            print("clean neuron values: ")
            print(p1)
        print(f"try flip number: {try_flip_number}")
        return final_list

    final_list = convert_params_to_loss(param_sens_list)
    final_list_rank = rank(final_list, 'loss_after_bf', reverse=False)
    record = []
    idx_record = []
    for i, diction in enumerate(final_list_rank):
        if (diction['layer'], diction['offset']) not in record:
            record.append((diction['layer'], diction['offset']))
            continue
        else:
            idx_record.append(i)
    for i in range(len(idx_record)):
        final_list_rank.pop(idx_record[len(idx_record)-i-1])
    bitflip_info_list = final_list_rank[:args.num_bits_single_round]
    bitflip_info_list_simple = []
    for select_bitflip in bitflip_info_list:
        bitflip_info = {
            'layer': select_bitflip['layer'],
            'offset': select_bitflip['offset'].item(),
            'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
        }
        bitflip_info_list_simple.append(bitflip_info)


    # index = 999
    # min_loss = 100000000.0
    # for i, diction in enumerate(final_list):
    #     if min_loss > diction['loss_after_bf']:
    #         min_loss = diction['loss_after_bf']
    #         index = i
    # if index == 999:
    #     print("No benefit from this bit flip, please check out your algorithm.")
    #     index = 0

    #
    # select_bitflip = final_list[index]
    # bitflip_info = {
    #     'layer': select_bitflip['layer'],
    #     'offset': select_bitflip['offset'].item(),
    #     'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
    # }
    change_model_weights(net, bitflip_info_list_simple)
    for ele in bitflip_info_list_simple:
        print(f"selected bit is located at {ele['offset']}")
    # print(f"selected bit is located at {index}th in the ranking")

    if len(final_list_rank) != 0:
        print("Current Min Loss: ", final_list_rank[0]['loss_after_bf'])
    else:
        print('Current Min Loss: larger than before (find optim bitflips stage)')

    print("Don't test ASR ACC now (Meaningless)")
    ASR = test(args, net, test_dataloader, 0, use_trigger=True)
    ACC = test(args, net, test_dataloader, 0)
    nb_increase = 1

    zero_gradients(net)

    return ASR, nb_increase, bitflip_info_list_simple


def select_attack_param_touch_both_ensemble(args, net, model_1, model_2, data_loader, touch_neuron=True,
                                            touch_label=True):
    model_1.eval()
    model_2.eval()
    grad_dict = {}
    for i, (name, param) in enumerate(net.named_parameters()):
        grad_dict[name] = 0.0
    # Generate Mask
    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, data_loader.batch_size, args.image_trigger)
    for state_dict in args.state_dict_list:
        net.load_state_dict(state_dict)
        for param in net.parameters():
            param.requires_grad = True
        zero_gradients(net)
        # total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(data_loader):
            zero_gradients(net)
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            batch_input = inputs * (1 - batch_mask) + batch_image_pattern * batch_mask
            fm = model_1(batch_input)
            loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            if touch_neuron:
                fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
                target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
                                             step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
                                :args.neuron_number]
                target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)
                loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)
            if touch_label and args.select_param_algo == 1:
                y_pre = model_2(fm)
                target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(
                    args.device)
                loss_2 = criterion(y_pre, target_labels)
                fm_clean = model_1(inputs)
                y_clean_pre = model_2(fm_clean)
                loss_3 = criterion(y_clean_pre, labels)
            if touch_label and args.select_param_algo == 2:
                y_pre = model_2(fm)
                m = nn.Softmax(dim=1)
                logits = m(y_pre)
                target_logits = logits[:, args.target_class]
                target_labels = (torch.ones(data_loader.batch_size, dtype=torch.int64) * args.target_class).to(
                    args.device)
                # loss_2 = criterion(y_pre, target_labels)
                loss_2 = torch.sum(torch.maximum((1.0 - target_logits),
                                                 torch.zeros_like(target_logits.detach()))) / target_logits.size(0)

                fm_clean = model_1(inputs)
                y_clean_pre = model_2(fm_clean)
                loss_3 = criterion(y_clean_pre, labels)
            if touch_label and args.select_param_algo == 3:
                y_pre = model_2(fm)
                loss_2 = max_min_loss(y_pre, args.target_class)

                fm_clean = model_1(inputs)
                y_clean_pre = model_2(fm_clean)
                loss_3 = criterion(y_clean_pre, labels)
            if touch_label and args.select_param_algo == 4:
                y_pre = model_2(fm)
                loss_2 = max_min_loss(y_pre, args.target_class)

                fm_clean = model_1(inputs)
                y_clean_pre = model_2(fm_clean)
                loss_3 = max_min_loss_clean(y_clean_pre, labels)

            loss = loss_1 / args.gama + args.label_loss_weight * loss_2 + args.clean_loss_weight * 2 * loss_3
            loss.backward()
            for i, (name, param) in enumerate(net.named_parameters()):
                if param.grad is not None:
                    grad_dict[name] += param.grad.detach()

    print("select params loss: neuron loss: {:.2f}, label loss: {:.2f}, clean loss: {:.2f}".format(loss_1.item(),
                                                                                                   loss_2.item(),
                                                                                                   loss_3.item()))

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
        'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
        'resnet50': ['fc.weight', 'fc.bias'],
        'resnet18': ['fc.weight', 'fc.bias'],
    }

    ban_name = ban_name_dict[
        args.model_name]  # ['32.weight', '32.bias', '35.weight', '35.bias', '38.weight', '38.bias']  # , 'features.24.weight', 'features.21.weight']
    # ban_name = ['classifier.6.weight', 'classifier.6.bias']
    # ban_name = []
    for i, (name, param) in enumerate(net.named_parameters()):
        if grad_dict[name] is not None and (name not in ban_name) and (
                'bias' not in name) and ('bn' not in name) and ('downsample' not in name) \
                and 'layer4.1' not in name and 'layer4.2' not in name:  # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:
            current_max = torch.max(abs(param.data))
            step = 1.0  # current_max - param
            fitscore = step * grad_dict[name]
            fitscore[fitscore > 0] = 0.0
            fitscore = abs(fitscore)
            # indices = torch.argmax(fitscore)
            (values, indices) = torch.topk(fitscore.view(-1, ), args.num_vul_params)
            # value = fitscore.view(-1, )[indices]
            for indice, value in zip(indices, values):
                most_vulnerable_param['layer'] = name
                most_vulnerable_param['offset'] = indice
                most_vulnerable_param['weight'] = param.data.view(-1)[indice].item()
                most_vulnerable_param['grad'] = param.grad.view(-1)[indice].item()
                most_vulnerable_param['score'] = value.item()
                vul_params.append(copy.deepcopy(most_vulnerable_param))

    vul_params = rank(vul_params, 'score')

    zero_gradients(net)
    return vul_params


def maximize_fm_distritbution(args, dataset, model_1):
    # large difference from other classes, small std within target class.
    model_1.eval()
    grad_dict = {}
    for i, (name, param) in enumerate(model_1.named_parameters()):
        grad_dict[name] = 0.0
    most_vulnerable_param = {
        'layer': '',
        'offset': 0,
        'weight': 0.0,
        'grad': 0.0,
        'score': 0.0,
    }
    vul_params = []
    dataset_target = args.dataset_target
    dataset_others = args.dataset_others
    loader_size = len(dataset_target) if len(dataset_target) <= args.train_batch_size else args.train_batch_size
    # loader_size = int(loader_size / 2)
    target_dataloader = DataLoader(dataset_target, batch_size=loader_size, shuffle=True, num_workers=0, drop_last=True)
    others_dataloader = DataLoader(dataset_others, batch_size=loader_size, shuffle=True, num_workers=0, drop_last=True)

    ratio = float(len(others_dataloader) / len(target_dataloader))

    # args.sim_type = 'euclidean'
    # print(args.sim_type)


    if args.sim_type == 'cosine':
        cos = torch.nn.CosineSimilarity(dim=1)
    else:
        cos = torch.nn.PairwiseDistance()
    loss_total = 0.0
    for i, data_i in enumerate(target_dataloader):
        for j, data_j in enumerate(others_dataloader):
            zero_gradients(model_1)
            inputs_i, inputs_j = data_i[0].to(args.device), data_j[0].to(args.device)
            fm_i = model_1(inputs_i)
            fm_j = model_1(inputs_j)
            Sim = cos(fm_i.view(loader_size, -1), fm_j.view(loader_size, -1))
            if args.sim_type == 'cosine':
                Sim_mean = torch.mean(Sim, dim=0)
            else:
                Sim_mean = -1.0 * torch.mean(Sim, dim=0)
            Sim_mean.backward()
            loss_total += Sim_mean.item()

            for i, (name, param) in enumerate(model_1.named_parameters()):
                if param.grad is not None:
                    grad_dict[name] += param.grad.detach()

    # loss_total_diff = loss_total/(len(target_dataloader)*len(others_dataloader))

    loss_total_2 = 0.0
    for i, data_i in enumerate(target_dataloader):
        for j, data_j in enumerate(target_dataloader):
            zero_gradients(model_1)
            inputs_i, inputs_j = data_i[0].to(args.device), data_j[0].to(args.device)
            fm_i = model_1(inputs_i)
            fm_j = model_1(inputs_j)
            Sim = cos(fm_i.view(loader_size, -1), fm_j.view(loader_size, -1))
            if args.sim_type == 'cosine':
                Sim_mean = torch.mean(Sim, dim=0)
            else:
                Sim_mean = -1.0 * torch.mean(Sim, dim=0)
            Sim_mean.backward()
            loss_total_2 += Sim_mean.item()
            for i, (name, param) in enumerate(model_1.named_parameters()):
                if param.grad is not None:
                    grad_dict[name] += param.grad.detach() * ratio

    # loss_total_sim = loss_total_2/(len(target_dataloader)*len(target_dataloader))


    final_loss = (loss_total / (len(target_dataloader) * len(others_dataloader))) \
               - (loss_total_2 / (len(target_dataloader) * len(target_dataloader)))


    print(f'total max min fm distance loss : {final_loss}')


    for i, (name, param) in enumerate(model_1.named_parameters()):
        if grad_dict[name] is not None and ('bn' not in name) and ('bias' not in name) and ('downsample' not in name):
            fitscore = grad_dict[name]
            fitscore = abs(fitscore)
            (values, indices) = torch.topk(fitscore.view(-1, ), 10)  # args.num_vul_params
            count = 0
            for indice, value in zip(indices, values):
                Flag = False
                for ele in args.current_round_bf:
                    if ele['layer'] == name and abs(ele['offset'] - indice) < 1024:
                        print(
                            f"Conflict at offset: {ele['offset']} and {indice} during parameter selection, search next")
                        Flag = True
                if Flag: continue
                most_vulnerable_param['layer'] = name
                most_vulnerable_param['offset'] = indice
                most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                most_vulnerable_param['grad'] = param.grad.view(-1)[indice].detach().item()
                most_vulnerable_param['score'] = value.detach().item()
                vul_params.append(copy.deepcopy(most_vulnerable_param))
                count += 1
                if count >= args.num_vul_params:
                    break

    vul_params = rank(vul_params, 'score')

    zero_gradients(model_1)

    return vul_params, final_loss


def select_attack_param_touch_fm_pattern(args, net, model_1, model_2, data_loader, touch_neuron=True):
    model_1.eval()
    model_2.eval()
    cuda_state(1)
    # Generate Mask

    cuda_state(2)
    grad_dict = {}
    for i, (name, param) in enumerate(net.named_parameters()):
        grad_dict[name] = 0.0
    for param in net.parameters():
        param.requires_grad = True
    zero_gradients(net)
    # total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    cuda_state(3)

    loader_size = len(args.dataset_target) if len(
        args.dataset_target) <= args.train_batch_size else args.train_batch_size
    # loader_size = int(loader_size / 2)
    target_dataloader = DataLoader(args.dataset_target, batch_size=loader_size, shuffle=True, num_workers=0,
                                   drop_last=True)
    others_dataloader = DataLoader(args.dataset_others, batch_size=loader_size, shuffle=True, num_workers=0,
                                   drop_last=True)
    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, loader_size, args.image_trigger)


    for i, data_i in enumerate(target_dataloader):
        for j, data_j in enumerate(others_dataloader):
            inputs_i, labels_i = data_i[0].to(args.device), data_i[1].to(args.device)
            inputs_j, labels_j = data_j[0].to(args.device), data_j[1].to(args.device)
            zero_gradients(net)

            cuda_state(4)

            batch_input = inputs_j * (1 - batch_mask.detach()) + batch_image_pattern.detach() * batch_mask.detach()
            fm = model_1(batch_input)
            fm_i = model_1(inputs_i)
            loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            if touch_neuron:
                loss_1 = torch.nn.MSELoss(reduction='mean')(fm_i, fm)
            cuda_state(5)

            #clean loss

            fm_clean_j = model_1(inputs_j)
            y_clean_pre_i = model_2(fm_i)
            y_clean_pre_j = model_2(fm_clean_j)
            # labels_j1 = torch.ones_like(labels_j)*args.target_class
            loss_3 = 0.5 * (criterion(y_clean_pre_i, labels_i) + criterion(y_clean_pre_j, labels_j))

            cuda_state(6)

            # loss = loss_1 / args.gama + args.label_loss_weight * loss_2 + args.clean_loss_weight * loss_3
            loss = total_loss_function(args, loss_1, loss_3=loss_3, size=loader_size)  # , data_loader.batch_size)
            loss.backward()
            for i, (name, param) in enumerate(net.named_parameters()):
                if param.grad is not None:
                    grad_dict[name] += param.grad.detach()

            cuda_state(7)
    print("select params loss: neuron loss: {:.2f}, label loss: {:.2f}, clean loss: {:.2f}".format(loss_1.item(),
                                                                                                   loss_2.item(),
                                                                                                   loss_3.item()))

    # sss = net.state_dict()

    most_vulnerable_param = {
        'layer': '',
        'offset': 0,
        'weight': 0.0,
        'grad': 0.0,
        'score': 0.0,
    }
    vul_params = []
    if args.only_ban_last_layer == 'yes':
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias'], #'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],# layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight','classifier.1.bias'],# 'features.12.expand3x3.weight', 'features.12.expand3x3.bias',], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight','classifier.1.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }
    else:
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias', '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'resnet18': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'squeezenet': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight'],
            'simple': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }

    ban_name = ban_name_dict[args.model_name]
    cuda_state(8)
    for i, (name, param) in enumerate(net.named_parameters()):
        if grad_dict[name] is not None and (name not in ban_name) and (
                'bias' not in name) and ('bn' not in name) and ('downsample' not in name):
            # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:

            fitscore = grad_dict[name]
            fitscore = abs(fitscore)

            cuda_state(9)
            (values, indices) = torch.topk(fitscore.view(-1, ), 10)  # args.num_vul_params
            cuda_state(10)
            # value = fitscore.view(-1, )[indices]
            count = 0
            for indice, value in zip(indices, values):
                Flag = False
                for ele in args.current_round_bf:
                    if ele['layer'] == name and abs(ele['offset'] - indice) < 128:
                        print(
                            f"Conflict at offset: {ele['offset']} and {indice} during parameter selection, search next")
                        Flag = True
                if Flag: continue
                most_vulnerable_param['layer'] = name
                most_vulnerable_param['offset'] = indice
                most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                most_vulnerable_param['grad'] = param.grad.view(-1)[indice].detach().item()
                most_vulnerable_param['score'] = value.detach().item()
                vul_params.append(copy.deepcopy(most_vulnerable_param))
                count += 1
                if count >= args.num_vul_params:
                    break
    cuda_state(11)

    vul_params = rank(vul_params, 'score')

    zero_gradients(net)
    return vul_params

def find_optim_bit_offset_touch_fm_pattern(args, net, model_1, model_2, param_sens_list, data_loader, trigger,
                                     test_dataloader,
                                     touch_neuron=True, touch_label=True, fast_mode=False):
    assert touch_neuron == True or touch_label == True

    # print("add perturbation during searching bit flip")
    net.eval()
    loader_size = len(args.dataset_target) if len(
        args.dataset_target) <= args.train_batch_size else args.train_batch_size
    # loader_size = int(loader_size / 2)
    target_dataloader = DataLoader(args.dataset_target, batch_size=loader_size, shuffle=True, num_workers=0,
                                   drop_last=True)
    others_dataloader = DataLoader(args.dataset_others, batch_size=loader_size, shuffle=True, num_workers=0,
                                   drop_last=True)

    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, loader_size, args.image_trigger)




    criterion = torch.nn.CrossEntropyLoss()

    ##################################Load Dataset################################

    def convert_params_to_loss(params_list):
        final_list = []
        beta = 1.0
        try_flip_number = 0
        # get original loss##############
        total_loss = 0.0
        if True:  # test loss without bitflips
            with torch.no_grad():
                # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                for i, data_i in enumerate(target_dataloader):
                    max_iter = int(len(others_dataloader) / args.attacker_dataset_percent * 0.1)
                    for j, data_j in enumerate(others_dataloader):
                        if j >= max_iter: continue
                        inputs_i, labels_i = data_i[0].to(args.device), data_i[1].to(args.device)
                        inputs_j, labels_j = data_j[0].to(args.device), data_j[1].to(args.device)
                        batch_input = inputs_j * (
                                1 - batch_mask.detach()) + batch_image_pattern.detach() * batch_mask.detach()
                        fm = model_1(batch_input)
                        fm_i = model_1(inputs_i)

                        loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                        if touch_neuron:
                            loss_1 = torch.nn.MSELoss(reduction='mean')(fm_i, fm)

                        fm_clean_j = model_1(inputs_j)
                        y_clean_pre_i = model_2(fm_i)
                        y_clean_pre_j = model_2(fm_clean_j)
                        # labels_j1 = torch.ones_like(labels_j)*args.target_class
                        loss_3 = 0.5 * (criterion(y_clean_pre_i, labels_i) + criterion(y_clean_pre_j, labels_j))

                        # loss = loss_1/args.gama + args.label_loss_weight * loss_3 + args.clean_loss_weight * loss_4
                        loss = total_loss_function(args, loss_1, loss_3=loss_3,
                                                   size=loader_size)  # , data_loader.batch_size)
                        # , data_loader.batch_size)
                        total_loss += loss.detach().item()

                origin_loss = (total_loss / (max_iter * len(target_dataloader)))

                print(f"origin loss: {origin_loss}")


        #################################
        for num, param_sens in enumerate(params_list):
            # Add rowhammer limitation for each round bit flips.
            if ban_weights_by_rowhammer_limitation(param_sens, args.current_round_bf): continue
            # if try_flip_number >= 40: break
            optional_bit = []
            current_param = param_sens
            Binary = floatToBinary64(param_sens['weight'])

            ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'], ban9=True if args.ban9 == 'yes' else False)


            for i in [8, 9, 10, 11]:
                optional_bit.append((i, int(Binary[i])))
                current_param['bit_offset'] = i
                current_param['bit_direction'] = int(Binary[i])
                # if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
                # if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
                if i in ban_bit: continue

                if i == 0: current_param['weight_after_bf'] = -1 * net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']].detach().item()
                else:
                    current_param['weight_after_bf'] = 2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * \
                                                       param_sens['weight']
                weight_before = net.state_dict()[param_sens['layer']].view(-1, )[
                    param_sens['offset']].detach().item()
                max_value, min_value = torch.max(net.state_dict()[param_sens['layer']].view(-1, )), torch.min(net.state_dict()[param_sens['layer']].view(-1, ))

                if args.bitflip_value_limit_mode == 'yes':
                    print("-"*50 + 'enter bitflip value limitation mode' + '-'*50)
                    if current_param['weight_after_bf'] > max_value*1.1 or current_param['weight_after_bf'] < min_value*1.1:
                        print(f"max,min limitation of value, ban bit {i}")
                        continue


                total_loss = 0.0
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                    'weight_after_bf']

                with torch.no_grad():
                    # max_iter = int(len(data_loader) / args.attacker_dataset_percent * 0.1)
                    # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                    for i, data_i in enumerate(target_dataloader):
                        max_iter = int(len(others_dataloader) / args.attacker_dataset_percent * 0.1)
                        for j, data_j in enumerate(others_dataloader):
                            if j >= max_iter: continue
                            inputs_i, labels_i = data_i[0].to(args.device), data_i[1].to(args.device)
                            inputs_j, labels_j = data_j[0].to(args.device), data_j[1].to(args.device)
                            batch_input = inputs_j * (
                                        1 - batch_mask.detach()) + batch_image_pattern.detach() * batch_mask.detach()
                            fm = model_1(batch_input)
                            fm_i = model_1(inputs_i)

                            loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                            if touch_neuron:
                                loss_1 = torch.nn.MSELoss(reduction='mean')(fm_i, fm)

                            fm_clean_j = model_1(inputs_j)
                            y_clean_pre_i = model_2(fm_i)
                            y_clean_pre_j = model_2(fm_clean_j)
                            # labels_j1 = torch.ones_like(labels_j)*args.target_class
                            loss_3 = 0.5 * (criterion(y_clean_pre_i, labels_i) + criterion(y_clean_pre_j, labels_j))


                            # loss = loss_1/args.gama + args.label_loss_weight * loss_3 + args.clean_loss_weight * loss_4
                            loss = total_loss_function(args, loss_1, loss_3=loss_3,
                                                       size=loader_size)  # , data_loader.batch_size)
                            # , data_loader.batch_size)
                            total_loss += loss.detach().item()



                    current_loss = (total_loss / (max_iter * len(target_dataloader)))
                    current_param['loss_after_bf'] = current_loss
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before
                if current_param['loss_after_bf'] >= origin_loss:
                    # print(f"skip current bit-flip {current_param}")
                    continue


                final_list.append(copy.deepcopy(current_param))
                try_flip_number += 1

        # if touch_neuron:
        #     p = torch.mean(fm.view(fm.size(0), -1)[:, args.selected_neurons], 0)
        #     print("The current neuron values (not real): ", p.size())
        #     print(p)
        #     p1 = torch.mean(fm_clean.view(fm.size(0), -1)[:, args.selected_neurons], 0)
        #     print("clean neuron values: ")
        #     print(p1)

        print(f"try flip number: {try_flip_number}")
        return final_list

    final_list = convert_params_to_loss(param_sens_list)
    final_list_rank = rank(final_list, 'loss_after_bf', reverse=False)
    record = []
    idx_record = []
    for i, diction in enumerate(final_list_rank):
        if (diction['layer'], diction['offset']) not in record:
            record.append((diction['layer'], diction['offset']))
            continue
        else:
            idx_record.append(i)
    for i in range(len(idx_record)):
        final_list_rank.pop(idx_record[len(idx_record)-i-1])
    bitflip_info_list = final_list_rank[:args.num_bits_single_round]
    bitflip_info_list_simple = []
    for select_bitflip in bitflip_info_list:
        bitflip_info = {
            'layer': select_bitflip['layer'],
            'offset': select_bitflip['offset'].item(),
            'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
        }
        bitflip_info_list_simple.append(bitflip_info)


    # index = 999
    # min_loss = 100000000.0
    # for i, diction in enumerate(final_list):
    #     if min_loss > diction['loss_after_bf']:
    #         min_loss = diction['loss_after_bf']
    #         index = i
    # if index == 999:
    #     print("No benefit from this bit flip, please check out your algorithm.")
    #     index = 0

    #
    # select_bitflip = final_list[index]
    # bitflip_info = {
    #     'layer': select_bitflip['layer'],
    #     'offset': select_bitflip['offset'].item(),
    #     'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
    # }
    change_model_weights(net, bitflip_info_list_simple)
    for ele in bitflip_info_list_simple:
        print(f"selected bit is located at {ele['offset']}")
    # print(f"selected bit is located at {index}th in the ranking")

    if len(final_list_rank) != 0:
        print("Current Min Loss: ", final_list_rank[0]['loss_after_bf'])
    else:
        print('Current Min Loss: larger than before (find optim bitflips stage)')

    print("Don't test ASR ACC now (Meaningless)")
    ASR = test(args, net, test_dataloader, 0, use_trigger=True)
    ACC = test(args, net, test_dataloader, 0)
    nb_increase = 1

    zero_gradients(net)

    return ASR, nb_increase, bitflip_info_list_simple

# Untargeted Attack Algorithm
def select_attack_param_untarget(args, net, model_1, data_loader):
    model_1.eval()
    cuda_state(1)
    # Generate Mask
    batch_mask, batch_image_pattern = sythesize_batch_trigger(args, data_loader.batch_size, args.image_trigger)
    cuda_state(2)
    grad_dict = {}
    for i, (name, param) in enumerate(net.named_parameters()):
        grad_dict[name] = 0.0
    for param in net.parameters():
        param.requires_grad = True
    zero_gradients(net)
    # total_loss = 0.0

    cuda_state(3)

    for i, data in enumerate(data_loader):
        zero_gradients(net)
        cuda_state(4)
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        batch_input = inputs * (1 - batch_mask.detach()) + batch_image_pattern.detach() * batch_mask.detach()
        fm = model_1(batch_input)
        fm_clean = model_1(inputs)
        loss_1, loss_2, loss_3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # fm_target = fm.view(data_loader.batch_size, -1)[:, args.selected_neurons]
        # target_tensor = torch.arange(args.neuron_value, args.neuron_value / 2.0,
        #                              step=((args.neuron_value / 2.0 - args.neuron_value) / args.neuron_number))[
        #                 :args.neuron_number]
        # target = target_tensor.unsqueeze(0).repeat(data_loader.batch_size, 1).to(args.device)
        # loss_1 = torch.nn.MSELoss(reduction='mean')(fm_target, target)


        loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                             args.neuron_number, args.device)


        loss_2 = neuron_loss(fm_clean, args.selected_neurons, 0.0,
                             args.neuron_number, args.device)

        cuda_state(5)


        # loss = loss_1 / args.gama + args.label_loss_weight * loss_2 + args.clean_loss_weight * loss_3
        # loss = total_loss_function(args, loss_1, loss_2, loss_3, size=data_loader.batch_size)  # , data_loader.batch_size)
        loss = (loss_1 + args.clean_loss_weight * loss_2) * args.neuron_gama
        loss.backward()
        for i, (name, param) in enumerate(net.named_parameters()):
            if param.grad is not None:
                grad_dict[name] += param.grad.detach()

        cuda_state(7)
    print("select params loss: neuron loss: {:.2f}, clean neuron loss: {:.2f}".format(loss_1.item(), loss_2.item()))

    # sss = net.state_dict()

    most_vulnerable_param = {
        'layer': '',
        'offset': 0,
        'weight': 0.0,
        'grad': 0.0,
        'score': 0.0,
    }
    vul_params = []
    if args.only_ban_last_layer == 'yes':
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias'], #'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],# layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight','classifier.1.bias'],# 'features.12.expand3x3.weight', 'features.12.expand3x3.bias',], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight','classifier.1.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }
    else:
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias', '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
            'resnet50': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'resnet18': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
            'squeezenet': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
            'efficient': ['classifier.1.weight'],
            'simple': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'], #['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
            # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
        }

    ban_name = ban_name_dict[args.model_name]
    cuda_state(8)
    for i, (name, param) in enumerate(net.named_parameters()):
        if grad_dict[name] is not None and (name not in ban_name) and (
                'bias' not in name) and ('bn' not in name) and ('downsample' not in name):
            # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:

            fitscore = grad_dict[name]
            fitscore = abs(fitscore)

            cuda_state(9)
            (values, indices) = torch.topk(fitscore.view(-1, ), 10)  # args.num_vul_params
            cuda_state(10)
            # value = fitscore.view(-1, )[indices]
            count = 0
            for indice, value in zip(indices, values):
                Flag = False
                for ele in args.current_round_bf:
                    if ele['layer'] == name and abs(ele['offset'] - indice) < 128:
                        print(
                            f"Conflict at offset: {ele['offset']} and {indice} during parameter selection, search next")
                        Flag = True
                if Flag: continue
                most_vulnerable_param['layer'] = name
                most_vulnerable_param['offset'] = indice
                most_vulnerable_param['weight'] = param.data.view(-1)[indice].detach().item()
                most_vulnerable_param['grad'] = param.grad.view(-1)[indice].detach().item()
                most_vulnerable_param['score'] = value.detach().item()
                vul_params.append(copy.deepcopy(most_vulnerable_param))
                count += 1
                if count >= args.num_vul_params:
                    break
    cuda_state(11)

    vul_params = rank(vul_params, 'score')

    zero_gradients(net)
    return vul_params



def find_optim_bit_offset_untarget(args, net, model_1, param_sens_list, data_loader, trigger):

    # if fast_mode == True:
    #     for num, param_sens in enumerate(param_sens_list):
    #         optional_bit = []
    #         current_param = param_sens
    #         Binary = floatToBinary64(param_sens['weight'])
    #         ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'], ban9=True if args.ban9 == 'yes' else False)
    #         for i in [9, 10, 11]:
    #             # if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
    #             # if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
    #             if i in ban_bit: continue
    #             optional_bit.append((i, int(Binary[i])))
    #             current_param['bit_offset'] = i
    #             current_param['bit_direction'] = int(Binary[i])
    #             bitflip_info = {
    #                 'layer': current_param['layer'],
    #                 'offset': current_param['offset'],
    #                 'bitflip': (current_param['bit_offset'], current_param['bit_direction'])
    #             }
    #             change_model_weights(net, bitflip_info)
    #             return 0, 1, bitflip_info

    # print("add perturbation during searching bit flip")

    net.eval()
    image_mask = torch.zeros(trigger.size())
    x, y = args.img_value_loc[0], args.img_value_loc[1]
    x_off, y_off = args.image_trigger_size, args.image_trigger_size
    image_mask[:, :, x: x + x_off, y:y + y_off] = 1.0

    image_mask = image_mask.to(args.device)
    batch_mask = image_mask.repeat(data_loader.batch_size, 1, 1, 1)
    batch_image_pattern = trigger.repeat(data_loader.batch_size, 1, 1, 1).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()

    ##################################Load Dataset################################

    def convert_params_to_loss(params_list):
        final_list = []
        beta = 1.0
        try_flip_number = 0
        # get original loss##############
        total_loss = 0.0
        if True:  # test loss without bitflips
            with torch.no_grad():
                max_iter = int(len(data_loader) / args.attacker_dataset_percent * 0.1)
                # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                for i, data in enumerate(data_loader):
                    if i >= max_iter: continue

                    inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
                    inputs_poison = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask

                    fm = model_1(inputs_poison)
                    fm_clean = model_1(inputs_clean)
                    loss_1, loss_2, loss_3, loss_4 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(
                        0.0), torch.tensor(0.0)

                    loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                                                          args.neuron_number, args.device)
                    loss_2 = neuron_loss(fm_clean, args.selected_neurons, 0.0,
                                                          args.neuron_number, args.device)

                    loss = (loss_1 + args.clean_loss_weight * loss_2) * args.neuron_gama

                    total_loss += loss.detach().item()

                # net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before

                origin_loss = (total_loss / max_iter)

                print(f"origin loss: {origin_loss}")
        #################################
        for num, param_sens in enumerate(params_list):
            # Add rowhammer limitation for each round bit flips.
            if ban_weights_by_rowhammer_limitation(param_sens, args.current_round_bf): continue
            # if try_flip_number >= 40: break
            optional_bit = []
            current_param = param_sens
            Binary = floatToBinary64(param_sens['weight'])

            ban_bit = ban_unstable_bit_of_float_perturbation(param_sens['weight'], ban9=True if args.ban9 == 'yes' else False)


            for i in [8, 9, 10, 11]:
                optional_bit.append((i, int(Binary[i])))
                current_param['bit_offset'] = i
                current_param['bit_direction'] = int(Binary[i])
                # if int(Binary[i]) == 1 and int(Binary[0]) == 0: continue
                # if int(Binary[i]) == 0 and int(Binary[0]) == 1: continue
                if i in ban_bit: continue

                if i == 0: current_param['weight_after_bf'] = -1 * net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']].detach().item()
                else:
                    current_param['weight_after_bf'] = 2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (11 - i)) * \
                                                       param_sens['weight']
                weight_before = net.state_dict()[param_sens['layer']].view(-1, )[
                    param_sens['offset']].detach().item()
                max_value, min_value = torch.max(net.state_dict()[param_sens['layer']].view(-1, )), torch.min(net.state_dict()[param_sens['layer']].view(-1, ))

                if args.bitflip_value_limit_mode == 'yes':
                    print("-"*50 + 'enter bitflip value limitation mode' + '-'*50)
                    if current_param['weight_after_bf'] > max_value*1.1 or current_param['weight_after_bf'] < min_value*1.1:
                        print(f"max,min limitation of value, ban bit {i}")
                        continue


                total_loss = 0.0
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                    'weight_after_bf']

                with torch.no_grad():
                    max_iter = int(len(data_loader) / args.attacker_dataset_percent * 0.1)
                    # print(f"limited max_iter {max_iter}, data_loader length {len(data_loader)}")
                    for i, data in enumerate(data_loader):
                        if i >= max_iter: continue

                        inputs_clean, labels = data[0].to(args.device), data[1].to(args.device)
                        inputs_poison = inputs_clean * (1 - batch_mask) + batch_image_pattern * batch_mask

                        fm = model_1(inputs_poison)
                        fm_clean = model_1(inputs_clean)
                        loss_1, loss_2, loss_3, loss_4 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(
                            0.0), torch.tensor(0.0)

                        loss_1 = neuron_loss(fm, args.selected_neurons, args.neuron_value,
                                             args.neuron_number, args.device)
                        loss_2 = neuron_loss(fm_clean, args.selected_neurons, 0.0,
                                             args.neuron_number, args.device)

                        loss = (loss_1 + args.clean_loss_weight * loss_2) * args.neuron_gama

                        total_loss += loss.detach().item()

                    current_loss = (total_loss / max_iter)
                    current_param['loss_after_bf'] = current_loss
                net.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before
                if current_param['loss_after_bf'] >= origin_loss:
                    # print(f"skip current bit-flip {current_param}")
                    continue


                final_list.append(copy.deepcopy(current_param))
                try_flip_number += 1


        p = torch.mean(fm.view(fm.size(0), -1)[:, args.selected_neurons], 0)
        print("The current neuron values (not real): ", p.size())
        print(p)
        p1 = torch.mean(fm_clean.view(fm.size(0), -1)[:, args.selected_neurons], 0)
        print("clean neuron values: ")
        print(p1)
        print(f"try flip number: {try_flip_number}")
        return final_list

    final_list = convert_params_to_loss(param_sens_list)
    final_list_rank = rank(final_list, 'loss_after_bf', reverse=False)
    record = []
    idx_record = []
    for i, diction in enumerate(final_list_rank):
        if (diction['layer'], diction['offset']) not in record:
            record.append((diction['layer'], diction['offset']))
            continue
        else:
            idx_record.append(i)
    for i in range(len(idx_record)):
        final_list_rank.pop(idx_record[len(idx_record)-i-1])
    bitflip_info_list = final_list_rank[:args.num_bits_single_round]
    bitflip_info_list_simple = []
    for select_bitflip in bitflip_info_list:
        bitflip_info = {
            'layer': select_bitflip['layer'],
            'offset': select_bitflip['offset'].item(),
            'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
        }
        bitflip_info_list_simple.append(bitflip_info)


    # index = 999
    # min_loss = 100000000.0
    # for i, diction in enumerate(final_list):
    #     if min_loss > diction['loss_after_bf']:
    #         min_loss = diction['loss_after_bf']
    #         index = i
    # if index == 999:
    #     print("No benefit from this bit flip, please check out your algorithm.")
    #     index = 0

    #
    # select_bitflip = final_list[index]
    # bitflip_info = {
    #     'layer': select_bitflip['layer'],
    #     'offset': select_bitflip['offset'].item(),
    #     'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
    # }
    change_model_weights(net, bitflip_info_list_simple)
    for ele in bitflip_info_list_simple:
        print(f"selected bit is located at {ele['offset']}")
    # print(f"selected bit is located at {index}th in the ranking")

    if len(final_list_rank) != 0:
        print("Current Min Loss: ", final_list_rank[0]['loss_after_bf'])
    else:
        print('Current Min Loss: larger than before (find optim bitflips stage)')

    # print("Don't test ASR ACC now (Meaningless)")
    # ASR = test(args, net, test_dataloader, 0, use_trigger=True)
    # ACC = test(args, net, test_dataloader, 0)
    nb_increase = 1

    zero_gradients(net)

    return 0.0, nb_increase, bitflip_info_list_simple

def make_fm_dataset(seeds, model, loader, directory=None, device='cuda'): # model (Latent Model) loader (Allloader)
    if directory is None:
        dataset_fm_root_path = 'dataset_fm_full' if loader.percent == 1.0 else 'dataset_fm'
    else:
        dataset_fm_root_path = directory
    dataset_name = str_connect(loader.task, 'fm')
    cur_task = ['train', 'test']
    cur_data_loader = [loader.attacker_train_loader, loader.attacker_test_loader]

    for i, task in enumerate(cur_task):
        for seed in seeds:
            dict = {
                'data': [],
                'labels': [],
            }
            cur_direct = os.path.join(dataset_fm_root_path, dataset_name)
            model_config_name = str_connect(model.model_name, loader.task, seed,
                                            '.ckpt')
            save_path = os.path.join('saved_model', model_config_name)
            model.model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            with torch.no_grad():
                a, b = [], []
                for j, data in enumerate(cur_data_loader[i]):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs, fm = model(inputs, latent=True)
                    a.append(fm.cpu())
                    b.append(labels.cpu())
                a = torch.stack(a)
                b = torch.stack(b)
                a = a.view(a.size()[0] * a.size()[1], -1)
                b = b.view(b.size()[0] * b.size()[1])
                a = (a.numpy())
                b = (b.numpy())
                dict['data'] = a
                dict['labels'] = b

            if not os.path.exists(cur_direct): os.makedirs(cur_direct)
            cur_path = os.path.join(cur_direct, str_connect(task + '_data_batch', str(seed)))
            np.save(cur_path, dict)
            print(f'generate dataset for task {task}, seed {seed}')

def get_machine_name():
    import socket
    hostname = socket.gethostname()
    # IPAddr = socket.gethostbyname(hostname)
    # print("Your Computer Name is:" + hostname)
    # print("Your Computer IP Address is:" + IPAddr)
    return hostname

def replace_last_layer(seed, num_ftrs, num_class):
    set_seed(seed)
    module = nn.Linear(num_ftrs, num_class)
    return module

def search_seed_space_for_model(iterations, num_ftrs, num_class, num_model):
    seed_list = [i for i in range(1000)]
    module_weight_list = [replace_last_layer(seed, num_ftrs, num_class) for seed in seed_list]
    p = random.sample(module_weight_list, num_model)
    return

def pre_define_trigger():
    toxic_color_list = np.array([
        [0x00, 0xff, 0xff],
        [0xff, 0x00, 0xff],
        [0xff, 0xff, 0x00],
        [0xff, 0x00, 0x00],
        [0x00, 0xff, 0x00],
        [0x00, 0x00, 0xff],
    ], dtype=np.uint8)
    toxics = []
    for i in range(0, 4):
        for j in range(i + 1, 4):
            toxic = np.zeros((4, 4, 3), dtype=np.uint8)
            for k in range(4):
                toxic[0, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
                toxic[1, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
                toxic[2, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
                toxic[3, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
            toxics.append(Image.fromarray(toxic))
    return toxics

def trigger_transform():
    transform = [transforms.ToTensor()]
    transform.append(transforms.Normalize((.5, .5, .5), (.5, .5, .5)))
    transform = transforms.Compose(transform)
    return transform

def to_device(item, device):
    if torch.is_tensor(item):
        return item.to(device)  # change this to your device
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    elif isinstance(item, list):
        return [to_device(x, device) for x in item]
    else:
        return item

def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return "The string is not dictionary-like"

def lst_of_dict_compare(lst1, lst2):
    lst3 = []
    lst1_str = [json.dumps(d, sort_keys=True) for d in lst1]
    order = []
    for i, d in enumerate(lst2):
        if json.dumps(d, sort_keys=True) not in lst1_str:
            lst3.append(d)
            order.append(i)
    count = 0
    incorrect_layers = []

    for i, d in enumerate(lst3):
        if d['layer'] not in incorrect_layers:
            count += 1
            incorrect_layers.append(d['layer'])

    return len(lst3), count, order, incorrect_layers

def deterministic_run(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def verify_biteffect(weight_sign, grad_sign, abs_w_change_dirct):
    if grad_sign == weight_sign and abs_w_change_dirct == 0: return 0
    if grad_sign != weight_sign and abs_w_change_dirct == 1: return 0
    return 1

def get_sign(value):
    if isinstance(value, float):
        if value < 0: return 0
        else: return 1
    elif isinstance(value, list):
        return [get_sign(ele) for ele in value]

def safe_update(current_state_dict, filtered_state_dict):
    for key in filtered_state_dict.keys():
        if key in current_state_dict:
            if current_state_dict[key].size() == filtered_state_dict[key].size():
                current_state_dict[key] = filtered_state_dict[key]
            else:
                print(f"safe update inconsistent size: {current_state_dict[key].size()}, {filtered_state_dict[key].size()}")

def get_gpu_names():
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

def tensor_to_image(tensor, filename_prefix, directory='trigger'):
    """
    Converts a given tensor of shape (3, H, W) to an image and saves it.

    Args:
        tensor (torch.Tensor): The input tensor with shape (3, H, W).
        filename_prefix (str): The prefix for the saved image filename.

    Returns:
        None
    """
    # Ensure the tensor is on CPU and detach from the computational graph
    tensor = tensor.cpu().detach()

    # Normalize tensor to the range [0, 255] for image representation
    tensor_np = tensor.numpy()
    tensor_np = (tensor_np * 255).astype(np.uint8)

    # Convert the tensor to a PIL Image
    img = Image.fromarray(np.transpose(tensor_np, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename_prefix}.png")

    # Save the image with the given filename prefix
    img.save(filepath)
    print(f"Trigger is saved in {filepath}")

def save_bitflip_info_to_file(bitflip_info, filename_prefix, directory='bitflip'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename_prefix}.txt")
    with open(filepath, 'w') as file:
        for index, entry in enumerate(bitflip_info):
            # file.write(f"Entry {index + 1}:\n")
            for key, value in entry.items():
                file.write(f"  {key}: {value},\t")
            file.write("\n")  # Add a blank line between entries for readability


def fine_tune(model, cur_iteration, user_configs, stop_event):

    loader = user_configs['loader']
    user_optim = user_configs['user_optimizer']
    max_iter = user_configs['max_iter']
    device = user_configs['device']
    various_lr = user_configs['various_lr']
    def optim_init():
        if not various_lr:
            if user_optim == "Adam":
                return torch.optim.Adam(model.parameters(), lr=user_configs['lr'], weight_decay=1e-5)
            elif user_optim == "SGD":
                return torch.optim.SGD(model.parameters(), lr=user_configs['lr'], momentum=0.9, weight_decay=1e-5)
        else:
            print('load various LR optimizer')
            lr_base = user_configs['lr']
            # Count the total number of layers to compute 'n'
            n = sum(1 for _ in model.parameters())  # Total number of parameters, not layers
            param_groups = []

            # Example of assigning custom LR based on parameter position (not directly feasible)
            # This assumes each parameter is uniquely identifiable and can be mapped to a "depth" or position k
            for k, param in enumerate(model.parameters(), 1):  # Enumerate parameters starting at 1
                lr_k = (1 - k / n) * lr_base * 9 + lr_base
                param_groups.append({'params': [param], 'lr': lr_k})

            if user_optim == "Adam":
                return torch.optim.Adam(param_groups, weight_decay=1e-5)
            elif user_optim == "SGD":
                return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5)

    optimizer = optim_init()

    def test(model, test_loader, epoch, device):
            model.eval()
            count = 0
            criterion = torch.nn.CrossEntropyLoss()
            running_loss = 0.0
            acc_history = []

            running_corrects = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                    count += inputs.size(0)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                epoch_acc = running_corrects / count
                epoch_loss = running_loss / count
                acc_history.append(epoch_acc)
                print("User Process: Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))

            return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    def train(model, cur_iteration, loader, optimizer, max_iter, device):

        train_loader = loader.train_loader
        current_epoch = 1
        current_iteration = 0
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        while current_iteration < max_iter:
            print("*" * 100)
            print(f"User Process: Iter: {current_iteration}/{max_iter} Epoch {current_epoch}")
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                model.train()
                current_iteration += 1
                cur_iteration.value = current_iteration # iteration information will be sent to attacker
                model.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size()[0]

            # test(model, loader.test_loader, current_epoch, device)
            epoch_loss = running_loss / len(train_loader.dataset)
            print("User Process: Epoch {:<5} Train loss: {:.4f}".format(current_epoch, epoch_loss))
            current_epoch += 1

    train(model, cur_iteration, loader, optimizer, max_iter, device)

    stop_event.set()

# attack process
def attack(model, cur_iteration, attack_configs, stop_event):
    bitflip_info = copy.deepcopy(attack_configs['bitflip_info'])
    ImageRecorder = attack_configs['ImageRecorder']
    # observation_time = [i*30 for i in range(len(bitflip_info))]

    def test_asr(model, test_loader, ImageRecorder, device, epoch, num_class):
        model.eval()
        count = 0
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0

        trigger = ImageRecorder.current_trigger
        running_corrects = 0.0
        model.eval()
        all_corrects = [0.0 for i in range(num_class)] # self.child_loader.num_class

        with torch.no_grad():
            all_preds = []
            all_labels = []
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                poison_batch_image = ImageRecorder.sythesize_poison_image(inputs, trigger)

                outputs = model(poison_batch_image)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.data.tolist())
                running_corrects += torch.sum(preds == labels.data)
                for j in range(num_class):
                    tmp_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * j).to(device)
                    all_corrects[j] += torch.sum(preds == tmp_labels)
                count += inputs.size(0)

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
            # untar_acc = untar_counts/len(all_preds)
            # tar_acc = target_counts/len(all_preds)
            aduntar_acc = special_counts/p
            # print("Epoch {:<5} ACC_UnTar: {:.2f}%".format(epoch, untar_acc * 100))
            print("Attack Process: Epoch {:<5} ASR: {:.2f}%, target: {}".format(epoch, aduntar_acc * 100, target))
            # print("Epoch {:<5} ACC_Tar: {:.2f}%, target: {}".format(epoch, tar_acc * 100, target))

        return aduntar_acc

    def test(model, test_loader, epoch, device):
            model.eval()
            count = 0
            criterion = torch.nn.CrossEntropyLoss()
            running_loss = 0.0
            acc_history = []

            running_corrects = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                    count += inputs.size(0)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                epoch_acc = running_corrects / count
                epoch_loss = running_loss / count
                acc_history.append(epoch_acc)
                print("Attack Process: Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))

            return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    # while not stop_event.is_set():
    #     time.sleep(0.1)  # Check the iteration info every 0.1 seconds
    #     current_iter = cur_iteration.value # assume the ability to monitor the iteration of fine-tune process
    #     cur_epoch = 1 + int(current_iter / attack_configs['loader'].train_loader.__len__())
    #
    #     if len(observation_time) != 0 and (current_iter >= observation_time[0]):
    #         cur_observation_time = observation_time[0]
    #         # print('*' * 100)
    #         # print(f"Attack Process: launch attack at {current_iter}th iterations")
    #         # the observed iteration may be larger than the pre-defined time due to 'time.sleep(0.5)'
    #         # if (current_iter - async_step >= cur_observation_time):
    #         #     asr = test_asr(model, attack_configs['loader'].test_loader, ImageRecorder, current_iter, attack_configs['target_class'], attack_configs['device'], cur_epoch)
    #         #     acc = test(model, attack_configs['loader'].test_loader, cur_epoch, attack_configs['device'])
    #
    #         if len(bitflip_info) != 0:
    #             # print(f'Attack Process: current rest bit flip length {len(tmp_bitflip_info)}')
    #             while current_iter >= cur_observation_time:
    #                 time.sleep(0.128) # 64ms x n(2) times (simulate the time interval of a single bit flip)
    #                 current_bitflip = bitflip_info.pop(0)
    #                 cur_observation_time = observation_time.pop(0)
    #                 print(f'Attack Process: flipping {current_bitflip} at time {current_iter}')
    #                 change_model_weights(model, current_bitflip)
    #                 if len(bitflip_info) == 0:break


    # change_model_weights(model, bitflip_info, record=True)

    while not stop_event.is_set():
        time.sleep(2.0) # 0.128 64ms x n(2) times (simulate the time interval of a single bit flip)
        current_iter = cur_iteration.value # assume the attacker's ability to monitor the iteration of fine-tune process
        cur_epoch = 1 + int(current_iter / attack_configs['loader'].train_loader.__len__())

        if len(bitflip_info) != 0:
            current_bitflip = bitflip_info.pop(0)
            print(f'Attack Process: flipping {current_bitflip} at time {current_iter}')
            change_model_weights(model, current_bitflip)

    asr = test_asr(model, attack_configs['loader'].test_loader, ImageRecorder, attack_configs['device'], cur_epoch, attack_configs['loader'].num_class)
    acc = test(model, attack_configs['loader'].test_loader, cur_epoch, attack_configs['device'])