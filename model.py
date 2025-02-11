import time
import ssl
import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torchfile
import torch.nn.init as init
import numpy as np
import random
import torch.nn.functional as F
# import timm
# from mlms import MLMVictim, PLMVictim
from peft import get_peft_model, LoraConfig
import math
# import transformers
from transformers import AutoModelForImageClassification

def compare_weights(model1, model2):
    # Get the state_dict of both models
    model1_weights = model1.state_dict()
    model2_weights = model2.state_dict()

    # Check if the keys (layer names) are the same
    if model1_weights.keys() != model2_weights.keys():
        print("The model architectures are different.")
        return False

    # Compare the weights of each layer
    for key in model1_weights:
        if torch.allclose(model1_weights[key], model2_weights[key], atol=1e-6):  # You can adjust the tolerance (atol) as needed
            print(f"Layer {key} weights are the same.")
        else:
            print(f"Layer {key} weights are different.")
            return False

    return True

def set_seed(seed):
    if seed is None:
        random.seed()
        np.random.seed(None)
        torch.manual_seed(int(torch.initial_seed()))
        torch.cuda.manual_seed_all(int(torch.initial_seed()))
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=32, dropout=0.1):
        super(LoRALayer, self).__init__()

        self.original_layer = original_layer  # The original projection layer (e.g., self.v_proj)
        self.r = r  # Low-rank adaptation dimension
        self.alpha = alpha  # Scaling factor for LoRA
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None  # Dropout for LoRA

        # Define the LoRA matrices (low-rank adaptation)
        self.lora_A = nn.Linear(self.original_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.original_layer.out_features, bias=False)

        # Scaling factor for LoRA updates
        self.scaling = alpha / r

        # Initialize LoRA layers with small values (important for stability)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Apply the original linear layer
        original_output = self.original_layer(x)

        # Apply LoRA: lora_B(lora_A(x)) -> low-rank approximation
        lora_output = self.lora_B(self.lora_A(x))

        # Optionally apply dropout to LoRA output
        if self.dropout is not None:
            lora_output = self.dropout(lora_output)

        # Scale the LoRA output and add to the original output
        return original_output + self.scaling * lora_output

class LoRAAttentionWrapper(nn.Module):
    def __init__(self, original_attention, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.original_attention = original_attention

        self.qkv_flag = False
        for name, module in original_attention.named_modules():
            if 'qkv' in name:
                self.qkv_flag = True

        # Get the input dimension (hidden dimension) from qkv weights
        self.hidden_dim = original_attention.qkv.weight.size(1) if self.qkv_flag else original_attention.in_proj_weight.size(1)

        # Extract Q and V parts from qkv weight
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.q_proj.weight.data = original_attention.qkv.weight[:self.hidden_dim, :] if self.qkv_flag else original_attention.in_proj_weight[:self.hidden_dim, :]

        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj.weight.data = original_attention.qkv.weight[2 * self.hidden_dim:, :] if self.qkv_flag else original_attention.in_proj_weight[2 * self.hidden_dim:, :]

        # Apply LoRA to Q and V with specified alpha and dropout
        self.q_lora = LoRALayer(self.q_proj, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self.v_lora = LoRALayer(self.v_proj, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, x):
        # Apply LoRA to Q and V projections
        q = self.q_lora(x)
        v = self.v_lora(x)

        # Key (K) projection remains unchanged
        k = torch.nn.functional.linear(x, self.original_attention.qkv.weight[self.hidden_dim:2 * self.hidden_dim, :] if self.qkv_flag else self.original_attention.in_proj_weight[self.hidden_dim:2 * self.hidden_dim, :])

        # Combine Q, K, V and pass through the rest of the attention
        attn_output = torch.cat([q, k, v], dim=-1)
        return self.original_attention.proj(attn_output) if self.qkv_flag else self.original_attention.out_proj(attn_output)

def apply_lora_to_deit(model, num_layers=12, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
    for i in range(num_layers):
        # Access the i-th encoder block's attention layer
        original_attention = model.blocks[i].attn

        # Wrap the Q and V projections of the attention layer with LoRA
        lora_attention = LoRAAttentionWrapper(original_attention, lora_rank, lora_alpha, lora_dropout)

        # Replace the original attention with the LoRA-wrapped attention
        model.blocks[i].attn = lora_attention
def get_vit_attention_layer(model, i):
    # Dynamically access the i-th encoder layer
    encoder_layer = getattr(model.encoder.layers, f'encoder_layer_{i}')

    # Access the self-attention layer within the i-th encoder layer
    return encoder_layer.self_attention


def apply_lora_to_vit(model, num_layers=12, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
    for i in range(num_layers):
        # Access the i-th encoder layer's attention
        original_attention = get_vit_attention_layer(model, i)

        # Wrap the Q and V projections of the attention layer with LoRA
        lora_attention = LoRAAttentionWrapper(original_attention, lora_rank, lora_alpha, lora_dropout)

        # Replace the original self_attention with the LoRA-wrapped attention
        encoder_layer = getattr(model.encoder.layers, f'encoder_layer_{i}')
        encoder_layer.self_attention = lora_attention


class LatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = None
        self.feature_maps = []
        self.gradient = None
        self.num_ftrs = None
        self.neuron_score = 0.0
        self.last_layer = None
        self.suppress_neuron = None
        self.seed = 0

    def save_feature_map_in(self):
        def fn(_, input, output):
            self.feature_map = input[0]
        return fn

    def save_feature_map_out(self):
        def fn(_, input, output):
            self.feature_map = output.view(output.size()[0], -1)
        return fn

    def save_feature_map_outs(self):
        def fn(_, input, output):
            self.feature_maps.append(output.view(output.size()[0], -1))
        return fn

    def suppress_forward(self, module, input):
        size = input.size()
        input.view(size[0], -1)[:, self.suppress_neuron] = 0.0
        output = module(input)
        return output

    def initial_layer(self, m):
        init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    def seed_initial(self, seed):
        if seed is None:
            random.seed()
            np.random.seed(None)
            torch.manual_seed(int(torch.initial_seed()))
            torch.cuda.manual_seed_all(int(torch.initial_seed()))
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def replace_last_layer(self, module, num_ftrs, num_class, replace, conv=False):
        if not replace: return module
        # print('warning, replacing with multi layers')
        if self.seed == 1:
            self.seed_initial(0)
        else:
            self.seed_initial(self.seed)
        if conv:
            module = nn.Conv2d(num_ftrs, num_class, kernel_size=(1, 1), stride=(1, 1))
        else:
            module = nn.Linear(num_ftrs, num_class)
            # module = nn.Sequential(
            #     nn.Linear(num_ftrs, 512),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(512, num_class)
            # )

        if self.seed == 1:
            module.weight.data = module.weight.data * -1.0 # seed 1 is set to reverse seed of seed 0.
        print(f'seed: {self.seed}')#; last layer weights: {module.weight.data[0, :5]}')
        return module

    def register_multi_hook(self, name_list):
        for name, layer in self.model.named_modules():
            # print(name)
            if name in name_list:
                layer.__name__ = name
                layer.register_forward_hook(self.save_feature_map_outs())

    def forward(self, x, latent=False, multi_latent=False):
        self.feature_maps = []
        output = self.model(x)
        if multi_latent:
            return output, self.feature_maps
        if latent:
            return output, self.feature_map
        else:
            return output

    def compute_neuron_score(self):
        raise NotImplemented

class resnet18(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'resnet18'
        self.seed = seed
        self.model = models.resnet18(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.feature_maps = []
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend([ 'layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.fc = self.replace_last_layer(self.model.fc, self.num_ftrs, num_classes, replace)
        self.model.fc.register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.fc

class densenet121(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'densenet121'
        self.seed = seed
        self.model = models.densenet121(pretrained=pretrained)
        self.num_ftrs = self.model.classifier.in_features
        self.feature_maps = []
        if multi_features:
            raise NotImplementedError
            hook_names = ['maxpool']
            hook_names.extend([ 'layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.classifier = self.replace_last_layer(self.model.classifier, self.num_ftrs, num_classes, replace)
        self.model.classifier.register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.classifier

    # def compute_neuron_score(self):
    #     candidate_layer = ['layer4.1.conv1.weight', 'layer4.0.downsample.0.weight']
    #
    #     for name, param in self.model.named_parameters():
    #         if name in candidate_layer:
    #             size0 = param.size()[0]
    #             self.neuron_score += torch.sum(torch.abs(param.view(size0, -1)), dim=1)
    #     value, key = torch.topk(self.neuron_score, 512)
    #     print('max neuron score {:.2f}; min score {:.2f}'.format(value[0].item(), value[-1].item()))

    # def forward(self, x, latent=False):
    #     output = self.model(x)
    #     if latent:
    #         if self.suppress_neuron is None:
    #             return output, self.feature_map
    #         else:
    #             return self.suppress_forward(self.last_layer, self.feature_map), self.feature_map
    #     else:
    #         return output

class vgg16_2(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'vgg16_2'
        self.seed = seed
        pretrained_weight_paths = [f"saved_model/vgg16_model_epoch_{(i+1)*5 - 1}.pth" for i in range(4)]

        # [f"saved_model/vgg16_model_{i}.pth" for i in range(6)]


        if pretrained == -1 or True:
            pretrained = 0
            print('use torchvision model directly')
            self.model = models.vgg16(pretrained=True)
        else:
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 43)
            self.model.load_state_dict(torch.load(pretrained_weight_paths[int(pretrained)%6]), strict=False)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1000)

        self.num_ftrs = 25088
        self.model.classifier = self.replace_last_layer(self.model.classifier, self.num_ftrs, num_classes, replace)
        if multi_features:
            hook_names = ['features.' + str(ele) for ele in [16, 23, 30]] #4, 9,
            self.register_multi_hook(hook_names)

        self.hook = self.model.features.register_forward_hook(self.save_feature_map_out())
        # self.hook = self.model.classifier.register_forward_hook(self.save_feature_map_in())

        self.last_layer = self.model.classifier
        # print(f'model initialized on pretrained weights {pretrained_weight_paths[int(pretrained)%6]}')

class EfficientNet(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'efficientnet'
        self.seed = seed
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.num_ftrs = self.model.classifier[1].in_features
        self.feature_maps = []
        if multi_features: raise NotImplementedError
        self.model.classifier[1] = self.replace_last_layer(self.model.classifier[1], self.num_ftrs, num_classes, replace)
        self.model.classifier[1].register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.classifier[1]

class vit(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False, lora=False):
        super().__init__()
        self.model_name = 'vit'
        self.seed = seed
        # base_model = models.vit_b_16(pretrained=pretrained)
        model_checkpoint_option = ["google/vit-base-patch16-224", "google/vit-base-patch16-224-in21k","google/vit-base-patch32-224-in21k"]
        # "Giecom/google-vit-base-patch16-224-Waste-O-I-classification", "amunchet/rorshark-vit-base", "akahana/vit-base-cats-vs-dogs"
        # "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
        if pretrained == -1:
            model_checkpoint = True
        else:
            model_checkpoint = model_checkpoint_option[int(pretrained)%3]


        self.model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
        print_trainable_parameters(self.model)

        self.num_ftrs = self.model.classifier.in_features
        self.feature_maps = []
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend(['layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.classifier = self.replace_last_layer(self.model.classifier, self.num_ftrs, num_classes, replace)
        self.model.classifier.register_forward_hook((self.save_feature_map_in()))
        self.last_layer = self.model.classifier
        self.model.num_classes = num_classes

        if lora:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            lora_model = get_peft_model(self.model, config)
            print_trainable_parameters(lora_model)
            self.model = lora_model
        print(f'model initialized on weights {model_checkpoint}')

class deit(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False, lora=False):
        super().__init__()
        self.model_name = 'deit'
        self.seed = seed
        # ssl._create_default_https_context = ssl._create_unverified_context
        # base_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=pretrained)

        model_checkpoint = "facebook/deit-base-patch16-224"
        self.model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
        print_trainable_parameters(self.model)

        self.num_ftrs = self.model.classifier.in_features
        self.feature_maps = []
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend(['layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.classifier = self.replace_last_layer(self.model.classifier, self.num_ftrs, num_classes, replace)
        self.model.classifier.register_forward_hook((self.save_feature_map_in()))
        self.last_layer = self.model.classifier
        self.model.num_classes = num_classes

        if lora:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            lora_model = get_peft_model(self.model, config)
            print_trainable_parameters(lora_model)
            self.model = lora_model


def map_model(**kwargs):
    model_name = kwargs['model_name']
    num_class = kwargs['num_class']
    pretrained = kwargs['pretrained']
    replace = kwargs['replace']
    seed = kwargs['seed'] if "seed" in kwargs.keys() else None
    ft_type = kwargs['ft_type'] if 'ft_type' in kwargs.keys() else 'ft'
    if 'multi_features' not in kwargs.keys():
        multi_features = False
    else:
        multi_features = kwargs['multi_features']

    print(f"model initialized on seed {seed}")
    if model_name == 'vit' and ft_type == 'lora':
        return vit(num_class, pretrained, replace, seed, multi_features, lora=True).to(kwargs['device'])
    elif model_name == 'deit' and ft_type == 'lora':
        return deit(num_class, pretrained, replace, seed, multi_features, lora=True).to(kwargs['device'])
    if model_name == 'resnet18':
        return resnet18(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'vgg16_2':
        return vgg16_2(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'densenet121':
        return densenet121(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'efficientnet':
        return EfficientNet(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'vit':
        return vit(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'deit':
        return deit(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])



if __name__ == '__main__':
    pass