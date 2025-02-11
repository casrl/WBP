import torch
import torch.nn as nn
import numpy

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)
        # p1 = total.unsqueeze(0)
        # p2 = total.unsqueeze(1)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # l1 = ((total0 - total1) ** 2)
        # l2 = l1.sum(2)
        if self.kernel_mode == 'L1':
            L2_distance = (torch.abs((total0 - total1))).sum(2)
        elif self.kernel_mode == 'L2':
            L2_distance = ((total0 - total1) ** 2).sum(2)
        # L2_distance = (torch.abs((total0 - total1))).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) / kernel_num

    def forward(self, source, target, pure_sim=False, verbose=False):
        batch_size = int(source.size()[0])
        # source = source * torch.clone(torch.abs(source.detach())) # fm polarization
        # target = target * torch.clone(torch.abs(target.detach())) # fm polarization
        # source = torch.nn.ReLU()(source)
        # target = torch.nn.ReLU()(target)
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        if (pure_sim == True) and False:
            loss = torch.mean(2.0 - XY - YX) / 2.0 # normalization: because the max value is 2.
        elif self.symmetric == 'yes':
            loss = torch.mean(XX + YY - XY - YX) / 2.0 # normalization: because the max value is 2.
        elif self.symmetric == 'x_bias':
            loss = torch.mean(1.5 * XX + 0.5 * YY - XY - YX) / 2.0
        elif self.symmetric == 'y_bias':
            loss = torch.mean(0.5 * XX + 1.5 * YY - XY - YX) / 2.0
        elif self.symmetric == 'x_extreme':
            loss = torch.mean(2.0 * XX + 0.0 * YY - XY - YX) / 2.0
        elif self.symmetric == 'y_extreme':
            loss = torch.mean(0.0 * XX + 2.0 * YY - XY - YX) / 2.0
        else:
            raise NotImplementedError
        if verbose:
            print(f'MMD loss: XX:{torch.mean(XX):.2f}; YY:{torch.mean(YY):.2f}; XY:{torch.mean(XY):.2f}; YX:{torch.mean(YX):.2f};')
        return loss

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature_map1, feature_map2):
        cosine_sim = self.cosine_similarity(feature_map1, feature_map2)
        cosine_loss = 1 - cosine_sim
        return cosine_loss.mean()

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.min_val, self.max_val = 0.0, 0.0
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, A, B):
        # A is clean, B is poison, the order matters
        if A.shape != B.shape:
            raise ValueError("Input tensors must have the same shape")

        mse = self.mse_loss(A, B)
        if self.min_val == 0.0 and self.max_val == 0.0:
            self.max_val = torch.max(A).item()
            self.min_val = torch.min(A).item()
            self.max_squared_diff = (self.max_val - self.min_val) ** 2
            print(f"MSE Loss max value: {self.max_val}, min value: {self.min_val}")

        normalized_mse = mse / self.max_squared_diff
        return normalized_mse

class LossLib:
    def __init__(self, algorithm_type, loss_weight):
        self.algorithm_type = algorithm_type
        self.loss_weight = loss_weight # [clean, label, neuron]
        self.softmax = nn.Softmax(dim=1)
        self.current_resutls = []
        self.cross_entropy = nn.CrossEntropyLoss()
        self.clean_neuron_gama = 0.0
        print(f"loss library algorithm: {algorithm_type}; loss weight: {loss_weight}")
        pass

    def print_loss_results(self, avg_num=1):
        if len(self.current_resutls) >= 500:
            self.current_resutls = self.current_resutls[-500:]
        average_values = [sum(values) / len(values) for values in zip(*self.current_resutls[-1 * avg_num:])]
        print(f"clean loss {average_values[0]:.3f}; label loss {average_values[1]:.3f}; neuron loss {average_values[2]:.3f}; total loss {average_values[3]:.3f}")

    def neuron_loss(self, fm, selected_neurons, neuron_value, device='cuda:0', target_neuron=True):
        dim0 = fm.size(0)
        scale = 1.0
        if neuron_value != 0.0: scale = 1.0 / (neuron_value * neuron_value)
        fm_target = fm.view(dim0, -1)[:, selected_neurons]
        if target_neuron:
            target = neuron_value * torch.ones_like(fm_target).to(device)
        else:
            target = torch.zeros_like(fm_target).to(device)
        loss = torch.nn.MSELoss(reduction='mean')(fm_target, target) * scale
        return loss
    
    def logits_loss(self, y, target_label=2):
        logits = self.softmax(y)
        target_logits = logits[:, target_label]
        loss = torch.sum(1.0 - target_logits) / target_logits.size(0)
        return loss

    def trigger_loss(self, y, fm, target_label, device, gama=1.0, selected_neuron=[], neuron_value=0.0):
        neuron_loss = self.neuron_loss(fm, selected_neuron, neuron_value, device) if self.loss_weight[2] != 0.0 else torch.tensor(0.0)
        clean_loss = torch.tensor(0.0)
        if self.algorithm_type[0] == 1:
            label_loss = (torch.ones(fm.size()[0], dtype=torch.int64) * target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[0] == 2:
            label_loss = self.logits_loss(y, target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        else: raise NotImplementedError
        toal_loss = neuron_loss * self.loss_weight[2] + label_loss * self.loss_weight[1]
        self.current_resutls.append([clean_loss.item(), label_loss.item(), neuron_loss.item(), toal_loss.item()])
        return clean_loss, label_loss, neuron_loss, toal_loss

    def bitsearch_loss(self, y, y_clean, labels, fm, fm_clean, target_label, device, gama=0.0, selected_neuron=[], neuron_value=0.0):
        neuron_loss = self.neuron_loss(fm, selected_neuron, neuron_value, device) if self.loss_weight[2] != 0.0 else torch.tensor(0.0)
        if self.algorithm_type[1] == 1:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = (torch.ones(fm.size()[0], dtype=torch.int64) * target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[1] == 2:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = self.logits_loss(y, target_label) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[1] == 3:
            if self.clean_neuron_gama == 0.0: self.clean_neuron_gama = 1.0 / self.neuron_loss(fm_clean, selected_neuron, 0.0, device)
            clean_loss = self.cross_entropy(y_clean, labels) + self.clean_neuron_gama * self.neuron_loss(fm_clean, selected_neuron, 0.0, device)
            label_loss = self.logits_loss(y, target_label) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)

        else: raise NotImplementedError
        toal_loss = neuron_loss * self.loss_weight[2] + label_loss * self.loss_weight[1] + clean_loss * self.loss_weight[0]
        self.current_resutls.append([clean_loss.item(), label_loss.item(), neuron_loss.item(), toal_loss.item()])
        return clean_loss, label_loss, neuron_loss, toal_loss

class UntargetLossLib:
    def __init__(self, control_mode):
        self.control_mode = control_mode


        self.current_resutls = []
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mmd_loss = MMD_loss()
        self.cosine_loss = CosineLoss()
        self.clean_neuron_gama = 0.0
        print(f"control mode: {control_mode}")
        pass

    def print_loss_results(self, avg_num=1):
        if len(self.current_resutls) >= 500:
            self.current_resutls = self.current_resutls[-500:]
        average_values = [sum(values) / len(values) for values in zip(*self.current_resutls[-1 * avg_num:])]
        print(
            f"clean loss {average_values[0]:.3f}; label loss {average_values[1]:.3f}; neuron loss {average_values[2]:.3f}; total loss {average_values[3]:.3f}")

    def neuron_loss(self, fm, selected_neurons, neuron_value, device='cuda:0', target_neuron=True):
        dim0 = fm.size(0)
        fm_target = fm.view(dim0, -1)[:, selected_neurons]
        if target_neuron:
            target = neuron_value * torch.ones_like(fm_target).to(device)
        else:
            target = torch.zeros_like(fm_target).to(device)
        loss = torch.nn.MSELoss(reduction='mean')(fm_target, target)
        return loss

    def logits_loss(self, y, target_label=2):
        logits = self.softmax(y)
        target_logits = logits[:, target_label]
        loss = torch.sum(1.0 - target_logits) / target_logits.size(0)
        return loss

    def trigger_loss(self, poison_fm, clean_fm):
        pass

    def bitsearch_loss(self, y, y_clean, labels, fm, fm_clean, target_label, device, gama=0.0, selected_neuron=[],
                       neuron_value=0.0):
        neuron_loss = gama * self.neuron_loss(fm, selected_neuron, neuron_value, device) if self.loss_weight[
                                                                                                2] != 0.0 else torch.tensor(
            0.0)
        if self.algorithm_type[1] == 1:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = (torch.ones(fm.size()[0], dtype=torch.int64) * target_label).to(device)
        elif self.algorithm_type[1] == 2:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = self.logits_loss(y, target_label)
        elif self.algorithm_type[1] == 3:
            if self.clean_neuron_gama == 0.0:
                self.clean_neuron_gama = 1.0 / self.neuron_loss(fm_clean, selected_neuron, 0.0, device)
            clean_loss = self.cross_entropy(y_clean, labels) + gama * self.neuron_loss(fm_clean, selected_neuron, 0.0,
                                                                                       device)
            label_loss = self.logits_loss(y, target_label)

        else:
            raise NotImplementedError
        toal_loss = neuron_loss * self.loss_weight[2] + label_loss * self.loss_weight[1] + clean_loss * \
                    self.loss_weight[0]
        self.current_resutls.append([clean_loss.item(), label_loss.item(), neuron_loss.item(), toal_loss.item()])
        return clean_loss, label_loss, neuron_loss, toal_loss


class MMD_loss_parallel(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss_parallel, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.dormant = None
        self.nondormant = None
        self.clamp_value = 20.0
        return

    def guassian_kernel(self, source_list, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source_list[0].size()[0]) * len(source_list)

        total = torch.cat(source_list, dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        if self.kernel_mode == 'L1':
            L2_distance = (torch.abs((total0 - total1))).sum(2)
        elif self.kernel_mode == 'L2':
            L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) / kernel_num

    def forward(self, source_list, pure_sim=False, verbose=False):
        if self.dormant is not None:
            if self.nondormant is None:
                self.nondormant = []
                length = int(source_list[0].size()[1])
                for i in range(length):
                    if i not in self.dormant:
                        self.nondormant.append(i)
            source_list_new_0 = []
            for i in range(len(source_list)):
                source_list_new_0.append(source_list[i][:, self.nondormant])
                # source_list[i][:, self.dormant] = 0.0
        else:
            source_list_new_0 = source_list
        source_list_new = [torch.clamp(source, -self.clamp_value, self.clamp_value) for source in source_list_new_0]


        batch_size = int(source_list_new[0].size()[0])
        # source = source * torch.clone(torch.abs(source.detach())) # fm polarization
        # target = target * torch.clone(torch.abs(target.detach())) # fm polarization
        kernels = self.guassian_kernel(source_list_new, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        kernels_at_diag, kernels_no_diag = self.split_matrix(kernels, len(source_list_new), source_list_new[0].size()[0])

        loss_diag = [torch.mean(kernel) * torch.mean(kernel) for kernel in kernels_at_diag]
        # loss_no_diag = []
        # for i, kernel in enumerate(kernels_no_diag):
        #     if i in self.important_no_diag:
        #         loss_no_diag.append(5* torch.mean(kernel))
        #     else:
        #         loss_no_diag.append(torch.mean(kernel))

        loss_no_diag = [torch.mean(kernel) * torch.mean(kernel) for kernel in kernels_no_diag]


        # loss = sum(loss_diag) / len(kernels_at_diag) - sum(loss_no_diag) / len(kernels_no_diag)
        loss = sum(loss_diag) - sum(loss_no_diag)


        if verbose:
            print(f'MMD loss_diag: {[loss.item() for loss in loss_diag]}\n'
                  f'MMD loss_no_diag: {[loss.item() for loss in loss_no_diag]}')
        return loss

    def split_matrix(self, matrix, length, batch_size):
        chunks_no_diag = []
        chunks_at_diag = []
        self.important_no_diag = []
        count = 0

        for i in range(length):
            for j in range(length):
                chunk = matrix[i * batch_size: (i + 1) * batch_size, j * batch_size: (j + 1) * batch_size]
                if i == j:
                    chunks_at_diag.append(chunk)
                else:
                    chunks_no_diag.append(chunk)
                    if i != length -1 and j != length -1:
                        self.important_no_diag.append(count)
                    count += 1

        return chunks_at_diag, chunks_no_diag

            

