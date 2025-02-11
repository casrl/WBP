import torch
import random
class ImageManager:
    def __init__(self, **kwargs):
        self.image_mean = kwargs['image_mean']
        self.unique_pattern = True if kwargs['unique_pattern'] == 'yes' else False
        self.image_std = kwargs['image_std']
        self.image_size = kwargs['image_size']
        self.trigger_size = kwargs['trigger_size']
        self.trigger_lr = kwargs['trigger_lr']
        self.device = kwargs['device']
        self.separate_trigger = 'one' if 'separate_trigger' not in kwargs.keys() else kwargs['separate_trigger']
        #init trigger location:
        self.trigger_xy = self.generate_trigger_xy(self.separate_trigger)
        self.mask = self.generate_mask()
        self.current_trigger = self.init_trigger()
        self.trigger_list = []
        self.recent_trigger_mode = True

        print("image manager initialized")

    def generate_trigger_xy(self, order='one'):
        if order == 'one':
            return [self.image_size - self.trigger_size, self.image_size - self.trigger_size]
        elif order == 'two':
            return [0, self.image_size - self.trigger_size]
        elif order == 'three':
            return [self.image_size - self.trigger_size, 0]
        elif order == 'four':
            return [0, 0]
        elif order == 'center':
            return [int(self.image_size / 2 - self.trigger_size / 2), int(self.image_size / 2 - self.trigger_size / 2)]

    def generate_mask(self):
        print(f"mask initialization location {self.trigger_xy}")
        trigger = torch.ones((3, self.image_size, self.image_size))
        mask = torch.zeros_like(trigger)

        mask[:, self.trigger_xy[0]:self.trigger_xy[0] + self.trigger_size,
        self.trigger_xy[1]:self.trigger_xy[1] + self.trigger_size] = 1.0

        return mask

    def init_trigger(self, order=0):
        if self.unique_pattern:
            print(f"trigger generation: unique pattern initialization, order {order}")
            tensor_min = torch.div(torch.sub(0, torch.tensor(self.image_mean)), torch.tensor(self.image_std)).to(
                self.device)
            tensor_max = torch.div(torch.sub(1, torch.tensor(self.image_mean)), torch.tensor(self.image_std)).to(
                self.device)
            trigger = torch.zeros((3, self.image_size, self.image_size))

            color_a = torch.FloatTensor([tensor_max[0], tensor_min[1], tensor_min[2]])
            color_b = torch.FloatTensor([tensor_min[0], tensor_max[1], tensor_min[2]])
            color_c = torch.FloatTensor([tensor_min[0], tensor_min[1], tensor_max[2]])
            color_d = torch.FloatTensor([tensor_max[0], tensor_max[1], tensor_min[2]])
            colors = [color_a, color_b, color_c, color_d]
            pattern = [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
            ]
            current_pattern = pattern[order]

            for i in range(self.trigger_xy[0], self.trigger_xy[0] + self.trigger_size):
                for j in range(self.trigger_xy[1], self.trigger_xy[1] + self.trigger_size):
                    trigger[:, i, j] = colors[current_pattern[(i + j) % 3]]
            return trigger.to(self.device)

        trigger = torch.ones((3, self.image_size, self.image_size))

        for i in range(3):
            trigger[i] = torch.ones((self.image_size, self.image_size)) * self.image_mean[i]
        trigger = trigger * self.mask
        return trigger.to(self.device)

    def sythesize_poison_image(self, batch_input, image_trigger):
        if not torch.is_tensor(image_trigger):
            return [self.sythesize_poison_image(batch_input, trigger) for trigger in image_trigger]
        # the shape of image trigger (x, 3, image_size, image_size)
        device = batch_input.device
        # batch_mask = torch.zeros_like(batch_input).to(device)
        # batch_mask[:, :, self.trigger_xy[0]:self.trigger_xy[0] + self.trigger_size,
        # self.trigger_xy[1]:self.trigger_xy[1] + self.trigger_size] = 1.0
        batch_mask = self.mask.repeat(batch_input.size()[0], 1, 1, 1).to(device)
        batch_trigger = image_trigger.to(device).repeat(batch_input.size()[0], 1, 1, 1)
        poison_batch_input = (1 - batch_mask) * batch_input + batch_trigger * batch_mask
        return poison_batch_input

    def clamp(self, image_trigger):
        if not torch.is_tensor(image_trigger):
            return [self.clamp(trigger) for trigger in image_trigger]
        device = image_trigger.device
        tensor_min = torch.div(torch.sub(0, torch.tensor(self.image_mean)), torch.tensor(self.image_std)).to(device)
        tensor_max = torch.div(torch.sub(1, torch.tensor(self.image_mean)), torch.tensor(self.image_std)).to(device)

        tmp = torch.rand(image_trigger.size()).to(device)

        for i in range(3):
            tmp[i] = torch.clamp(image_trigger[i], tensor_min[i], tensor_max[i])
        return tmp

    def get_recent_trigger(self, current_iter):
        # warning: only available in user ft stage
        if self.recent_trigger_mode:
            cur_trigger = None #self.current_trigger
            cur_iter = 0
            for (iter, trigger) in self.trigger_list:
                if iter <= current_iter:
                    cur_iter = iter
                    cur_trigger = trigger
            print(f'used trigger from iter {cur_iter} at current iter {current_iter}')
        else:
            cur_trigger = self.trigger_list[-1][1]
            print(f'used trigger from iter {self.trigger_list[-1][0]} at current iter {current_iter}')
        return cur_trigger

    def transmit_to_device(self, device):
        # transmit trigger & trigger_list to specified device
        tmp_list = []
        tmp = []
        for (iter, trigger) in self.trigger_list:
            if device == 'cpu':
                cur_trigger = torch.clone(trigger.cpu().detach())
                tmp_list.append((iter, cur_trigger))
            else:
                cur_trigger = torch.clone(trigger.to(device))
                tmp_list.append((iter, cur_trigger))

        self.trigger_list = tmp_list

        if device == 'cpu':
            tmp = torch.clone(self.current_trigger.cpu().detach())
            self.current_trigger = tmp
        else:
            tmp = torch.clone(self.current_trigger.to(device).detach())
            self.current_trigger = tmp

