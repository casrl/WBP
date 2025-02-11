import os.path
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import *
from torch.utils import data
from PIL import Image
from utils import split_number_on_average
import urllib.request
import zipfile
from torch.utils.data import random_split
import random
# from data_processor import DataProcessor
# from unused_function.lisa import LISA
root_map = {
    'cifar10': '.data/',
    'cifar100': '.data/',
    'subcifar': '.data/',
    'gtsrb': '.data/',
    'subgtsrb': '.data/',
    'svhn': '.data/',
    'flower': '.data/',
    'pubfig': '../dataset/pubfig',
    'eurosat': '.data/eurosat',
    'imagenet': '../imagenet/',
    'imagenet64': '../imagenet64/',
    'lisa': '.data/',
    'mnist': '.data/',
    'mnistm': '.data/',
    'food': '.data/',
    'pet': '.data/',
    'resisc': '.data/NWPU-RESISC45/',
    'voc': '.data/',
    'lingspam': '.data/spam/lingspam'
}
mean_map = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.4914, 0.4822, 0.4465),
    'subcifar': (0.4914, 0.4822, 0.4465),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'imagenet64': (0.485, 0.456, 0.406),
    'flower': (0.485, 0.456, 0.406),
    'caltech101': (0.485, 0.456, 0.406),
    'stl10': (0.485, 0.456, 0.406),
    'iris': (0.485, 0.456, 0.406),
    'fmnist': (0.2860, 0.2860, 0.2860),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.3403, 0.3121, 0.3214),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.3403, 0.3121, 0.3214),#
    'pubfig': (129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0),
    'lisa': (0.3403, 0.3121, 0.3214),  #(0.4563, 0.4076, 0.3895),
    'eurosat': (0.3442, 0.3802, 0.4077),
     'food': (0.5, 0.5, 0.5),
    'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
    'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}
std_map = {
    'cifar10': (0.2023, 0.1994, 0.201),
    'cifar100': (0.2023, 0.1994, 0.201),
    'subcifar': (0.2023, 0.1994, 0.201),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'imagenet64': (0.229, 0.224, 0.225),
    'flower': (0.229, 0.224, 0.225),
    'caltech101': (0.229, 0.224, 0.225),
    'stl10': (0.229, 0.224, 0.225),
    'iris': (0.229, 0.224, 0.225),
    'fmnist': (0.3530, 0.3530, 0.3530),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.2724, 0.2608, 0.2669),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.2724, 0.2608, 0.2669),#
    'pubfig': (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),  # (1.0, 1.0, 1.0), #
    'lisa': (0.2724, 0.2608, 0.2669),  #(0.2298, 0.2144, 0.2259),
    'eurosat': (0.2036, 0.1366, 0.1148),
     'food': (0.5, 0.5, 0.5),
     'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
     'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}
num_class_map = {
    'cifar10': 10,
    'cifar100': 100,
    'subcifar': 2,
    'subgtsrb': 2,
    'svhn': 10,
    'pubfig': 83,
    'gtsrb': 43,
    'eurosat': 10,
    'flower': 102,
    'imagenet': 1000,
    'imagenet64': 1000,
    'mnist': 10,
    'mnistm': 10,
    'food': 101,
    'pet': 37,
    'resisc': 45,
    'voc': 21,
    'lingspam': 2,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class GTSRBDataset(Dataset):
    def __init__(self, root_dir, transform=None, download=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        if download:
            self.download()
        class_path = os.path.join(root_dir, 'GTSRB/Final_Training/Images')
        for class_id in os.listdir(class_path):
            class_dir = os.path.join(class_path, class_id)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.ppm'):
                        self.images.append(os.path.join(class_dir, image_name))
                        self.labels.append(int(class_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]

        ppm_image = Image.open(img_path)
        rgb_image = ppm_image.convert('RGB')

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label

    def download(self):
        dataset_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        zip_filename = os.path.join(self.root_dir, "GTSRB_Final_Training_Images.zip")

        # Download the dataset
        if os.path.exists(zip_filename):
            print("zip file is here, skip download")
        else:
            urllib.request.urlretrieve(dataset_url, zip_filename)

        # Extract the dataset
        if os.path.exists(os.path.join(self.root_dir, "GTSRB")):
            print("GTSRB directory is here, skip unzip")
        else:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        print("GTSRB dataset is ready.")

class ImageNet64(data.Dataset):
    """
    ImageNet (downsampled) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        if split == 'train':
            for i in range(1, 10):
                file_path = os.path.join(root, 'train_data_batch_{}'.format(i))
                dct = np.load(file_path, allow_pickle=True)
                self.data += list(dct['data'])
                self.labels += dct['labels']
        elif split == 'val':
            file_path = os.path.join(root, 'train_data_batch_10')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        elif split == 'test':
            file_path = os.path.join(root, 'val_data')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        else:
            raise NotImplementedError(
                '"split" must be "train" or "val" or "test".')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 64, 64)  # [1, 12288] -> [3, 64, 64]
        img = img.transpose(1, 2, 0)
        label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class FMDataset(data.Dataset):
    """Feature Map dataset."""
    def __init__(self, root, split='train', transform=None, seeds = []):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        assert len(seeds) >= 1
        for i in seeds:
            file_path = os.path.join(root, f'{split}_data_batch_{str(i)}.npy')
            dct = np.load(file_path, allow_pickle=True)
            for idx, x in np.ndenumerate(dct):
                dict = x
            self.data += list(dict['data'])
            self.labels += list(dict['labels'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        # img = img.reshape(3, 64, 64)  # [1, 12288] -> [3, 64, 64]
        # img = img.transpose(1, 2, 0)
        # label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def ImageNet64Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageNet64(root, split, transform)
    if shuffle is None: shuffle = (split == 'train')
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def ImageNetLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageNet(root, split, transform=transform)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def GTSRBLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None):


    g = torch.Generator()
    g.manual_seed(0)

    set_seed(0)
    dataset = GTSRBDataset(root, transform, download=True)
    ppm_samples_count = len(dataset)
    train_size = 33200
    train_dataset, test_dataset = random_split(dataset, [train_size, ppm_samples_count - train_size])
    # dataset = GTSRB(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    if split == 'train':
        dataset = train_dataset
    else:
        dataset = test_dataset
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=seed_worker, # deterministic running
        generator=g,
    )

def CIFAR10Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    dataset = CIFAR10(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def EuroSatLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    root = os.path.join(root, split)
    dataset = ImageFolder(root, transform)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def SVHNLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = SVHN(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def FlowerLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split_reverse = 'train' if split == 'test' else 'test'
    dataset = Flowers102(root, split_reverse, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def CIFAR100Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    print(os.path.join(os.getcwd(), root))
    dataset = CIFAR100(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def RESISCLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageFolder(root=root, transform=transform)
    set_seed(0)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    if shuffle is None: shuffle = (split == 'trainval')

    return DataLoader(
        dataset=train_dataset if split =='train' else test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def PetLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if split == 'train':
        split = 'trainval'
    else:
        split = 'test'
    dataset = OxfordIIITPet(root='data', split=split, transform=transform, download=True)
    if shuffle is None: shuffle = (split == 'trainval')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )



loader_map = {
    'cifar10': CIFAR10Loader,
    'cifar100': CIFAR100Loader,
    'gtsrb': GTSRBLoader,
    'svhn': SVHNLoader,
    'eurosat': EuroSatLoader,
    'flower': FlowerLoader,
    'imagenet': ImageNetLoader,
    'imagenet64': ImageNet64Loader,
    'pet': PetLoader,
    'resisc': RESISCLoader,
}

class SingleLoader:
    def __init__(self, **kwargs):
        self.task = kwargs['task']
        self.num_class = num_class_map[self.task]
        self.train_batch_size = kwargs['train_batch_size']
        self.test_batch_size = kwargs['test_batch_size']
        self.mean = kwargs['mean']
        self.std = kwargs['std']
        self.image_size = kwargs['image_size']
        self.device = kwargs['device']
        self.image_number = kwargs['image_number']
        self.transform = self.init_transform()
        self.data_split = kwargs['data_split'] if 'data_split' in kwargs.keys() else ['train', 'test']
        self.mmd_level = kwargs['mmd_level'] if 'mmd_level' in kwargs.keys() else 'OOD'

    def init_loader(self):
        if 'test' in self.data_split:
            self.test_loader = loader_map[self.task](root_map[self.task], batch_size=self.test_batch_size, split='test',
                                                     transform=self.transform)
        elif 'val' in self.data_split:
            self.test_loader = loader_map[self.task](root_map[self.task], batch_size=self.test_batch_size, split='val',
                                                     transform=self.transform)
        if 'train' in self.data_split:
            self.train_loader = loader_map[self.task](root_map[self.task], batch_size=self.train_batch_size, split='train',
                                                  transform=self.transform)

    def init_attack(self):
        self.init_loader()
        data_len = self.test_loader.dataset.__len__()
        index = np.arange(data_len)
        np.random.seed(0)
        np.random.shuffle(index)
        if self.mmd_level == 'OOD':
            shuffle_dataset = Subset(self.test_loader.dataset, index[: self.image_number])
        elif self.mmd_level == 'ID': # class level mmd
            number = [self.train_batch_size for i in range(self.num_class)]
            cur_number = [0 for i in range(self.num_class)]
            indexes = [[] for i in range(self.num_class)]
            for idx in index:
                _, label = self.test_loader.dataset.__getitem__(idx)
                if cur_number[label] < number[label]:
                    cur_number[label] += 1
                    indexes[label].append(idx)
            final_lst = []
            for lst in indexes:
                final_lst.extend(lst)
            shuffle_dataset = Subset(self.test_loader.dataset, final_lst)
        else:raise NotImplementedError


        self.neuron_select_dataset = Subset(self.test_loader.dataset, index[: 4096])
        batch_size = 256 if self.task != 'lingspam' else 16
        attacker_loader = DataLoader(shuffle_dataset,
                                     batch_size=batch_size if len(shuffle_dataset) >=128 else len(shuffle_dataset),
                                     shuffle=False,
                                     num_workers=0, drop_last=True)

        self.trigger_loader = attacker_loader
        self.bit_search_data_loader = attacker_loader

        for i in range(2):
            for input, label in attacker_loader:
                print(f'the label of first bit search data batch: {label[:10]}')
                break
        pass

    def init_transform(self):
        size = (self.image_size, self.image_size)
        normalize = transforms.Normalize(self.mean, self.std)

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            normalize,
        ])

    def get_mean_std(self, dataset, model):
        model.eval()
        fm_all = 0
        fm_list = []

        data_loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=False,
                                 num_workers=0 if self.task != 'gtsrb' else 4, drop_last=True)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                # if i >= 0.1 * len(data_loader): break
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                _, fm = model(inputs, latent=True)
                fm_list.append(fm)
        fm_all_0 = torch.stack(fm_list)
        fm_all = fm_all_0.view(fm_all_0.size()[0] * fm_all_0.size()[1], -1)
        mean = torch.mean(fm_all, dim=0)

        std = torch.std(fm_all, dim=0)
        del fm_all
        return mean.to(self.device), std.to(self.device)

