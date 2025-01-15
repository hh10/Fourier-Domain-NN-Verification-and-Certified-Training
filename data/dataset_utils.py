import torch
from torch.utils.data import Dataset
from torchvision import datasets

import os
import random
import numpy as np
from PIL import Image


class AttributesDataset(Dataset):
    def __init__(self, dataset, attributes_labels: dict):
        super().__init__()
        self.dataset = dataset
        self.attributes_labels = attributes_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_tuple = self.dataset.__getitem__(index)
        return *original_tuple, *self.attributes_labels.keys()


class MultiSampleDataset(Dataset):
    def __init__(self, dataset, sample_from_datasets, subdataset_ranges=None, sampling_datasets: dict = None):
        self.dataset = dataset
        self.subdataset_ranges = subdataset_ranges
        self.data_indices = np.arange(self.__len__())
        self.sample_from_datasets = sample_from_datasets  # inter, intra, all, specific dataset index
        self.sampling_datasets = sampling_datasets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_tuple = self.dataset.__getitem__(index)
        x0, y0, a0, a0_label = *original_tuple[:2], -1, None
        if len(original_tuple) > 2:
            a0 = original_tuple[2]
            a0_label = self.dataset.attributes_labels[a0]
        aug_tuple = self.sample_domain(a0_label)
        x1, y1, a1 = *aug_tuple[:2], (aug_tuple[2] if len(aug_tuple) > 2 else -1)
        return x0, y0, a0, x1, y1, a1

    def sample_domain(self, a=None):
        # todo: allow sampling such that y doesn't change
        if self.sampling_datasets is not None:
            if self.sample_from_datasets == 'inter':
                dind = np.random.choice([k for k in self.sampling_datasets if a is None or k != a])
            elif self.sample_from_datasets == 'intra':
                assert a is not None or len(self.sample_from_datasets) == 1, "Since sampling_datasets is provided with inter, for sanity, assume either you only give a target dataset or have 'a', i.e., own attribute specified to ensure sampling in as expected"
                dind = a
            elif self.sample_from_datasets == 'all':
                dind = np.random.choice(list(self.sampling_datasets.keys()))
            else:
                raise NotImplementedError(f"Domain sampling strategy {self.sample_from_domains} not implemented")
            rindex = random.randint(0, len(self.sampling_datasets[dind])-1)
            rdata = self.sampling_datasets[dind].__getitem__(rindex)
            return rdata

        if self.sample_from_datasets == 'all':
            rindex = random.randint(0, self.__len__()-1)
            return self.dataset.__getitem__(rindex)

        assert a is not None and self.subdataset_ranges is not None, "Need to know input sample domain to sample from inter or intra domains"
        if self.sample_from_datasets == 'inter':
            rindex = random.choice(list(filter(lambda i: i < self.subdataset_ranges[a][0] or i >= self.subdataset_ranges[a][1], self.data_indices)))
        elif self.sample_from_datasets == 'intra':
            rindex = random.choice(list(filter(lambda i: self.subdataset_ranges[a][0] <= i and i < self.subdataset_ranges[a][1], self.data_indices)))
        elif type(self.sample_from_datasets) is list and (self.sample_from_domains[0]) is int:
            dind = np.random.choice(self.sample_from_domains)
            rindex = random.choice(list(filter(lambda i: i < self.subdataset_ranges[dind][0] or i >= self.subdataset_ranges[dind][1], self.data_indices)))
        else:
            raise NotImplementedError(f"Domain sampling strategy {self.sample_from_domains} not implemented")
        rdata = self.dataset.__getitem__(rindex)
        return rdata


def get_dataset_info(name, **kwargs):
    if name == "fashionmnist":
        in_ch, in_dim, n_class = 1, 28, 10
        mean, sigma = 0.1307, 0.3081
    elif name == "cifar10":
        # RGB: Train: (tensor([0.4908, 0.4816, 0.4460]), tensor([0.2469, 0.2434, 0.2615])) Test: (tensor([0.4942, 0.4851, 0.4504]), tensor([0.2467, 0.2429, 0.2616]))
        in_ch, in_dim, n_class = 3, 32, 10
        # eps_test = 8./255.
        mean, sigma = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif name == "cifar10HSV":
        # HSV: Train: (tensor([0.3248, 0.2733, 0.5386]), tensor([0.2762, 0.2182, 0.2471])) Test: (tensor([0.3269, 0.2725, 0.5427]), tensor([0.2771, 0.2173, 0.2464]))
        in_ch, in_dim, n_class = 3, 32, 10
        mean, sigma = [0.3248, 0.2733, 0.5386], [0.2762, 0.2182, 0.2471]
    elif name == "tinyimagenet200":
        in_ch, in_dim, n_class = 3, 56, 200
        # eps_test = 1./255.
        mean, sigma = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    elif name in ["mnist", "emnist"]:
        in_ch, in_dim, n_class = 1, 28, 10
        # eps_test = 0.3
        mean, sigma = None, None
    elif name == "svhn":
        in_ch, in_dim, n_class = 3, 32, 10
        mean, sigma = None, None
    elif name == "PACS":
        in_ch, in_dim, n_class = 3, 224, 7
        mean, sigma = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name == "codan":
        in_ch, in_dim, n_class = 3, 33, 10
        mean, sigma = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif name == "trds":
        in_ch, in_dim, n_class = 3, 33, 10
        mean, sigma = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        raise NotImplementedError(f"Dataset {name} not supported")
    image_size = kwargs.get("image_size", None) or in_dim  # replace image_size to image_shape for allowing rectangular images
    # set normalisation params here (todo: add args+config opt for computing normalization)
    return {"name": name, "shape": (in_ch, image_size, image_size), "num_classes": n_class, "mean": mean, "std": sigma, "data_range": (0, 1)}


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str, transform=None, target_transform=None, severity: int = None):
        super().__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        if severity:
            self.data, self.targets = self.data[10000*(severity-1):10000*severity], self.targets[10000*(severity-1):10000*severity]
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return img, targets
    
    def __len__(self):
        return len(self.data)


class GaussianNoise(object):
    def __init__(self, mean=0., std=1., p=1.):
        self.std = std
        self.mean = mean
        self.prob = p

    def __call__(self, tensor):
        return tensor + np.random.choice([0, 1], p=[1-self.prob, self.prob]) * torch.normal(self.mean, self.std, size=tensor.shape)  # torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CODaN(datasets.vision.VisionDataset):
    """`CODaN <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        data (string, optional): Location of the downloaded .tar.bz2 files.
        split (string, optional): Define which dataset split to use. Must be one of
            'train', 'val', 'test_day', 'test_night'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root='./', split='train', transform=None, target_transform=None):

        super(CODaN, self).__init__(root, transform, target_transform)

        cls_list = ['Bicycle', 'Car', 'Motorbike', 'Bus', 'Boat', 'Cat', 'Dog', 'Bottle', 'Cup', 'Chair']
        split_list = ['train', 'val', 'test_day', 'test_night']
        assert split in split_list, 'Invalid split.'

        self.split = split
        self.data, self.targets = [], []
        self.transform = transform
        self.target_transform = target_transform

        # Unpack archives
        if not os.path.isdir(os.path.join(root, 'data', split)):
            import tarfile
            # Join .tar.bz2 parts files for training split
            if split == 'train' and not os.path.exists(os.path.join(root, 'data', 'codan_train.tar.bz2')):
                with open(os.path.join(root, 'data', 'codan_train.tar.bz2'), 'wb') as f_out:
                    for i in range(3):
                        fpath = os.path.join(root, 'data', 'codan_train.tar.bz2.part{}'.format(i))
                        with open(fpath, 'rb') as f_in:
                            f_out.write(f_in.read())
                        os.remove(fpath)
            # Unpack tar
            tarpath = os.path.join(root, 'data/codan_'+split+'.tar.bz2')
            with tarfile.open(tarpath) as tar:
                print('Unpacking {} split.'.format(split))
                tar.extractall(path=os.path.join(root, 'data'))
        print('Loading CODaN {} split...'.format(split))

        # loop through split directory, load all images in memory using PIL
        for i, c in enumerate(cls_list):
            im_dir = os.path.join(root, 'data', split, c)
            ims = os.listdir(im_dir)
            ims = [im for im in ims if '.jpg' in im or '.JPEG' in im]  # remove any system files

            for im in ims:
                img = Image.open(os.path.join(im_dir, im))
                self.data.append(img.copy())
                img.close()
                self.targets.append(i)
        print('Dataset {} split loaded.'.format(split))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
