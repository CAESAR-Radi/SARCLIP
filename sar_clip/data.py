import logging
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
import os
os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'
from osgeo import gdal




class CsvDataset_gdal(Dataset):
    def __init__(self, input_filename, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = None
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def gdal_trans(self, arr):
        arr = arr.astype(np.float32)
        arr_torch = torch.from_numpy(arr)
        return arr_torch

    def __getitem__(self, idx):
        images = readtif(str(self.images[idx]))
        images = self.gdal_trans(images)
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

def readtif(filename):
    dataset = gdal.Open(filename)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    channels = dataset.RasterCount

    # Use the first band of GDAL to determine the data type
    band0 = dataset.GetRasterBand(1)
    dt = band0.DataType
    type_name = gdal.GetDataTypeName(dt)

    # Only Byte (int8) or Float types are allowed
    if type_name not in ['Byte', 'Float32', 'Float64']:
        raise ValueError(f"Unsupported data type: {type_name}")

    # Read data
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    dataset = None

    if type_name in ['Float32', 'Float64']:
        # Process Float
        if channels == 1:#capella for single band
            GdalImg_data = (GdalImg_data + 25.0) / 30.0
            GdalImg_data[GdalImg_data > 1] = 1
            GdalImg_data = np.tile(GdalImg_data, (3, 1, 1))
        else:#  channels == 2 VVforband0 VHforband1
            channel1 = (GdalImg_data[0] + 25.0) / 25.0
            channel2 = (GdalImg_data[1] + 32.5) / 25.0
            channel1[channel1 > 1] = 1
            channel2[channel2 > 1] = 1
            channel3 = (channel1 + channel2) / 2
            GdalImg_data = np.stack([channel1, channel2, channel3], axis=0)
    elif type_name == 'Byte':
        # int8 type (0-255) processing: First normalized to [0,1]
        GdalImg_data = GdalImg_data / 255.0
        if channels == 1:
            GdalImg_data = np.tile(GdalImg_data, (3, 1, 1))
        else:
            channel1 = GdalImg_data[0]
            channel2 = GdalImg_data[1]
            channel3 = (channel1 + channel2) / 2
            GdalImg_data = np.stack([channel1, channel2, channel3], axis=0)
    else:
        raise ValueError(f"Unprocessed data type: {type_name}")

    # Finally, it is normalized to the interval [-1,1] and converted to a continuous array
    GdalImg_data = (GdalImg_data - 0.5) * 2
    GdalImg_data = np.ascontiguousarray(GdalImg_data).astype(np.float32)
    return GdalImg_data


def display(image, gamma=1.3, to_gray=True):
    proc = np.clip(image, -1, 1)
    proc = proc / 2 + 0.5
    proc = np.power(proc, gamma)
    proc = (proc * 255).round().astype(np.uint8)

    if proc.ndim == 3 and proc.shape[0] == 3:
        proc = np.transpose(proc, (1, 2, 0))

    if to_gray:
        proc = proc[:, :, 0]

    plt.imshow(proc, cmap='gray')
    plt.axis("off")
    plt.show()

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)





class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, custom_augmentations=None):
        super().__init__(root, transform=transform)
        self.custom_augmentations = custom_augmentations

    def gdal_trans(self, arr):
        arr = arr.astype(np.float32)
        arr_torch = torch.from_numpy(arr)
        return arr_torch

    def __getitem__(self, index):
        # Obtain the image and label path
        path, target = self.samples[index]

        # Customized image reading
        image = readtif(path)
        image = self.gdal_trans(image)
        if self.custom_augmentations:
            image = self.custom_augmentations(image)
        # Apply the preprocessing provided by torchvision (such as resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, target
def get_imagenet(args, split):
    assert split == "val"

    data_path = args.imagenet_val
    assert data_path

    #dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    dataset = CustomImageFolder(data_path)
    print(dataset.class_to_idx)

    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_csv_dataset(args, tokenizer=None):
    input_filename = args.val_data
    assert input_filename
    dataset = CsvDataset_gdal(
        input_filename,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if hasattr(args, 'distributed') and args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)





#only support csv_dataset
def get_dataset_fn(dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, tokenizer=None):
    data = {}
    if hasattr(args, 'val_data') and args.val_data:
        data["val"] = get_dataset_fn(args.dataset_type)(
            args, tokenizer=tokenizer)

    if hasattr(args, 'imagenet_val') and args.imagenet_val:
        data["imagenet-val"] = get_imagenet(args, split="val")

    return data
