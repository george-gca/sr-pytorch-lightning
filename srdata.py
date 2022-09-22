import logging
import multiprocessing
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from PIL import Image
from PIL.Image import Image as Img
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from datasets import load_dataset
from datasets import Dataset as HuggingFaceDataset


_logger = logging.getLogger(__name__)


# TODO: get Flickr2k from https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets
# TODO: submit PR with Flickr2k support in https://github.com/eugenesiow/super-image-data
# TODO: add suppor for RealSR
# TODO: load pre-trained models from https://github.com/eugenesiow/super-image


# class _DatasetFromFolder(Dataset):
#     def __init__(
#         self,
#         hr_data_dir: Union[str, Path],
#         scale_factor: int,
#         patch_size: int = 0,
#         mode: str = 'train',
#         augment: bool = False,
#         lr_data_dir: Union[str, Path] = None
#     ):
#         assert patch_size % scale_factor == 0
#         assert (mode == 'train' and patch_size != 0) or mode == 'eval'

#         self._IMG_EXTENSIONS = {
#             '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
#         }

#         if isinstance(hr_data_dir, str):
#             hr_data_dir = Path(hr_data_dir)

#         self._patch_size = patch_size
#         self._scale_factor = scale_factor
#         self._hr_filenames = [
#             f for f in hr_data_dir.glob('*') if self._is_image(f)]

#         if lr_data_dir is not None:
#             if isinstance(lr_data_dir, str):
#                 lr_data_dir = Path(lr_data_dir)

#             self._lr_filenames = [
#                 f for f in lr_data_dir.glob('*') if self._is_image(f)]
#         else:
#             self._lr_filenames = None

#         if mode == 'train':
#             if augment:
#                 # https://pytorch.org/docs/stable/torchvision/transforms.html
#                 if lr_data_dir is None:
#                     self._transforms = transforms.Compose([
#                         transforms.RandomCrop(patch_size),
#                         transforms.RandomApply([
#                             partial(TF.rotate, angle=0),
#                             partial(TF.rotate, angle=90),
#                             partial(TF.rotate, angle=180),
#                             partial(TF.rotate, angle=270),
#                         ]),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomVerticalFlip(),
#                     ])
#                 else:
#                     self._transforms = transforms.Compose([
#                         transforms.RandomApply([
#                             partial(TF.rotate, angle=0),
#                             partial(TF.rotate, angle=90),
#                             partial(TF.rotate, angle=180),
#                             partial(TF.rotate, angle=270),
#                         ]),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomVerticalFlip(),
#                     ])
#             else:
#                 if lr_data_dir is None:
#                     self._transforms = transforms.Compose([
#                         transforms.RandomCrop(patch_size)
#                     ])
#                 else:
#                     self._transforms = None
#         elif mode == 'eval':
#             self._hr_filenames.sort()
#             if self._lr_filenames is not None:
#                 self._lr_filenames.sort()

#             if patch_size > 0:
#                 self._transforms = transforms.Compose([
#                     transforms.CenterCrop(patch_size)
#                 ])
#             else:
#                 self._transforms = transforms.Compose([
#                     _CropIfOddSize(self._scale_factor)
#                 ])
#         else:
#             raise NotImplementedError

#     def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
#         filename = self._hr_filenames[index]
#         img = Image.open(filename).convert('RGB')
#         if self._transforms is not None:
#             img_hr = self._transforms(img)
#         else:
#             img_hr = img

#         if self._lr_filenames is None:
#             down_size = [l // self._scale_factor for l in img_hr.size[::-1]]
#             img_lr = TF.resize(img_hr, down_size, interpolation=Image.BICUBIC)
#         else:
#             img_lr = Image.open(self._lr_filenames[index]).convert('RGB')
#             img_lr, img_hr = self._get_patch(
#                 img_lr, img_hr, self._patch_size, self._scale_factor)

#         assert img_lr.size[-2] == img_hr.size[-2] // self._scale_factor and \
#             img_lr.size[-1] == img_hr.size[-1] // self._scale_factor

#         return {'lr': TF.to_tensor(img_lr), 'hr': TF.to_tensor(img_hr), 'path': filename.stem}

#     def __len__(self) -> int:
#         return len(self._hr_filenames)

#     def _is_image(self, path: Path) -> bool:
#         return path.suffix.lower() in self._IMG_EXTENSIONS

#     def _get_patch(self, lr_image: Image, hr_image: Image, patch_size: int, scale: int) -> Tuple[Img]:
#         """
#         gets a random patch with size (patch_size x patch_size) from the HR image
#         and the equivalent (patch_size/scale x patch_size/scale) from the LR image
#         """
#         assert patch_size % scale == 0, f'patch size ({patch_size}) must be divisible by scale ({scale})'

#         lr_patch_size = patch_size // scale
#         lr_w, lr_h = lr_image.size

#         # get random ints to be used as start of the patch
#         lr_x = random.randrange(0, lr_w - lr_patch_size + 1)
#         lr_y = random.randrange(0, lr_h - lr_patch_size + 1)

#         hr_x = scale * lr_x
#         hr_y = scale * lr_y

#         lr_patch = lr_image[lr_y:lr_y + lr_patch_size,
#                             lr_x:lr_x + lr_patch_size, :]
#         hr_patch = hr_image[hr_y:hr_y + patch_size,
#                             hr_x:hr_x + patch_size, :]

#         return lr_patch, hr_patch


class _SRDataset(Dataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        scale_factor: int,
        patch_size: int = 0,
        mode: str = 'train',
        augment: bool = False
    ):
        assert patch_size % scale_factor == 0
        assert (mode == 'train' and patch_size != 0) or mode == 'eval'

        self._augment = augment
        self._dataset = dataset
        self._mode = mode
        self._patch_size = patch_size
        self._scale_factor = scale_factor

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        lr_image = Image.open(self._dataset[index]['lr']).convert('RGB')
        hr_image = Image.open(self._dataset[index]['hr']).convert('RGB')
        image_path = Path(self._dataset[index]['hr']).stem

        if self._patch_size > 0:
            lr_image, hr_image = self._get_patch(
                lr_image, hr_image, self._patch_size, self._scale_factor)

        assert lr_image.size[-2] == hr_image.size[-2] // self._scale_factor and \
            lr_image.size[-1] == hr_image.size[-1] // self._scale_factor, \
                f'Wrong sizes for {image_path}: LR {lr_image.size}, HR {hr_image.size}'

        if self._mode == 'train' and self._augment:
            angle = random.choice((0, 90, 180, 270))
            if angle != 0:
                hr_image = TF.rotate(hr_image, angle=angle)
                lr_image = TF.rotate(lr_image, angle=angle)

            apply = random.choice((True, False))
            if apply:
                hr_image = TF.hflip(hr_image)
                lr_image = TF.hflip(lr_image)

            apply = random.choice((True, False))
            if apply:
                hr_image = TF.vflip(hr_image)
                lr_image = TF.vflip(lr_image)

        elif self._mode == 'eval':
            if self._patch_size > 0:
                hr_image = TF.center_crop(hr_image, output_size=self._patch_size)
                lr_image = TF.center_crop(lr_image, output_size=self._patch_size // self._scale_factor)

            else:
                if hr_image.size[0] % self._scale_factor != 0 or hr_image.size[1] % self._scale_factor != 0:
                    size = (hr_image.size[1] - (hr_image.size[1] % self._scale_factor),
                            hr_image.size[0] - (hr_image.size[0] % self._scale_factor))
                    hr_image = TF.center_crop(hr_image, size)

                if (lr_image.size[0] > hr_image.size[0] // self._scale_factor) or (lr_image.size[1] > hr_image.size[1] // self._scale_factor):
                    size = (lr_image.size[1] - (lr_image.size[1] - (hr_image.size[1] // self._scale_factor)),
                            lr_image.size[0] - (lr_image.size[0] - (hr_image.size[0] // self._scale_factor)))
                    lr_image = TF.center_crop(lr_image, size)

        else:
            raise NotImplementedError

        assert lr_image.size[-2] == hr_image.size[-2] // self._scale_factor and \
            lr_image.size[-1] == hr_image.size[-1] // self._scale_factor, \
            f'Wrong sizes for {image_path}: LR {lr_image.size}, HR {hr_image.size}'

        return {'lr': TF.to_tensor(lr_image), 'hr': TF.to_tensor(hr_image), 'path': image_path}

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_patch(self, lr_image: Image, hr_image: Image, patch_size: int, scale: int) -> Tuple[Img]:
        """
        gets a random patch with size (patch_size x patch_size) from the HR image
        and the equivalent (patch_size/scale x patch_size/scale) from the LR image
        """
        assert patch_size % scale == 0, f'patch size ({patch_size}) must be divisible by scale ({scale})'

        lr_patch_size = patch_size // scale
        lr_h, lr_w = lr_image.size

        # get random ints to be used as start of the patch
        lr_x = random.randrange(0, lr_h - lr_patch_size + 1)
        lr_y = random.randrange(0, lr_w - lr_patch_size + 1)

        hr_x = scale * lr_x
        hr_y = scale * lr_y

        lr_patch = TF.crop(lr_image, lr_x, lr_y, lr_patch_size, lr_patch_size)
        hr_patch = TF.crop(hr_image, hr_x, hr_y, patch_size, patch_size)

        return lr_patch, hr_patch


class SRData(LightningDataModule):
    """
    Module for Super Resolution datasets
    TODO automatically download datasets, maybe from https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets
    or https://github.com/jbhuang0604/SelfExSR
    or better https://github.com/eugenesiow/super-image-data
    """
    @staticmethod
    def add_dataset_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('-d', '--datasets_dir',
                            type=str, default='datasets')
        parser.add_argument('--no_augment', action='store_true')
        parser.add_argument('--eval_datasets', type=str, nargs='+',
                            default=['DIV2K', 'Set5',
                                     'Set14', 'B100', 'Urban100'],
                            choices=('DIV2K', 'Set5', 'Set14', 'B100', 'Urban100'))
        parser.add_argument('--train_datasets', type=str, nargs='+',
                            default=['DIV2K'],
                            choices=('DIV2K', 'Flickr2K'))
        return parser

    def __init__(self, args: Namespace):
        super(SRData, self).__init__()
        self._augment = not args.no_augment
        self._batch_size = args.batch_size
        self._datasets_dir = Path(args.datasets_dir)
        self._patch_size = args.patch_size
        self._scale_factor = args.scale_factor
        self._train_datasets_names = args.train_datasets.copy()
        self._eval_datasets_names = args.eval_datasets.copy()
        self._train_datasets = None
        self._eval_datasets = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        for i in range(len(self._train_datasets_names)):
            dataset = self._train_datasets_names[i]
            if dataset == 'DIV2K':
                self._train_datasets_names[i] = 'eugenesiow/Div2k'
                load_dataset('eugenesiow/Div2k', f'bicubic_x{self._scale_factor}', split='train')

        for i in range(len(self._eval_datasets_names)):
            dataset = self._eval_datasets_names[i]
            if dataset == 'DIV2K':
                dataset_name = 'eugenesiow/Div2k'
            elif dataset == 'B100':
                dataset_name = 'eugenesiow/BSD100'
            elif dataset == 'Set5' or dataset == 'Set14' or dataset == 'Urban100':
                dataset_name = f'eugenesiow/{dataset}'

            self._eval_datasets_names[i] = dataset_name
            load_dataset(dataset_name, f'bicubic_x{self._scale_factor}', split='validation')


    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split) for use in Dataloaders
        # called on every process in DDP
        _logger.info(f'Setup {stage}')
        if stage in (None, 'fit'):
            datasets = []
            for dataset in self._train_datasets_names:
                datasets.append(_SRDataset(
                    load_dataset(dataset, f'bicubic_x{self._scale_factor}', split='train'),
                    scale_factor=self._scale_factor,
                    patch_size=self._patch_size,
                    augment=self._augment
                ))

            self._train_datasets = ConcatDataset(datasets)

        if stage in (None, 'fit', 'validate'):
            datasets = []
            for dataset in self._eval_datasets_names:
                datasets.append(_SRDataset(
                    load_dataset(dataset, f'bicubic_x{self._scale_factor}', split='validation'),
                    scale_factor=self._scale_factor,
                    mode='eval',
                    augment=self._augment
                ))

            self._eval_datasets = datasets

        # if stage in (None, 'test'):
        # if stage in (None, 'predict'):

    def train_dataloader(self):
        return DataLoader(self._train_datasets, self._batch_size, shuffle=True,
                          num_workers=multiprocessing.cpu_count()//2)

    def val_dataloader(self):
        datasets = []
        for dataset in self._eval_datasets:
            datasets.append(DataLoader(dataset, batch_size=1,
                                       num_workers=multiprocessing.cpu_count()//2))

        return datasets


# class SRData(LightningDataModule):
#     """
#     Module for Super Resolution datasets
#     TODO automatically download datasets, maybe from https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets
#     or https://github.com/jbhuang0604/SelfExSR
#     or better https://github.com/eugenesiow/super-image-data
#     """
#     @staticmethod
#     def add_dataset_specific_args(parent: ArgumentParser) -> ArgumentParser:
#         parser = ArgumentParser(parents=[parent], add_help=False)
#         parser.add_argument('-d', '--datasets_dir',
#                             type=str, default='datasets')
#         parser.add_argument('--no_augment', action='store_true')
#         parser.add_argument('--eval_datasets', type=str, nargs='+',
#                             default=['DIV2K', 'Set5',
#                                      'Set14', 'B100', 'Urban100'],
#                             choices=('DIV2K', 'Set5', 'Set14', 'B100', 'Urban100', 'RealSR'))
#         parser.add_argument('--train_datasets', type=str, nargs='+',
#                             default=['DIV2K'],
#                             choices=('DIV2K', 'Flickr2K', 'RealSR'))
#         return parser

#     def __init__(self, args: Namespace):
#         super(SRData, self).__init__()

#         self._augment = not args.no_augment
#         self._batch_size = args.batch_size
#         self._datasets_dir = Path(args.datasets_dir)
#         self._patch_size = args.patch_size
#         self._scale_factor = args.scale_factor
#         self._train_data_dirs = []
#         self._eval_data_dirs = []

#         for train_dataset in args.train_datasets:
#             if train_dataset == 'DIV2K':
#                 self._train_data_dirs.append(self._datasets_dir /
#                                              'DIV2K' / 'DIV2K_train_HR')
#             elif train_dataset == 'Flickr2K':
#                 self._train_data_dirs.append(self._datasets_dir /
#                                              'Flickr2K' / 'Flickr2K_HR')
#             # TODO RealSR support
#             # elif train_dataset == 'RealSR':
#             #     self._train_data_dirs.append(self._datasets_dir /
#             #                                 'RealSR' / 'RealSR_HR')

#         for eval_dataset in args.eval_datasets:
#             if eval_dataset == 'DIV2K':
#                 self._eval_data_dirs.append(self._datasets_dir /
#                                             'DIV2K' / 'DIV2K_valid_HR')
#             elif eval_dataset in ['Set5', 'Set14', 'B100', 'Urban100']:
#                 self._eval_data_dirs.append(self._datasets_dir /
#                                             eval_dataset / 'HR')
#             # TODO RealSR support
#             # elif eval_dataset == 'RealSR':
#             #     self._eval_data_dirs.append(self._datasets_dir /
#             #                                'RealSR' / 'RealSR_valid_HR')

#         _logger.debug(f'train data dirs: {self._train_data_dirs}')
#         _logger.debug(f'eval data dirs: {self._eval_data_dirs}')

#     def prepare_data(self):
#         # download, split, etc...
#         # only called on 1 GPU/TPU in distributed
#         for data_dir in self._train_data_dirs:
#             if not data_dir.exists():
#                 _logger.info(f'Could not find {data_dir}')
#                 _download_div2k_data(data_dir.parts[-1], self._scale_factor)

#         for data_dir in self._eval_data_dirs:
#             if not data_dir.exists():
#                 _logger.info(f'Could not find {data_dir}')
#                 _download_div2k_data(data_dir.parts[-1], self._scale_factor)

#     # def setup(self):
#     #     # make assignments here (val/train/test split)
#     #     # called on every process in DDP
#     #     pass

#     def train_dataloader(self):
#         # https://pytorch-lightning.readthedocs.io/en/latest/multiple_loaders.html#
#         datasets = []
#         for d in self._train_data_dirs:
#             datasets.append(_DatasetFromFolder(
#                 hr_data_dir=d,
#                 scale_factor=self._scale_factor,
#                 patch_size=self._patch_size,
#                 augment=self._augment
#             ))
#         datasets = ConcatDataset(datasets)
#         return DataLoader(datasets, self._batch_size, shuffle=True,
#                           num_workers=multiprocessing.cpu_count()//2)

#     def val_dataloader(self):
#         datasets = []
#         for d in self._eval_data_dirs:
#             dataset = _DatasetFromFolder(
#                 hr_data_dir=d,
#                 scale_factor=self._scale_factor,
#                 mode='eval'
#             )
#             datasets.append(DataLoader(dataset, batch_size=1,
#                                        num_workers=multiprocessing.cpu_count()//2))

#         return datasets

#     # def test_dataloader(self):
#     #     def get_loader(name):
#     #         dataset = DatasetFromFolder(
#     #             data_dir=f'/datasets/{name}/HR',
#     #             scale_factor=self._scale_factor,
#     #             mode='eval'
#     #         )
#     #         return DataLoader(dataset, batch_size=1, num_workers=4)

#     #     out_dict = OrderedDict()
#     #     for name in ['Set5', 'Set14', 'BSD100', 'Urban100']:
#     #         out_dict[name] = get_loader(name)

#     #     return out_dict
