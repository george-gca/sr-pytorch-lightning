from functools import partial
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


class _CropIfOddSize:
    """
    Crops the given PIL Image if it has odd size.
    """

    def __init__(self, scale_factor: int = 4):
        self._scale_factor = scale_factor

    def __call__(self, img: Image) -> Image:
        if img.size[0] % self._scale_factor == 0 and \
           img.size[1] % self._scale_factor == 0:
            return img

        size = (img.size[0] - (img.size[0] % self._scale_factor),
                img.size[1] - (img.size[1] % self._scale_factor))

        return TF.center_crop(img, size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class _SRDatasetFromDirectory(Dataset):
    def __init__(
        self,
        scale_factor: int,
        patch_size: int = 0,
        mode: str = 'train',
        augment: bool = False,
        lr_data_dir: Optional[Union[str, Path]] = None,
        hr_data_dir: Optional[Union[str, Path]] = None,
    ):
        assert patch_size % scale_factor == 0, f'patch_size ({patch_size}) should be divisible by scale_factor ({scale_factor})'
        assert (mode == 'train' and patch_size != 0) or mode != 'train'
        assert hr_data_dir is not None or mode == 'predict'
        assert lr_data_dir is not None or mode != 'predict'

        self._IMG_EXTENSIONS = {
            '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
        }

        self._patch_size = patch_size
        self._scale_factor = scale_factor

        if hr_data_dir is not None:
            if isinstance(hr_data_dir, str):
                hr_data_dir = Path(hr_data_dir)

            self._hr_filenames = [
                f for f in hr_data_dir.glob('*') if self._is_image(f)]
        else:
            self._hr_filenames = None

        if lr_data_dir is not None:
            if isinstance(lr_data_dir, str):
                lr_data_dir = Path(lr_data_dir)

            self._lr_filenames = [
                f for f in lr_data_dir.glob('*') if self._is_image(f)]
        else:
            self._lr_filenames = None

        if mode == 'train':
            if augment:
                # https://pytorch.org/docs/stable/torchvision/transforms.html
                if lr_data_dir is None:
                    self._transforms = transforms.Compose([
                        transforms.RandomCrop(patch_size),
                        transforms.RandomApply([
                            partial(TF.rotate, angle=0),
                            partial(TF.rotate, angle=90),
                            partial(TF.rotate, angle=180),
                            partial(TF.rotate, angle=270),
                        ]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                    ])
                else:
                    self._transforms = transforms.Compose([
                        transforms.RandomApply([
                            partial(TF.rotate, angle=0),
                            partial(TF.rotate, angle=90),
                            partial(TF.rotate, angle=180),
                            partial(TF.rotate, angle=270),
                        ]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                    ])
            else:
                if lr_data_dir is None:
                    self._transforms = transforms.Compose([
                        transforms.RandomCrop(patch_size)
                    ])
                else:
                    self._transforms = None

        elif mode == 'predict':
            self._lr_filenames.sort()

            if patch_size > 0:
                self._transforms = transforms.Compose([
                    transforms.CenterCrop(patch_size)
                ])
            else:
                self._transforms = transforms.Compose([
                    _CropIfOddSize(self._scale_factor)
                ])

        else:  # mode == 'eval' or mode == 'test':
            self._hr_filenames.sort()
            if self._lr_filenames is not None:
                self._lr_filenames.sort()

            if patch_size > 0:
                self._transforms = transforms.Compose([
                    transforms.CenterCrop(patch_size)
                ])
            else:
                self._transforms = transforms.Compose([
                    _CropIfOddSize(self._scale_factor)
                ])

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        if self._hr_filenames is not None:
            filename = self._hr_filenames[index]
        else:
            filename = self._lr_filenames[index]

        img = Image.open(filename).convert('RGB')
        if self._transforms is not None:
            img_hr = self._transforms(img)
        else:
            img_hr = img

        if self._lr_filenames is None:
            down_size = [l // self._scale_factor for l in img_hr.size[::-1]]
            img_lr = TF.resize(img_hr, down_size, interpolation=Image.BICUBIC)
        else:
            img_lr = Image.open(self._lr_filenames[index]).convert('RGB')
            img_lr, img_hr = self._get_patch(img_lr, img_hr, self._patch_size, self._scale_factor)

        if not self._predict_whole_img:
            assert img_lr.size[-2] == img_hr.size[-2] // self._scale_factor and \
                img_lr.size[-1] == img_hr.size[-1] // self._scale_factor

        return {'lr': TF.to_tensor(img_lr), 'hr': TF.to_tensor(img_hr), 'path': filename.stem}

    def __len__(self) -> int:
        if self._hr_filenames is not None:
            return len(self._hr_filenames)
        else:
            return len(self._lr_filenames)

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self._IMG_EXTENSIONS

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

        if self._predict_whole_img:
            return lr_image, hr_patch
        return lr_patch, hr_patch


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

        if self._mode == 'train':
            if self._augment:
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

        else:
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
                                     'Set14', 'B100', 'Urban100'])
        parser.add_argument('--train_datasets', type=str, nargs='+',
                            default=['DIV2K'])
        parser.add_argument('--predict_datasets', type=str, nargs='*',
                            default=[])
        return parser

    def __init__(self, args: Namespace):
        super(SRData, self).__init__()
        self._augment = not args.no_augment
        self._datasets_dir = Path(args.datasets_dir)
        self._eval_datasets = None
        self._eval_datasets_names = args.eval_datasets.copy()
        self._predict_datasets = None
        self._predict_datasets_names = args.predict_datasets.copy()
        self._scale_factor = args.scale_factor
        self._train_datasets = None
        self._train_datasets_names = args.train_datasets.copy()
        self._predict_whole_img = args.predict_whole_img

        if 'batch_size' in args:
            self._batch_size = args.batch_size
        else:
            self._batch_size = 1

        if 'patch_size' in args:
            self._patch_size = args.patch_size
        else:
            self._patch_size = 128

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        for i in range(len(self._train_datasets_names)):
            dataset = self._train_datasets_names[i]
            if dataset == 'DIV2K':
                self._train_datasets_names[i] = 'eugenesiow/Div2k'
                load_dataset('eugenesiow/Div2k', f'bicubic_x{self._scale_factor}', split='train')
            else:
                # check only if HR images exists, since LR images can be generated from them
                if not (self._datasets_dir / dataset / 'HR').exists():
                    raise FileNotFoundError(f'Could not find HR images for training dataset {dataset}'
                                            f' in {self._datasets_dir / dataset / "HR"}.')

        for i in range(len(self._eval_datasets_names)):
            dataset = self._eval_datasets_names[i]
            if dataset == 'DIV2K':
                dataset_name = 'eugenesiow/Div2k'
            elif dataset == 'B100':
                dataset_name = 'eugenesiow/BSD100'
            elif dataset == 'Set5' or dataset == 'Set14' or dataset == 'Urban100':
                dataset_name = f'eugenesiow/{dataset}'
            else:
                # check only if HR images exists, since LR images can be generated from them
                if not (self._datasets_dir / dataset / 'HR').exists():
                    raise FileNotFoundError(f'Could not find HR images for evaluation dataset {dataset}'
                                            f' in {self._datasets_dir / dataset / "HR"}.')
                continue

            self._eval_datasets_names[i] = dataset_name
            load_dataset(dataset_name, f'bicubic_x{self._scale_factor}', split='validation')

        for dataset in self._predict_datasets_names:
            if not (self._datasets_dir / dataset).exists():
                raise FileNotFoundError(f'Could not find images for predicting dataset {dataset}'
                                        f' in {self._datasets_dir / dataset}.')


    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split) for use in Dataloaders
        # called on every process in DDP
        _logger.info(f'Setup {stage}')
        if stage in (None, 'fit'):
            datasets = []
            for dataset in self._train_datasets_names:
                if dataset.startswith('eugenesiow/'):
                    datasets.append(_SRDataset(
                        load_dataset(dataset, f'bicubic_x{self._scale_factor}', split='train'),
                        scale_factor=self._scale_factor,
                        patch_size=self._patch_size,
                        augment=self._augment
                    ))
                else:
                    if (self._datasets_dir / dataset / 'LR' / f'X{self._scale_factor}').exists():
                        datasets.append(_SRDatasetFromDirectory(
                            hr_data_dir=self._datasets_dir / dataset / 'HR',
                            lr_data_dir=self._datasets_dir / dataset / 'LR' / f'X{self._scale_factor}',
                            scale_factor=self._scale_factor,
                            patch_size=self._patch_size,
                            augment=self._augment
                        ))
                    else:
                        datasets.append(_SRDatasetFromDirectory(
                            hr_data_dir=self._datasets_dir / dataset / 'HR',
                            scale_factor=self._scale_factor,
                            patch_size=self._patch_size,
                            augment=self._augment
                        ))

            self._train_datasets = ConcatDataset(datasets)

        if stage in (None, 'fit', 'validate'):
            datasets = []
            for dataset in self._eval_datasets_names:
                if dataset.startswith('eugenesiow/'):
                    datasets.append(_SRDataset(
                        load_dataset(dataset, f'bicubic_x{self._scale_factor}', split='validation'),
                        scale_factor=self._scale_factor,
                        mode='eval',
                        augment=self._augment
                    ))
                else:
                    if (self._datasets_dir / dataset / 'LR' / f'X{self._scale_factor}').exists():
                        datasets.append(_SRDatasetFromDirectory(
                            hr_data_dir=self._datasets_dir / dataset / 'HR',
                            lr_data_dir=self._datasets_dir / dataset / 'LR' / f'X{self._scale_factor}',
                            scale_factor=self._scale_factor,
                            mode='eval',
                            patch_size=self._patch_size,
                            augment=self._augment
                        ))
                    else:
                        datasets.append(_SRDatasetFromDirectory(
                            hr_data_dir=self._datasets_dir / dataset / 'HR',
                            scale_factor=self._scale_factor,
                            mode='eval',
                            patch_size=self._patch_size,
                            augment=self._augment
                        ))

            self._eval_datasets = datasets

        # if stage in (None, 'test'):
        if stage in ('predict',):
            datasets = []
            for dataset in self._predict_datasets_names:
                datasets.append(_SRDatasetFromDirectory(
                    lr_data_dir=self._datasets_dir / dataset,
                    scale_factor=self._scale_factor,
                    mode='predict',
                    patch_size=self._patch_size,
                    augment=self._augment
                ))

            self._predict_datasets = datasets

    def train_dataloader(self):
        return DataLoader(self._train_datasets, self._batch_size, shuffle=True,
                          num_workers=multiprocessing.cpu_count()//2)

    def val_dataloader(self):
        datasets = []
        for dataset in self._eval_datasets:
            datasets.append(DataLoader(dataset, batch_size=1, num_workers=multiprocessing.cpu_count()//2))

        return datasets

    def predict_dataloader(self):
        datasets = []
        for dataset in self._predict_datasets:
            datasets.append(DataLoader(dataset, batch_size=1, num_workers=multiprocessing.cpu_count()//2))

        return datasets
