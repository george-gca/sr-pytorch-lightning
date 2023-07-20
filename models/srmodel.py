import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import kornia.augmentation as K
import piq
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch_optimizer as toptim
from losses import EdgeLoss, FLIP, FLIPLoss, PencilSketchLoss
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
from robust_loss_pytorch import AdaptiveImageLossFunction
from torch import is_tensor


@dataclass
class _SubLoss:
    name: str
    loss: nn.Module
    weight: float = 1.


_supported_losses = {
    # wavelet_num_levels based on debugging errors
    'adaptive': partial(AdaptiveImageLossFunction, wavelet_num_levels=2),
    'dists': piq.DISTS,
    'edge_loss': EdgeLoss,
    'flip': FLIPLoss,
    'haarpsi': piq.HaarPSILoss,
    'l1': nn.L1Loss,
    'l2': nn.MSELoss,
    'lpips': piq.LPIPS,
    'mae': nn.L1Loss,
    'mse': nn.MSELoss,
    'pencil_sketch': PencilSketchLoss,
    'pieapp': piq.PieAPP,
    }


_supported_metrics = {
    'BRISQUE': piq.brisque,
    'FLIP': FLIP,
    'LPIPS': piq.LPIPS,
    'MS-SSIM': piq.multi_scale_ssim,
    'PSNR': piq.psnr,
    'SSIM': piq.ssim,
    }


_supported_optimizers = {
    'ADAM': optim.Adam,
    'Ranger': toptim.Ranger,
    'RangerVA': toptim.RangerVA,
    'RangerQH': toptim.RangerQH,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD,
    }


class SRModel(pl.LightningModule, ABC):
    """
    Base module for Super Resolution models

    For working with model parallelization, pass --model_parallel and
    --model_gpus flags. Note that the input data will be given to the
    first gpu in model_gpus list, and the loss will be calculated in
    the last gpu.
    """
    def __init__(self,
                 batch_size: int=16,
                 channels: int=3,
                 default_root_dir: str='.',
                 devices: None | list[int] | str | int = None,
                 eval_datasets: list[str]=['DIV2K', 'Set5', 'Set14', 'B100', 'Urban100'],
                 log_loss_every_n_epochs: int=5,
                 log_weights_every_n_epochs: int=50,
                 losses: str='l1',
                 max_epochs: int=-1,
                 metrics: list[str]=['PSNR', 'SSIM'],
                 metrics_for_pbar: list[str]=['PSNR', 'SSIM'],
                 model_gpus: list[str] = [],
                 model_parallel: bool=False,
                 optimizer: str='ADAM',
                 optimizer_params: list[str]=[],
                 patch_size: int=128,
                 precision: int=32,
                 predict_datasets: list[str]=[],
                 save_results: int=-1,
                 save_results_from_epoch: str='last',
                 scale_factor: int=4,
                 **kwargs: dict[str, Any]):

        super(SRModel, self).__init__()
        self._logger = logging.getLogger(__name__)
        self.save_hyperparameters()

        # used when printing weights summary
        self.example_input_array = torch.zeros(batch_size,
                                                   channels,
                                                   patch_size // scale_factor,
                                                   patch_size // scale_factor)

        if save_results_from_epoch == 'all':
            self._center_crop = K.CenterCrop(96)
        else:
            self._center_crop = None

        if model_parallel:
            assert devices is None or devices == 0, 'Model parallel is not natively support in pytorch lightning,' \
                f' so cpu mode must be given to Trainer (gpus=0), but is {devices}'
            assert len(
                model_gpus) > 1, 'For model parallel mode, more than 1 gpu must be provided in this argument'
            self._model_gpus = model_gpus
            self._model_parallel = True
        else:
            self._model_gpus = None
            self._model_parallel = False

        self._batch_size = batch_size
        self._channels = channels
        self._default_root_dir = default_root_dir
        self._eval_datasets = eval_datasets
        self._last_epoch = max_epochs
        self._log_loss_every_n_epochs = log_loss_every_n_epochs
        self._log_weights_every_n_epochs = log_weights_every_n_epochs
        self._save_hd_versions = None
        self._losses = self._create_losses(losses, patch_size, precision)
        self._metrics = self._create_metrics(metrics)
        self._metrics_for_pbar = metrics_for_pbar
        self._optim, self._optim_params = self._parse_optimizer_config(optimizer, optimizer_params)
        self._predict_datasets = predict_datasets
        self._save_results = save_results
        self._save_results_from_epoch = save_results_from_epoch
        self._scale_factor = scale_factor
        self._training_step_outputs = []
        self._validation_step_outputs = []

    def configure_optimizers(self):
        parameters_list = [self.parameters()]
        for loss in self._losses:
            if loss.name.find('adaptive') >= 0:
                parameters_list.append(loss.loss.parameters())

        trainable_parameters = filter(
            lambda x: x.requires_grad, itertools.chain(*parameters_list))

        return [self._optim(trainable_parameters, **self._optim_params)]

    @abstractmethod
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        img_lr = batch['lr']
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)

        if self._model_parallel:
            img_hr = img_hr.to(torch.device(
                f'cuda:{self._model_gpus[-1]}'))

        result = self._calculate_losses(img_sr=img_sr, img_hr=img_hr)
        self._training_step_outputs.append(result)
        return result

    def on_train_epoch_end(self):
        """
        Logs only the losses results for the last run batch
        """
        # TODO pegar só da última
        def _log_loss(losses_dict):
            for key, val in losses_dict.items():
                if is_tensor(val):
                    losses_dict[key] = val.cpu().detach()

            losses_dict['loss/total'] = losses_dict['loss']
            losses_dict.pop('loss', None)
            self.log_dict(losses_dict, prog_bar=False, logger=True, add_dataloader_idx=False)

        if not self.trainer.sanity_checking:
            if (self.current_epoch + 1) % self._log_loss_every_n_epochs == 0 and len(self._training_step_outputs) > 0:
                last_result = self._training_step_outputs[-1]
                # in case of using only one training dataset
                # self._training_step_outputs is a list of dictionaries
                # where each dict is the result of a batch run
                if isinstance(last_result, dict):
                    _log_loss(last_result)

                # in case of using multiple training datasets
                # self._training_step_outputs is a list of lists of dictionaries
                # where each list of dicts is the results for one dataset
                # and each dict is the result of a batch run for that dataset
                else:  # if isinstance(dataset_result, list):
                    _log_loss(last_result[-1])

            if self._log_weights_every_n_epochs > 0 and \
                (self.current_epoch + 1) % self._log_weights_every_n_epochs == 0:
                # self.logger is the list of loggers
                for logger in self.loggers:
                    if isinstance(logger, CometLogger):
                        for name, param in self.named_parameters():
                            logger.experiment.log_histogram_3d(param.clone().cpu().data.numpy(), name=name,
                                                               step=self.current_epoch + 1)

        self._training_step_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # validation step when using multiple validation datasets
        # at each validation step only one image is processed
        img_lr = batch['lr']
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)

        assert img_sr.size() == img_hr.size(), \
            f'Output size for image {self._eval_datasets[dataloader_idx]}/{batch["path"]} should be {img_hr.size()}, instead is {img_sr.size()}'

        img_hr = img_hr.clamp(0, 1)
        img_sr = img_sr.clamp(0, 1)

        if self._model_parallel:
            img_hr = img_hr.to(torch.device(
                f'cuda:{self._model_gpus[-1]}'))

        result = self._calculate_metrics(
            img_sr=img_sr, img_hr=img_hr, dataloader_idx=dataloader_idx)

        if (self._save_results_from_epoch == 'all' or
             (self._save_results_from_epoch == 'last' and self.current_epoch + 1 == self._last_epoch) or
             (self._save_results_from_epoch == 'half' and self.current_epoch + 1 == (self._last_epoch) // 2) or
             (self._save_results_from_epoch == 'quarter' and self.current_epoch + 1 == (self._last_epoch) // 4)) and \
                (self._save_results == -1 or batch_idx < self._save_results):

            if self._center_crop is None:
                self._center_crop = K.CenterCrop(96)

            imgs_to_save = []
            imgs_suffixes = []
            imgs_to_save.append(img_sr)
            imgs_suffixes.append('')

            try:
                img_sr_crop = self._center_crop(img_sr)
                imgs_to_save.append(img_sr_crop)
                imgs_suffixes.append('_center')
            except RuntimeError:
                # catch RuntimeError that may happen with center_crop
                self._logger.exception('Runtime Error')
                img_sr_crop = None

            for l in self._losses:
                if l.loss is not None and l.name == 'edge_loss':
                    # save edges version of img_sr
                    imgs_to_save.append(l.loss.extract_edges(img_sr).repeat(1, 3, 1, 1))
                    imgs_suffixes.append('_edges')

                    if img_sr_crop is not None:
                        imgs_to_save.append(l.loss.extract_edges(img_sr_crop).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_center_edges')

                    if self._save_hd_versions[l.name]:
                        # only save HR version once, since it won't change
                        imgs_to_save.append(l.loss.extract_edges(img_hr).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_hr_edges')

                        img_hr_crop = self._center_crop(img_hr)
                        imgs_to_save.append(l.loss.extract_edges(img_hr_crop).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_hr_center_edges')

                        self._save_hd_versions[l.name] = False

                elif l.loss is not None and l.name == 'pencil_sketch':
                    # save pencil sketch version of img_sr
                    imgs_to_save.append(l.loss.pencil_sketch(img_sr).repeat(1, 3, 1, 1))
                    imgs_suffixes.append('_sketch')

                    if img_sr_crop is not None:
                        imgs_to_save.append(l.loss.pencil_sketch(img_sr_crop).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_center_sketch')

                    if self._save_hd_versions[l.name]:
                        # only save HR version once, since it won't change
                        imgs_to_save.append(l.loss.pencil_sketch(img_hr).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_hr_sketch')

                        img_hr_crop = self._center_crop(img_hr)
                        imgs_to_save.append(l.loss.pencil_sketch(img_hr_crop).repeat(1, 3, 1, 1))
                        imgs_suffixes.append('_hr_center_sketch')

                        self._save_hd_versions[l.name] = False

            # save images from each val dataset to be visualized in logger
            # e.g.: DIV2K/0001/epoch_2000
            image_path = f'{self._eval_datasets[dataloader_idx]}/{batch["path"][0]}/epoch_{self.current_epoch+1:05d}'
            self._logger.debug(
                f'Saving {image_path}')

            # save images on disk
            for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
                image_local_path = Path(
                    f'{self._default_root_dir}') / self._eval_datasets[dataloader_idx] / batch["path"][0]
                image_local_path.mkdir(parents=True, exist_ok=True)
                self._logger.debug(
                    f'Saving local file: {image_local_path}/epoch_{self.current_epoch+1:05d}{suffix}.png')
                torchvision.utils.save_image(
                    img_to_save.view(*img_to_save.size()[1:]).cpu().detach(),
                    image_local_path /
                    f'epoch_{self.current_epoch+1:05d}{suffix}.png'
                )

            # save images on loggers
            for logger in self.loggers:
                if isinstance(logger, TensorBoardLogger):
                    for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
                        logger.experiment.add_image(
                            f'{image_path}{suffix}', img_to_save.view(*img_to_save.size()[1:]), self.global_step)

                elif isinstance(logger, CometLogger):
                    for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
                        logger.experiment.log_image(
                            img_to_save.view(*img_to_save.size()[1:]).cpu().detach(),
                            name=f'{image_path}{suffix}',
                            image_channels='first',
                            step=self.global_step,
                        )

            # log images metrics
            image_metrics = {}
            for k, v in result.items():
                new_k = k.split('/')
                new_k = '/'.join([new_k[0], batch["path"][0], new_k[1]])
                image_metrics[new_k] = v

            self.log_dict(image_metrics, prog_bar=False, logger=True, add_dataloader_idx=False)

        self._validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            def _log_metrics(keys, metrics):
                metrics_dict = {}
                for k in keys:
                    if len(metrics[0][k].size()) > 0:
                        # fix LPIPS output that comes as [[[[3.3]]]]
                        # with shape (1,1,1,1) instead of only a number
                        metrics_dict[k] = torch.stack([m[k].squeeze() for m in metrics if k in m]).mean()
                    else:
                        metrics_dict[k] = torch.stack([m[k] for m in metrics if k in m]).mean()
                    metrics_dict[k] = metrics_dict[k].cpu().detach()

                self.log_dict(metrics_dict, prog_bar=False, logger=True, add_dataloader_idx=False)

            if isinstance(self._validation_step_outputs[0], dict):
                # in case of using only one validation dataset
                # outputs is a list of dictionaries
                # where each dict is the result of a batch run
                _log_metrics(self._validation_step_outputs[0].keys(), self._validation_step_outputs)
            else:  # if isinstance(outputs[0], list):
                # in case of using multiple validation datasets
                # outputs is a list of list of dictionaries
                # where each list of lists if referent to a dataset and
                # dict is the result of a batch run
                for dataset_result in self._validation_step_outputs:
                    _log_metrics(dataset_result[0].keys(), dataset_result)

        self._validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # prediction step when using multiple prediction datasets
        # at each prediction step only one image is processed
        img_lr = batch['lr']
        img_sr = self.forward(img_lr)
        img_sr = img_sr.clamp(0, 1)

        if self._center_crop is None:
            self._center_crop = K.CenterCrop(96)

        imgs_to_save = []
        imgs_suffixes = []
        imgs_to_save.append(img_sr)
        imgs_suffixes.append('')

        try:
            img_sr_crop = self._center_crop(img_sr)
            imgs_to_save.append(img_sr_crop)
            imgs_suffixes.append('_center')
        except RuntimeError:
            # catch RuntimeError that may happen with center_crop
            self._logger.exception('Runtime Error')
            img_sr_crop = None

        # save images from each predict dataset to be visualized in logger
        # e.g.: DIV2K/0001
        image_path = f'{self._predict_datasets[dataloader_idx]}/{batch["path"][0]}'
        self._logger.debug(f'Saving {image_path}')

        # save images on disk
        for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
            image_local_path = Path(f'{self._default_root_dir}') / self._predict_datasets[dataloader_idx]
            image_local_path.mkdir(parents=True, exist_ok=True)
            self._logger.debug(f'Saving local file: {image_local_path}/{batch["path"][0]}{suffix}.png')
            torchvision.utils.save_image(
                img_to_save.view(*img_to_save.size()[1:]).cpu().detach(),
                image_local_path / f'{batch["path"][0]}{suffix}.png'
            )

        # save images on loggers
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
                    logger.experiment.add_image(
                        f'{image_path}{suffix}', img_to_save.view(*img_to_save.size()[1:]), self.global_step)

            elif isinstance(logger, CometLogger):
                for img_to_save, suffix in zip(imgs_to_save, imgs_suffixes):
                    if img_to_save.size()[1] > 1:
                        # comet logger currently don't support greyscale images
                        logger.experiment.log_image(
                            img_to_save.view(
                                *img_to_save.size()[1:]).cpu().detach(),
                            name=f'{image_path}{suffix}',
                            image_channels='first',
                            step=self.global_step
                        )

        return img_sr

    def _create_losses(self, losses_str: str, patch_size: int, precision: int=32) -> list[_SubLoss]:
        # support for composite losses, like
        # 0.5 * L1 + 0.5 * adaptive
        self._logger.debug('Preparing loss functions:')
        losses = []
        for loss in losses_str.split('+'):
            loss_split = loss.split('*')
            if len(loss_split) == 2:
                weight, loss_type = loss_split
                try:
                    weight = float(weight)
                except ValueError:
                    raise ValueError(
                        f'{weight} is not a valid number to be used as weight for loss function {loss_type.strip()}')
            else:
                weight = 1.
                loss_type = loss_split[0]

            loss_type = loss_type.strip().lower()

            if loss_type in _supported_losses:
                if loss_type == 'adaptive':
                    loss_function = _supported_losses[loss_type](
                        image_size=(patch_size, patch_size, 3),
                        float_dtype=torch.float32 if precision == 32 else torch.float16,
                        device=torch.device(
                            f'cuda:{self._model_gpus[-1]}') if self._model_parallel else self.device)
                else:
                    if loss_type in {'edge_loss', 'pencil_sketch'}:
                        if self._save_hd_versions is None:
                            self._save_hd_versions = {}

                        self._save_hd_versions[loss_type] = True

                    loss_function = _supported_losses[loss_type]()
                # elif loss_type.find('gan') >= 0:
                #     module = import_module('loss.adversarial')
                #     loss_function = getattr(module, 'Adversarial')(
                #         args,
                #         loss_type
                #     )
                # elif loss_type.find('vgg') >= 0:
                #     module = import_module('loss.vgg')
                #     loss_function = getattr(module, 'VGG')(
                #         loss_type[3:],
                #         rgb_range=args.1.
                #     )
            else:
                raise AttributeError(
                    f'Couldn\'t find loss {loss_type}. Supported losses: {", ".join(_supported_losses)}')

            self._logger.info(f'{weight:.3f} * {loss_type}')

            if self._model_parallel:
                loss_function = loss_function.to(torch.device(
                    f'cuda:{self._model_gpus[-1]}'))

            losses.append(_SubLoss(
                name=loss_type,
                loss=loss_function,
                weight=weight
            ))
            # if loss_type.find('gan') >= 0:
            #     self._losses.append(
            #         {'type': 'dis', 'weight': 1, 'function': None})

        return losses

    def _create_metrics(self, metrics: list[str]) -> list[tuple[str, Callable]]:
        used_metrics = []
        for metric in metrics:
            if metric in _supported_metrics:
                if metric in {'FLIP', 'LPIPS'}:
                    # metrics that are objects and need to be created
                    used_metrics.append((metric, _supported_metrics[metric]()))
                else:
                    # metrics that are functions
                    used_metrics.append((metric, _supported_metrics[metric]))
            else:
                raise AttributeError(
                    f'Couldn\'t find metric {metric}. Supported metrics: {", ".join(_supported_metrics)}')

        return used_metrics

    def _calculate_losses(self, img_sr: torch.Tensor, img_hr: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = []
        losses_names = []
        for l in self._losses:
            # calculate losses individually
            if l.loss is not None:
                if l.name in {'haarpsi', 'pieapp'}:
                    # for these the image must have values between 0 and 1
                    loss = l.loss(torch.clamp(
                        img_sr, 0, 1), img_hr)
                elif l.name == 'adaptive':
                    if self._model_parallel:
                        l.loss.to(torch.device(
                            f'cuda:{self._model_gpus[-1]}'))
                    else:
                        l.loss.to(self.device)
                    loss = torch.mean(l.loss.lossfun(
                        (img_sr - img_hr)).permute(0, 3, 2, 1))
                elif l.name in {'brisque'}:
                    # no-reference loss functions
                    loss = l.loss(torch.clamp(img_sr, 0, 1))
                elif 'lpips' in l.name:
                    if self._model_parallel:
                        l.loss.to(torch.device(
                            f'cuda:{self._model_gpus[-1]}'))
                    else:
                        l.loss.to(self.device)
                    loss = l.loss(img_sr, img_hr)
                    loss = loss.mean()
                else:
                    loss = l.loss(img_sr, img_hr)

                effective_loss = l.weight * loss
                losses_names.append(l.name)
                losses.append(effective_loss)

        losses_dict = {n: l for n, l in zip(losses_names, losses)}
        if len(losses_names) > 1:
            # add other losses to progress bar since total loss
            # is added automatically
            self.log_dict(losses_dict, prog_bar=True, logger=False, add_dataloader_idx=False)

        losses_dict = {f'loss/{k}': v for k, v in losses_dict.items()}
        # training_step must always return None, a Tensor, or a dict with at least
        # one key being 'loss'
        losses_dict['loss'] = sum(losses)
        return losses_dict

    def _calculate_metrics(self, img_sr: torch.Tensor, img_hr: torch.Tensor, dataloader_idx: int = 0) -> torch.Tensor:
        metrics_dict = {}
        for name, metric in self._metrics:
            if name in {'BRISQUE'}:
                # no-reference metrics
                value = metric(img_sr)
            elif name in {'FLIP', 'LPIPS'}:
                # metrics that use a neural network inside
                if self._model_parallel:
                    metric.to(torch.device(
                        f'cuda:{self._model_gpus[-1]}'))
                else:
                    metric.to(self.device)
                value = metric(img_sr, img_hr)
            else:
                value = metric(img_sr, img_hr)

            metrics_dict[f'{self._eval_datasets[dataloader_idx]}/{name}'] = value

        # log so callbacks can use the metrics
        prog_bar_metrics_dict = {k: v for k, v in metrics_dict.items() for m in self._metrics_for_pbar if m in k}
        if len(prog_bar_metrics_dict) == 0:
            prog_bar_metrics_dict = metrics_dict.copy()

        self.log_dict(prog_bar_metrics_dict, prog_bar=True, logger=False, add_dataloader_idx=False)

        return metrics_dict

    def _parse_optimizer_config(self, optimizer: str, optimizer_params: list[str]) -> tuple[optim.Optimizer, dict[str, float | str]]:
        if optimizer in _supported_optimizers:
            optimizer_class = _supported_optimizers[optimizer]
        else:
            raise ValueError(
                f'Optimizer not recognized: {optimizer}. Supported optimizers: {", ".join(_supported_optimizers)}')

        optimizer_params = {}
        for param in optimizer_params:
            param_name, param_value = param.strip().split('=')
            param_name = param_name.strip()
            if param_name in ['eps', 'lr', 'lr_decay', 'weight_decay']:
                # convert to float
                optimizer_params[param_name] = float(param_value)
            elif param_name in ['betas']:
                # convert to tuple of floats
                param_value_list = []
                for v in param_value.split(','):
                    param_value_list.append(float(v))
                optimizer_params[param_name] = tuple(param_value_list)
            else:
                # use param as string
                optimizer_params[param_name] = param_value

        self._logger.debug(f'Optimizer params: {optimizer_params}')

        return optimizer_class, optimizer_params
