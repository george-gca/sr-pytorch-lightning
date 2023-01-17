import argparse
import inspect
import logging
from typing import List
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

import models
from srdata import SRData


class ItemsProgressBar(TQDMProgressBar):
    r"""
    This is the same default progress bar used by Lightning, but printing
    items instead of batches during training. It prints to `stdout` using the
    :mod:`tqdm` package and shows up to four different bars:
    - **sanity check progress:** the progress during the sanity check run
    - **main progress:** shows training + validation progress combined. It also accounts for
      multiple validation runs during training when
      :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
    - **validation progress:** only visible during validation;
      shows total progress over all validation datasets.
    - **test progress:** only active when testing; shows total progress over all test datasets.
    For infinite datasets, the progress bar never ends.
    If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
    specific methods of the callback class and pass your custom implementation to the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`:
    Example::
        class LitProgressBar(ProgressBar):
            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.set_description('running validation ...')
                return bar
        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])
    Args:
        refresh_rate:
            Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display. By default, the
            :class:`~pytorch_lightning.trainer.trainer.Trainer` uses this implementation of the progress
            bar and sets the refresh rate to the value provided to the
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
        process_position:
            Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together. This corresponds to
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.process_position` in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0, batch_size: int = 16):
        super().__init__(refresh_rate, process_position)
        self.batch_size = batch_size

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate * self.batch_size

    @property
    def train_batch_idx(self) -> int:
        """
        The current batch index being processed during training.
        Use this to update your progress bar.
        """
        return self.trainer.fit_loop.epoch_loop.batch_progress.current.processed * self.batch_size
        # return self._train_batch_idx * self.batch_size

    @property
    def total_train_batches(self) -> int:
        """
        The total number of training batches during training, which may change from epoch to epoch.
        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the
        training dataloader is of infinite size.
        """
        return self.trainer.num_training_batches * self.batch_size


def setup_log(args: argparse.Namespace, logs_to_silence: List[str] = []) -> logging.Logger:
    def log_print(self, message, *args, **kws):
        if self.isEnabledFor(logging.PRINT):
            # yes, logger takes its '*args' as 'args'.
            self._log(logging.PRINT, message, args, **kws)

    # add print level to logging, just to be able to print to both console and log file
    logging.PRINT = 60
    logging.addLevelName(60, 'PRINT')
    logging.Logger.print = log_print

    log_level = {
        'debug': logging.DEBUG,  # 10
        'info': logging.INFO,  # 20
        'warning': logging.WARNING,  # 30
        'error': logging.ERROR,  # 40
        'critical': logging.CRITICAL,  # 50
        'print': logging.PRINT,  # 60
    }[args.log_level]

    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()

    # create a logging format
    if log_level >= logging.INFO:
        stderr_formatter = logging.Formatter('{message}', style='{')
    else:
        stderr_formatter = logging.Formatter(
            # format:
            # <10 = pad with spaces if needed until it reaches 10 chars length
            # .10 = limit the length to 10 chars
            '{name:<10.10} [{levelname:.1}] {message}', style='{')

    stderr_handler.setFormatter(stderr_formatter)

    # create a handler to log to file
    log_file = Path(args.default_root_dir)
    log_file.mkdir(parents=True, exist_ok=True)
    log_file = log_file / 'run.log'
    file_handler = logging.FileHandler(log_file, mode='w')

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    file_formatter = logging.Formatter(
        '{asctime} - {name:<12.12} {levelname:<8} {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
    file_handler.setFormatter(file_formatter)

    # add the handlers to the root logger
    logging.basicConfig(level=log_level, handlers=[
                        file_handler, stderr_handler])

    # change logger level of logs_to_silence to warning
    for other_logger in logs_to_silence:
        logging.getLogger(other_logger).setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.print(f'Saving logs to {log_file.absolute()}')
    logger.print(f'Log level: {logging.getLevelName(log_level)}')
    return logger


def main(Model: models.SRModel, args: argparse.Namespace):
    logger = setup_log(args, ['PIL'])

    model = Model(**vars(args))
    dataset = SRData(args)
    args.logger = []

    if args.deterministic:
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED
        seed_everything(0)

    if 'comet' in args.loggers:
        '''
        for this to work, create the file ~/.comet.config with
        [comet]
        api_key = YOUR API KEY

        for more info, see https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
        '''
        comet_logger = CometLogger(
            save_dir=args.default_root_dir,
            project_name=args.comet_project,
            experiment_name=Path(args.default_root_dir).name,
            offline=False
        )

        # all code will be under /work when running on docker
        comet_logger.experiment.log_code(folder='/work')
        comet_logger.experiment.set_model_graph(str(model))

        comet_logger.experiment.log_other(
            'trainable params',
            sum(p.numel() for p in model.parameters() if p.requires_grad))

        total_params = sum(p.numel() for p in model.parameters())
        comet_logger.experiment.log_other('total params', total_params)

        total_loss_params = 0
        total_loss_trainable_params = 0
        for loss in model._losses:
            if loss.name.find('adaptive') >= 0:
                total_loss_params += sum(p.numel() for p in loss.loss.parameters())
                total_loss_trainable_params += sum(p.numel()
                                for p in loss.loss.parameters() if p.requires_grad)

        if total_loss_params > 0:
            comet_logger.experiment.log_other(
                'loss total params', total_loss_params)
            comet_logger.experiment.log_other(
                'loss trainable params', total_loss_trainable_params)

        # assume 4 bytes/number (float on cuda)
        denom = 1024 ** 2.
        input_size = abs(np.prod(model.example_input_array.size()) * 4. / denom)
        params_size = abs(total_params * 4. / denom)
        comet_logger.experiment.log_other('input size (MB)', input_size)
        comet_logger.experiment.log_other('params size (MB)', params_size)

        args.logger.append(comet_logger)

    if 'tensorboard' in args.loggers:
        tensorboard_logger = TensorBoardLogger(
            save_dir=args.default_root_dir,
            name='tensorboard_logs',
            log_graph=args.log_graph,
            default_hp_metric=False
        )

        args.logger.append(tensorboard_logger)

    # enable saving checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.default_root_dir}/checkpoints',
        filename=f'{args.model}_{{epoch}}_{{{args.save_metric}:.3f}}',
        mode='max',
        monitor=args.save_metric,
        verbose=0 < logger.level < logging.WARNING,
        every_n_epochs=args.check_val_every_n_epoch,
        save_last=True,
        save_top_k=3,
    )

    # enable items in progress bar
    progressbar = ItemsProgressBar(batch_size=args.batch_size)
    args.callbacks = [checkpoint_callback, progressbar]

    # os.makedirs(args.default_root_dir, exist_ok=True)

    trainer = Trainer.from_argparse_args(args)

    # start training
    try:
        trainer.fit(model, dataset)

        # upload last model checkpoint to comet.ml
        if 'comet' in args.loggers:
            last_checkpoint = Path(args.default_root_dir) / 'checkpoints' / 'last.ckpt'
            model_name = str(Path(args.default_root_dir).name)
            comet_logger.experiment.log_model(
                f'{model_name}', f'{last_checkpoint}', overwrite=True)
    except RuntimeError:
        # catch the RuntimeError: CUDA error: out of memory and finishes execution
        torch.cuda.empty_cache()
        logger.exception('Runtime Error')
    except:
        # catch other errors and finish execution so the log is uploaded to comet ml
        torch.cuda.empty_cache()
        logger.exception('Fatal error')

    if 'comet' in args.loggers:
        # upload log of execution to comet.ml
        comet_logger.experiment.log_asset(f'{Path(args.default_root_dir) / "run.log"}')


if __name__ == '__main__':
    available_eval_datasets = (
        'DIV2K',
        'Set5',
        'Set14',
        'B100',
        'Urban100',
        # 'RealSR'
    )

    available_metrics = (
        'BRISQUE',
        'FLIP',
        'LPIPS',
        'MS-SSIM',
        'PSNR',
        'SSIM',
    )

    # read available models from `models` module
    available_models = {k.lower(): v for k, v in inspect.getmembers(models) if inspect.isclass(v) and k != 'SRModel'}

    parser = argparse.ArgumentParser()

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser = SRData.add_dataset_specific_args(parser)

    # add general options to argparse
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--comet_project', type=str, default='sr-pytorch-lightning')
    parser.add_argument('--log_graph', action='store_true',
                        help='log model graph to tensorboard')
    parser.add_argument('--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'))
    parser.add_argument('--loggers', type=str, nargs='+',
                        choices=('comet', 'tensorboard'),
                        default=('comet', 'tensorboard'))
    parser.add_argument('-m', '--model', type=str,
                        choices=tuple(available_models),
                        default='srcnn')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('-s', '--scale_factor', type=int, default=4)
    args, remaining = parser.parse_known_args()

    # load model class
    Model = available_models[args.model]

    # add model specific arguments to original parser
    parser = Model.add_model_specific_args(parser)
    args, remaining = parser.parse_known_args(remaining, namespace=args)

    # add save_metric arg choices based on selected eval datasets and metrics
    available_save_metrics = []
    for d in args.eval_datasets:
        for m in args.metrics:
            available_save_metrics.append(f'{d}/{m}')
    parser.add_argument('--save_metric', type=str, default=available_save_metrics[0],
                        choices=available_save_metrics,
                        help='metric to be used for selecting top result')
    args = parser.parse_args(remaining, namespace=args)

    main(Model, args)
