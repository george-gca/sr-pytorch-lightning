import argparse
import inspect
import logging
from typing import List
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger

import models
from srdata import SRData


def setup_log(args: argparse.Namespace, logs_to_silence: list[str] = []) -> logging.Logger:
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

    model = Model.load_from_checkpoint(args.checkpoint, predict_datasets=args.predict_datasets)
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

    trainer = Trainer.from_argparse_args(args)
    try:
        trainer.predict(model, dataset)
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
    # read available models from `models` module
    available_models = {k.lower(): v for k, v in inspect.getmembers(models) if inspect.isclass(v) and k != 'SRModel'}

    parser = argparse.ArgumentParser()

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser = SRData.add_dataset_specific_args(parser)

    # add general options to argparse
    parser.add_argument('--checkpoint', type=str, default='')
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
    parser.add_argument('-s', '--scale_factor', type=int, default=4)
    args, remaining = parser.parse_known_args()

    # load model class
    Model = available_models[args.model]

    # add model specific arguments to original parser
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args(remaining, namespace=args)

    main(Model, args)
