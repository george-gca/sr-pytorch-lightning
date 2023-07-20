import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import numpy as np
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CometLogger

import models
from srdata import SRData


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning', 'error', 'critical'))
        parser.add_argument('--file_log_level', type=str, default='info',
                        choices=('debug', 'info', 'warning', 'error', 'critical'))

        # https://lightning.ai/docs/pytorch/LTS/cli/lightning_cli_expert.html#argument-linking
        parser.link_arguments('data.batch_size', 'model.init_args.batch_size')
        parser.link_arguments('data.eval_datasets', 'model.init_args.eval_datasets')
        parser.link_arguments('data.patch_size', 'model.init_args.patch_size')
        parser.link_arguments('data.scale_factor', 'model.init_args.scale_factor')

        parser.link_arguments('trainer.check_val_every_n_epoch', 'model.init_args.log_weights_every_n_epochs')
        parser.link_arguments('trainer.check_val_every_n_epoch', 'trainer.callbacks.init_args.every_n_epochs')
        parser.link_arguments('trainer.default_root_dir', 'model.init_args.default_root_dir')
        parser.link_arguments('trainer.default_root_dir', 'trainer.logger.init_args.save_dir') # not working for comet logger
        parser.link_arguments('trainer.default_root_dir', 'trainer.callbacks.init_args.dirpath',
                              compute_fn=lambda x: f'{x}/checkpoints')
        parser.link_arguments('trainer.max_epochs', 'model.init_args.max_epochs')

    def before_fit(self):
        # setup logging
        default_root_dir = Path(self.config['fit']['trainer']['default_root_dir'])
        default_root_dir.mkdir(parents=True, exist_ok=True)

        setup_log(
            level=self.config['fit']['log_level'],
            log_file=default_root_dir / 'run.log',
            file_level=self.config['fit']['file_log_level'],
            logs_to_silence=['PIL'],
        )

        for logger in self.trainer.loggers:
            if isinstance(logger, CometLogger):
                # all code will be under /work when running on docker
                logger.experiment.log_code(folder='/work')
                logger.experiment.log_parameters(self.config.as_dict())
                logger.experiment.set_model_graph(str(self.model))
                logger.experiment.log_other(
                    'trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

                total_params = sum(p.numel() for p in self.model.parameters())
                logger.experiment.log_other('total params', total_params)

                total_loss_params = 0
                total_loss_trainable_params = 0
                for loss in self.model._losses:
                    if loss.name.find('adaptive') >= 0:
                        total_loss_params += sum(p.numel() for p in loss.loss.parameters())
                        total_loss_trainable_params += sum(p.numel()for p in loss.loss.parameters() if p.requires_grad)

                if total_loss_params > 0:
                    logger.experiment.log_other('loss total params', total_loss_params)
                    logger.experiment.log_other('loss trainable params', total_loss_trainable_params)

                # assume 4 bytes/number (float on cuda)
                denom = 1024 ** 2.
                input_size = abs(np.prod(self.model.example_input_array.size()) * 4. / denom)
                params_size = abs(total_params * 4. / denom)
                logger.experiment.log_other('input size (MB)', input_size)
                logger.experiment.log_other('params size (MB)', params_size)
                break

    def after_fit(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, CometLogger):
                default_root_dir = Path(self.config['fit']['trainer']['default_root_dir'])
                last_checkpoint = default_root_dir / 'checkpoints' / 'last.ckpt'
                model_name = self.config['fit']['model']['class_path'].split('.')[-1]
                logger.experiment.log_model(f'{model_name}', f'{last_checkpoint}', overwrite=True)
                logger.experiment.log_asset(f'{default_root_dir / "run.log"}')
                break


def cli_main() -> None:
    _ = CustomLightningCLI(
        model_class=models.SRModel,
        subclass_mode_model=True,
        datamodule_class=SRData,
        parser_kwargs={"parser_mode": "omegaconf"},
        )


def setup_log(
        level: str = 'warning',
        log_file: str | Path = Path('run.log'),
        file_level: str = 'info',
        logs_to_silence: list[str] = [],
        ) -> None:
    """
    Setup the logging.

    Args:
        log_level (str): stdout log level. Defaults to 'warning'.
        log_file (str | Path): file where the log output should be stored. Defaults to 'run.log'.
        file_log_level (str): file log level. Defaults to 'info'.
        logs_to_silence (list[str]): list of loggers to be silenced. Useful when using log level < 'warning'. Defaults to [].
    """
    # TODO: fix this according to this
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    # https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/

    # convert log levels to int
    int_log_level = {
        'debug': logging.DEBUG,  # 10
        'info': logging.INFO,  # 20
        'warning': logging.WARNING,  # 30
        'error': logging.ERROR,  # 40
        'critical': logging.CRITICAL,  # 50
    }

    stdout_log_level = int_log_level[level]
    file_log_level = int_log_level[file_level]

    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(stdout_log_level)

    # create a logging format
    if stdout_log_level >= logging.WARNING:
        stderr_formatter = logging.Formatter('{message}', style='{')
    else:
        stderr_formatter = logging.Formatter(
            # format:
            # <10 = pad with spaces if needed until it reaches 10 chars length
            # .10 = limit the length to 10 chars
            '{name:<10.10} [{levelname:.1}] {message}', style='{')
    stderr_handler.setFormatter(stderr_formatter)

    # create a file handler that have size limit
    if isinstance(log_file, str):
        log_file = Path(log_file).expanduser()

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)  # ~ 5 MB
    file_handler.setLevel(file_log_level)

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    file_formatter = logging.Formatter(
        '{asctime} - {name:<20.20} {levelname:<8} {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
    file_handler.setFormatter(file_formatter)

    # add the handlers to the root logger
    logging.basicConfig(handlers=[file_handler, stderr_handler], level=logging.DEBUG)

    # change logger level of logs_to_silence to warning
    for other_logger in logs_to_silence:
        logging.getLogger(other_logger).setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(__name__)

    logger.info(f'Saving logs to {log_file.absolute()}')
    logger.info(f'Log level: {logging.getLevelName(stdout_log_level)}')


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
