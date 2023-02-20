<p>
  <img alt="Python 3" src="https://img.shields.io/badge/-Python-2b5b84?style=flat-square&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" />
  <img alt="PyTorch Lightning" src="https://img.shields.io/badge/-PyTorch%20Lightning-792de4?style=flat-square&logo=pytorch-lightning&logoColor=white" />
  <img alt="Docker" src="https://img.shields.io/badge/-Docker-0073ec?style=flat-square&logo=docker&logoColor=white" />
  <img alt="Comet ML" src="https://custom-icon-badges.herokuapp.com/badge/Comet%20ML-262c3e?style=flat-square&logo=logo_comet_ml&logoColor=white" />
</p>

# sr-pytorch-lightning

## Introduction

Super resolution algorithms implemented with [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). Based on [code by So Uchida](https://github.com/S-aiueo32/sr-pytorch-lightning).

Currently supports the following models:

- [DDBPN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf)
- [EDSR](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)
- [RCAN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)
- [RDN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf)
- [SRCNN](https://ieeexplore.ieee.org/document/7115171?arnumber=7115171) - [arXiv](https://arxiv.org/pdf/1501.00092.pdf)
- [SRGAN](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)
- [SRResNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)
- [WDSR](https://bmvc2019.org/wp-content/uploads/papers/0288-paper.pdf)

## Requirements

- [docker](https://docs.docker.com/engine/install/)
- make
  - install support for Makefile on Ubuntu-based distros using `sudo apt install build-essential`

## Usage

I decided to split the logic of dealing with `docker` (contained in [Makefile](Makefile)) from running the `python` code itself (contained in [start_here.sh](start_here.sh)). Since I run my code in a remote machine, I use `gnu screen` to keep the code running even if my connection fails.

In [Makefile](Makefile) there is a `environment variables` section, where a few variables might be set. More specifically, `DATASETS_PATH` must point to the root folder of your super resolution datasets.

In [start_here.sh](start_here.sh) a few variables might be set in the `variables` region. Default values have been set to allow easy experimentation.

### Creating docker image

```bash
make
```

If you want to build the docker image using the specific versions that I used during my last experiments, simply run

```bash
make DOCKERFILE=Dockerfile_fixed_versions
```

### Testing docker image

```bash
make test
```

This should print information about all available GPUs, like this:

```
Found 2 devices:
        _CudaDeviceProperties(name='NVIDIA Quadro RTX 8000', major=7, minor=5, total_memory=48601MB, multi_processor_count=72)
        _CudaDeviceProperties(name='NVIDIA Quadro RTX 8000', major=7, minor=5, total_memory=48601MB, multi_processor_count=72)
```

### Training model

If you haven't configured the [telegram bot](#finished-experiment-telegram-notification) to notify when running is over, or don't want to use it, simply remove the line

```bash
$(TELEGRAM_BOT_MOUNT_STRING) \
```

from the `make run` command on the [Makefile](Makefile), and also comment the line

```bash
send_telegram_msg=1
```

in [start_here.sh](start_here.sh).

Then, to train the models, simply call

```bash
make run
```

By default, it will run the file [start_here.sh](start_here.sh).

If you want to run another command inside the docker container, just change the default value for the `RUN_STRING` variable.

```bash
make RUN_STRING="ipython3" run
```

## Creating your own model

To create your own model, create a new file inside `models/` and create a class that inherits from [SRModel](models/srmodel.py). Your class should implement the `forward` method. Then, add your model to [\_\_init\_\_.py](models/__init__.py). The model will be automatically available as a `model` parameter option in [train.py](train.py) or [test.py](test.py).

Some good starting points to create your own model are the [SRCNN](models/srcnn.py) and [EDSR](models/edsr.py) models.

## Using Comet

If you want to use [Comet](https://www.comet.ml/) to log your experiments data, just create a file named `.comet.config` in the root folder here, and add the following lines:

```config
[comet]
api_key=YOUR_API_KEY
```

More configuration variables can be found [here](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).

Most of the things that I found useful to log (metrics, codes, log, image results) are already being logged. Check [train.py](train.py) and [srmodel.py](models/srmodel.py) for more details. All these loggings are done by the [comet logger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.comet.html) already available from pytorch lightning. An example of these experiments logged in Comet can be found [here](https://www.comet.ml/george-gca/super-resolution-experiments).

## Finished experiment Telegram notification

Since the experiments can run for a while, I decided to use a telegram bot to notify me when experiments are done (or when there is an error). For this, I use the [telegram-send](https://github.com/rahiel/telegram-send) python package. I recommend you to install it in your machine and configure it properly.

To do this, simply use:

```bash
pip3 install telegram-send
telegram-send --configure
```

Then, simply copy the configuration file created under `~/.config/telegram-send.conf` to another directory to make it easier to mount on the docker image. This can be configured in the source part of the `TELEGRAM_BOT_MOUNT_STRING` variable (by default is set to `$(HOME)/Docker/telegram_bot_config`) in the [Makefile](Makefile).
