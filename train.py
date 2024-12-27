# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""
# Ultralytics YOLOv5 🚀，AGPL-3.0 许可协议
"""
在自定义数据集上训练 YOLOv5 模型。模型和数据集会自动从最新的 YOLOv5 发布版本下载。

用法 - 单 GPU 训练：
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # 从预训练模型开始（推荐）
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # 从头开始训练

用法 - 多 GPU DDP 训练：
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

模型：     https://github.com/ultralytics/yolov5/tree/master/models
数据集：   https://github.com/ultralytics/yolov5/tree/master/data
教程：     https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

# 标准库模块
import argparse # 参数解析
import math # 数学运算
import os # 文件系统操作
import random # 随机数生成
import subprocess # 子进程管理
import sys # Python 运行环境
import time # 时间操作
from copy import deepcopy # 深拷贝
from datetime import datetime, timedelta # 日期与时间操作
from pathlib import Path # 路径处理
# 一个安全引入模块的机制
# comet_ml 是一个用于实验追踪和日志记录的工具，可以帮助开发者记录模型训练的超参数、指标和结果。
# 其引入需要在 torch 之前完成，因为 comet_ml 可能会修改一些底层行为，如加速日志记录的功能。
try:
    import comet_ml  # must be imported before torch (if installed) 如果模块存在并成功引入，则后续代码可以正常使用它。
except ImportError: # 如果模块不存在（ImportError），将 comet_ml 变量设置为 None，以避免脚本报错
    comet_ml = None

# 第三方模块
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml # yaml 文件处理
from torch.optim import lr_scheduler
from tqdm import tqdm # 进度条显示

# 1.确保当前脚本能够正确找到 YOLOv5 项目的根目录。
# 2.通过将根目录加入 sys.path，保证代码能够导入项目中的模块，即使用户从任意路径运行该脚本。
# 3.转换为相对路径，使调试输出和日志记录更具可读性。
FILE = Path(__file__).resolve() # __file__ 是当前脚本的相对路径，resolve() 将其转换为绝对路径。
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取当前脚本所在的目录路径，即将当前文件的路径向上一级转换。
if str(ROOT) not in sys.path: # 检查 ROOT 是否已经在 Python 的 sys.path 中。 sys.path 是 Python 用来搜索模块的路径列表。
    sys.path.append(str(ROOT))  # add ROOT to PATH 如果 ROOT 不在 sys.path 中，将其加入。这样可以确保脚本能够从 ROOT 中导入模块（如 models、utils 等）。
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 将绝对路径 ROOT 转换为相对于当前工作目录的路径。

# 导入验证模块 val，并给它一个别名 validate。
# 通常，验证模块用于在每个训练周期结束时计算 mAP（mean Average Precision） 等评估指标，验证模型的表现。
import val as validate  # for end-of-epoch mAP
# attempt_load：用于加载模型文件（如 .pt 格式的权重文件）。
# experimental 中的模块通常包含一些实验性的功能，可能是最新的或尚未正式稳定的特性。
from models.experimental import attempt_load
# 导入 YOLOv5 的核心模型结构，用于定义和训练 YOLO 网络模型。
from models.yolo import Model
# 用于检查和优化锚框。YOLO 模型的锚框是预定义的边界框尺寸，用于预测目标的位置。
from utils.autoanchor import check_anchors
# 用于检查并调整训练时的批量大小。批量大小会影响训练的速度和内存消耗，因此需要根据硬件条件来优化。
from utils.autobatch import check_train_batch_size
# 这个模块通常用于实现训练过程中的回调函数。例如，训练中可以执行特定操作（如学习率调整、模型保存等）当满足某些条件时。
from utils.callbacks import Callbacks
# 用于创建数据加载器，读取训练、验证数据，并支持批量加载。
from utils.dataloaders import create_dataloader
# attempt_download：用于尝试下载文件，通常用于下载预训练的模型或数据集。
# is_url：用于检查给定的字符串是否为 URL，通常用于验证下载源的有效性。
from utils.downloads import attempt_download, is_url
# 导入常见工具函数
from utils.general import (
    LOGGER,                     # LOGGER 用于记录日志
    TQDM_BAR_FORMAT,            # TQDM_BAR_FORMAT 用于调整进度条的格式
    check_amp,                  # check_amp 用于检查自动混合精度（AMP）是否支持
    check_dataset,              # check_dataset 用于验证数据集的正确性
    check_file,
    check_git_info,             # check_git_info 和 check_git_status 用于获取 Git 仓库信息
    check_git_status,
    check_img_size,             # check_img_size 用于确保图像尺寸符合要求
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,             # increment_path 用于创建新路径以避免文件覆盖
    init_seeds,                 # init_seeds 用于设置随机种子确保可复现性
    intersect_dicts,            # intersect_dicts 用于字典的交集操作
    labels_to_class_weights,    # labels_to_class_weights 和 labels_to_image_weights 用于处理标签权重
    labels_to_image_weights,
    methods,                    # methods 和 one_cycle 通常用于训练过程中的算法策略
    one_cycle,
    print_args,                 # print_args 用于打印训练配置
    print_mutation,
    strip_optimizer,            # strip_optimizer 用于优化器修剪
    yaml_save,                  # yaml_save 用于保存配置
)
# LOGGERS 和 Loggers：用于管理和记录训练日志，可以支持不同的日志记录器（如本地文件日志、Comet.ml 等）。
from utils.loggers import LOGGERS, Loggers
# check_comet_resume：用于检查是否可以从 Comet.ml 中恢复训练，通常用于支持训练的中断恢复。
from utils.loggers.comet.comet_utils import check_comet_resume
# ComputeLoss：计算 YOLO 模型的损失函数。该损失函数结合了位置损失、置信度损失、类别损失等，用于指导模型优化。
from utils.loss import ComputeLoss
# fitness：用于计算模型的 fitness，通常是与模型性能相关的度量，可能包括准确率、mAP 等。
from utils.metrics import fitness
# plot_evolve：用于绘制训练过程中指标的演变（如损失、mAP 等），以帮助分析模型训练的进展和性能。
from utils.plots import plot_evolve
# 导入 PyTorch 相关模块
from utils.torch_utils import (
    EarlyStopping,      # 提前停止机制，如果在若干个训练周期内没有显著提升性能，则提前终止训练，防止过拟合。
    ModelEMA,           # 用于模型指数移动平均（EMA）的方法，帮助平滑模型训练过程中的波动，提升最终模型的稳定性。
    de_parallel,        # 用于去除多卡训练时的模型并行化部分，恢复为单卡模型。
    select_device,      # 选择计算设备（如 CPU、GPU），根据硬件环境动态选择。
    smart_DDP,          # 用于智能分布式数据并行训练，自动适配多卡训练环境。
    smart_optimizer,    # 智能选择优化器，依据任务自动选择最合适的优化算法。
    smart_resume,       # 用于恢复中断的训练，自动加载断点文件，恢复模型训练。
    torch_distributed_zero_first, # 用于分布式训练，确保 rank 0 在进行分布式训练前能够顺利加载模型和数据。
)

# 功能：LOCAL_RANK 是在分布式训练（尤其是使用多卡训练）中，每个进程的本地排名（即在单个节点上的GPU编号）。它从环境变量中获取，如果环境变量中没有设置，默认为 -1。
# 用途：在多 GPU 环境下，LOCAL_RANK 用来标识当前进程使用的是哪一块 GPU。在 PyTorch 分布式训练中，LOCAL_RANK 经常用来选择特定的设备（如 GPU）。
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 功能：RANK 是在整个分布式训练过程中，当前进程的全局排名（即在所有进程中的编号）。它从环境变量中获取，如果没有设置，默认为 -1。
# 用途：RANK 标识了当前进程在所有训练进程中的位置，通常用于标识每个进程的角色（如 rank 0 通常用于主节点的任务，如数据加载、保存模型等）。
#      在多节点训练中，RANK 用来区分不同节点之间的任务。
RANK = int(os.getenv("RANK", -1))
# 功能：WORLD_SIZE 是分布式训练中的总进程数，表示参与训练的所有进程的数量。它从环境变量中获取，如果没有设置，默认为 1。
# 用途：在分布式训练中，WORLD_SIZE 用来知道总共有多少个进程参与。它在分布式通信、数据分配、梯度同步等过程中起到了重要作用。
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
# 功能：check_git_info() 是一个自定义的函数，用于获取当前代码库的 Git 信息，如当前的提交哈希、分支名等。
# 用途：通过 GIT_INFO 可以记录当前模型训练时的代码版本信息，确保模型的可复现性和版本追溯。如果出现问题，可以根据 Git 信息回溯到出错的代码版本。
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Train a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        None

    Models and datasets download automatically from the latest YOLOv5 release.

    Example:
        Single-GPU training:
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    """
    使用指定的超参数、选项和设备，在自定义数据集上训练一个 YOLOv5 模型，管理数据集、模型架构、损失计算和优化器步骤。

    参数：
        hyp (str | dict): 超参数 YAML 文件的路径，或超参数的字典。
        opt (argparse.Namespace): 解析后的命令行参数，包含训练选项。
        device (torch.device): 训练所在的设备，例如 'cuda' 或 'cpu'。
        callbacks (Callbacks): 用于各种训练事件的回调函数。

    返回：
        None

    模型和数据集会自动从最新的 YOLOv5 发布版本下载。

    示例：
        单GPU训练：
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # 从预训练开始（推荐）
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # 从头开始训练
        ```

        多GPU DDP训练：
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        更多使用详情，请参考：
        - 模型: https://github.com/ultralytics/yolov5/tree/master/models
        - 数据集: https://github.com/ultralytics/yolov5/tree/master/data
        - 教程: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """

    # 从命令行参数 opt 中提取各个训练相关的配置选项，并将它们赋值给变量
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir), # 保存模型的目录路径。
        opt.epochs,         # 训练的轮数。
        opt.batch_size,     # 每个批次的大小。
        opt.weights,        # 权重文件路径，通常是预训练模型的路径。
        opt.single_cls,     # 是否将所有类别合并为单一类别训练。
        opt.evolve,         # 是否启用超参数进化（例如，自动调整学习率等）。
        opt.data,           # 数据集配置文件路径。
        opt.cfg,            # 模型配置文件路径。
        opt.resume,         # 是否从上次训练中断的地方恢复训练。
        opt.noval,          # 是否跳过验证。
        opt.nosave,         # 是否跳过保存模型。
        opt.workers,        # 用于数据加载的工作线程数。
        opt.freeze,         # 是否冻结网络的部分层进行训练。
    )
    # 调用 callbacks.run("on_pretrain_routine_start") 来触发预训练开始前的回调函数
    callbacks.run("on_pretrain_routine_start")

    # Directories
    # 根据指定的 save_dir 目录路径创建存储模型权重的子目录，
    # 并定义 last.pt 和 best.pt 文件路径，分别用于存储训练过程中最后的模型权重和最好的模型权重。
    w = save_dir / "weights"  # weights dir
    # 创建权重文件夹（如果不存在的话）
    # w.parent: 这是 w 的父目录，也就是 save_dir。如果 evolve 为 True，则使用父目录来存储文件。
    # evolve: 这个变量决定是否启用超参数进化（例如，动态调整模型的训练参数）。如果启用 evolve，则会使用 w.parent（父目录）来存储文件。
    # mkdir(parents=True, exist_ok=True): 创建目录时，parents=True 表示如果父目录不存在也会创建，exist_ok=True 表示如果目录已存在，不会报错。
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # last: 用于存储训练过程中最后一个保存的模型权重文件的路径，文件名为 last.pt。
    # best: 用于存储训练过程中性能最好的模型权重文件的路径，文件名为 best.pt。
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    # 加载超参数，并输出它们的值，
    # 同时将超参数存储在 opt 对象中，以便在训练过程中的检查点（checkpoints）中进行保存
    if isinstance(hyp, str):
        # errors="ignore" 参数确保如果文件编码出现问题时，不会抛出错误，而是忽略错误并继续执行。
        with open(hyp, errors="ignore") as f:
            # yaml.safe_load(f) 会将 YAML 文件内容转换为 Python 字典格式
            hyp = yaml.safe_load(f)  # load hyps dict
    # LOGGER.info(...) 用于输出超参数的日志。
    # colorstr("hyperparameters: "):
    #   这部分是一个带颜色的日志输出，其中 "hyperparameters: " 会以特定的颜色显示。
    #   colorstr 是一个函数，用于为日志文本添加颜色，通常用于增强输出的可读性。
    # ", ".join(f"{k}={v}" for k, v in hyp.items())：
    #   将超参数字典中的每个键值对格式化成 "key=value" 的形式，并用逗号分隔，形成一个字符串。
    #   例如：learning_rate=0.001, batch_size=32。
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    # 将超参数字典 hyp 的副本赋值给 opt.hyp。
    # 这样做的目的是确保训练过程中的超参数可以保存在模型的检查点中，以便在恢复训练时可以查看或使用这些超参数。
    # hyp.copy() 是为了避免后续对 hyp 字典的修改影响到 opt.hyp，确保 opt.hyp 保存的是当时加载的超参数副本。
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    # 只有当 evolve 为 False 时，才会执行保存操作。
    # 否则，如果启用了超参数进化，可能会跳过保存设置的步骤，因为进化过程可能会在多次训练中逐步调整超参数。
    if not evolve:
        # yaml_save(save_dir / "hyp.yaml", hyp) 将超参数字典 hyp 保存为 hyp.yaml 文件。
        # yaml_save 是一个自定义函数，通常会调用 yaml.dump() 来将字典数据格式化为 YAML 格式并写入文件。
        yaml_save(save_dir / "hyp.yaml", hyp)
        # vars(opt) 是一个内置函数，它返回一个字典，包含了 opt 对象的所有实例变量（即训练的命令行参数和选项）。
        # 这些选项包括批大小、学习率、数据集路径等参数。
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    # RANK 变量通常与分布式训练（如多GPU训练）相关。
    # RANK 为 -1 或 0 表示这是主进程（通常是进行日志记录和控制的节点）。
    # 如果在分布式训练中只有主进程会记录日志。
    if RANK in {-1, 0}:
        # LOGGERS 是一个定义了可用日志记录器的集合
        include_loggers = list(LOGGERS)
        # 如果 opt.ndjson_console 为 True，则将 ndjson_console 添加到日志记录器列表中。
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        # 如果 opt.ndjson_file 为 True，则将 ndjson_file 添加到日志记录器列表中。
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")
        # 创建一个 Loggers 对象，传入训练的相关参数：
        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        # methods(loggers) 获取 loggers 对象中可用的方法列表（例如，保存模型、记录指标等）。
        # callbacks.register_action(k, callback=getattr(loggers, k))
        # 将这些方法作为回调注册到 callbacks 中，确保在训练过程中适当时机调用日志记录方法。
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        # loggers.remote_dataset 可能是一个从远程存储或云平台下载的数据集的链接或路径信息。
        # 如果配置了远程数据集链接，data_dict 将包含相关信息。
        data_dict = loggers.remote_dataset
        # 如果 resume 为 True，则意味着正在恢复之前的训练
        if resume:  # If resuming runs from remote artifact
            # 在恢复时，重新加载权重、训练轮数、超参数和批量大小等信息，确保恢复的训练状态和之前的训练一致
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    # plots 变量决定是否创建可视化图表。
    # evolve 为 True 时，通常表示正在进行超参数优化过程，这时不需要生成训练图表。
    # opt.noplots 如果为 True，也表示禁用绘图。
    plots = not evolve and not opt.noplots  # create plots
    # 判断是否使用 GPU（CUDA）。如果 device.type 不等于 "cpu"，说明正在使用 GPU。
    cuda = device.type != "cpu"
    # init_seeds 用于初始化随机种子，确保实验的可复现性。
    # opt.seed 是用户指定的随机种子，RANK 是当前训练进程的编号，+1 和 + RANK 确保每个进程的种子不同。
    # deterministic=True 表示启用确定性计算，确保训练过程中使用固定的随机数生成序列。
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 该代码块确保在分布式训练环境中，只有主进程（RANK == 0）会执行数据集检查。
    # check_dataset(data) 会检查并加载数据集。如果 data_dict 已经存在，则跳过。
    # torch_distributed_zero_first(LOCAL_RANK) 确保只在主进程（RANK == 0）进行操作，避免在其他进程中重复工作。
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    # data_dict 是通过 check_dataset(data) 加载的，包含了训练集和验证集的路径信息。
    train_path, val_path = data_dict["train"], data_dict["val"]
    # nc 表示类别数（number of classes）。
    # 如果 single_cls 为 True，则表示只有一个类别，nc 设置为 1。
    # 否则，nc 从数据字典中获取，表示数据集中的类别数量。
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    # 如果是单类任务（single_cls=True），并且数据字典中的 names 列表长度不是 1，则将类别名称设置为 {0: "item"}，表示唯一类别为 "item"。
    # 否则，使用数据字典中的 names（即多个类别的名称）。
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # is_coco 是一个布尔值，用于检查验证集路径是否为 COCO 数据集的路径。
    # 如果 val_path 是字符串并且以 coco/val2017.txt 结尾，说明是 COCO 数据集。
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    # Model
    # check_suffix 是一个函数，用来检查 weights 路径是否以 .pt 结尾，
    # 确保提供的是一个 PyTorch 模型权重文件（通常为 .pt 格式）
    check_suffix(weights, ".pt")  # check weights
    # 如果 weights 是以 .pt 结尾，表示是预训练模型
    pretrained = weights.endswith(".pt")
    if pretrained:
        # torch_distributed_zero_first(LOCAL_RANK) 确保在分布式训练中只有主进程下载模型权重文件。
        with torch_distributed_zero_first(LOCAL_RANK):
            # attempt_download(weights) 尝试从远程下载 weights 文件（如果在本地找不到的话）
            weights = attempt_download(weights)  # download if not found locally
        # 加载预训练模型的 checkpoint（权重文件），并将其加载到 CPU 上，避免直接加载到 GPU 时可能的 CUDA 内存泄漏。
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        # 根据配置文件或加载的 checkpoint 创建模型实例。
        # cfg or ckpt["model"].yaml：如果 cfg 存在，则使用它作为配置文件；否则，使用 checkpoint 中的配置。
        # ch=3 表示输入通道数，这里假设为 RGB 图像（3 个通道）。
        # nc=nc 是类别数，前面已经设置。
        # anchors=hyp.get("anchors") 是使用超参数中定义的 anchor。
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 如果模型的配置或超参数中包含 anchors，并且不需要恢复（即不使用 resume），则排除 anchor 部分的权重。
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        # 获取预训练模型的权重，转换为 FP32 精度。
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # 将预训练模型的权重与当前模型的权重进行交集操作，只加载匹配的权重。
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 加载权重到模型中，strict=False 表示不严格要求所有层都匹配，允许某些层不匹配（例如，排除的 anchors 层）。
        model.load_state_dict(csd, strict=False)  # load
        # 记录已从预训练模型中成功加载了多少个参数。
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        # 如果没有预训练权重，则创建一个新模型
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    # 检查自动混合精度（AMP）
    # AMP 有助于提高训练效率并节省内存，特别是在使用 GPU 时
    amp = check_amp(model)  # check AMP

    # Freeze
    # freeze 变量是一个列表，表示需要冻结的层的名称。
    # 如果 freeze 列表的长度大于 1，则使用给定的 freeze 值作为层名列表。如果 freeze 只有一个元素，则认为它是一个整数，表示要冻结的层的范围。
    # f"model.{x}." 是用来格式化每个层名称的字符串。例如，如果 freeze = [0, 1]，则会生成 ["model.0.", "model.1."]，用于表示 model 模型中的前两层。
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # model.named_parameters() 会返回模型中所有的参数及其对应的名称
    for k, v in model.named_parameters():
        # 默认情况下，v.requires_grad = True，这表示所有参数的梯度是计算的，意味着所有层默认都可以训练。
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 如果当前层的名称 k 包含在 freeze 列表中，那么这层就会被冻结。
        # any(x in k for x in freeze) 是一个条件判断，表示如果 k（层名称）中包含 freeze 中的任意一个元素，则将该层的 requires_grad 设置为 False，即冻结该层的参数。
        # LOGGER.info(f"freezing {k}") 会输出一条日志，记录被冻结的层的名称。
        # v.requires_grad = False 将该层的 requires_grad 设置为 False，表示这个层的参数不参与训练。
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    # model.stride.max()：stride 表示模型中各层的步幅（stride），它是网络中每个卷积层或池化层的步长。
    # model.stride 是一个包含每层步幅的张量。stride.max() 是获取模型中最大步幅值。
    # max(int(model.stride.max()), 32)：选择 stride.max() 和 32 之间的较大值作为 gs。
    # 步幅的大小决定了图像的网格划分，因此需要确保图像尺寸是步幅的倍数，避免在网络中出现不规则的尺寸。
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # check_img_size(opt.imgsz, gs, floor=gs * 2)：该函数用于检查并确保传入的图像尺寸（opt.imgsz）是 gs 的倍数。
    # gs 是最大步幅，图像尺寸需要是 gs 的整数倍，以确保在网络中处理时没有尺寸不匹配的问题。
    # floor=gs * 2：这表示如果输入图像尺寸不符合要求，可以将其调整为 gs 的倍数，且不小于 gs * 2。
    # floor 参数确保在调整图像尺寸时不会太小。
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    # RANK == -1：这意味着当前模型不是在分布式训练模式下运行，而是单GPU模式。
    # RANK 用于标识不同的训练进程，在单GPU模式下，RANK 的值为 -1。
    # batch_size == -1：表示当前批量大小未指定或为默认值（通常是 -1），此时需要估算最佳的批量大小。
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # check_train_batch_size(model, imgsz, amp)：这是一个函数，
        # 用来根据当前模型（model）、输入图像尺寸（imgsz）和是否使用混合精度训练（amp）来估算最合适的批量大小。
        # check_train_batch_size 会根据可用的 GPU 内存以及模型和图像的大小，计算出一个合适的批量大小。
        # 较大的批量大小通常会加速训练，但需要更多的 GPU 内存。
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    # nbs = 64：nbs 代表“名义批量大小”，这是优化器调整的一个基准值。
    # 它指示了理想的批量大小，以便优化器在这个批量大小下表现最佳。在这种情况下，名义批量大小被设为 64。
    nbs = 64  # nominal batch size
    # accumulate 是梯度累积的次数。梯度累积是一个在显存受限时有效的技术，它允许模型在多个小批量上计算梯度，然后一次性更新权重。
    # 这段代码通过将名义批量大小除以当前批量大小，来计算梯度累积的次数。
    # 例如，如果实际批量大小为 32，那么需要累积 64 / 32 = 2 次梯度。
    # 如果批量大小比名义批量大小大，则累积次数为 1。
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 调整权重衰减 (weight_decay)
    # 这里调整了 weight_decay 的值。weight_decay 是一种正则化技术，用于防止模型过拟合。
    # 权重衰减是根据实际的批量大小和累积次数进行了缩放。
    # 因为 weight_decay 的效果通常与批量大小有关，所以我们按比例调整它以适应当前的批量大小。
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    # smart_optimizer 是一个函数，用来创建并返回一个优化器实例。该优化器将用于训练过程中的参数更新。
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler 学习率调度器
    # 余弦退火学习率调度 (opt.cos_lr)
    if opt.cos_lr:
        # one_cycle 是一种常见的学习率调度策略，它会将学习率在训练过程中先升高再降下来，通常用于快速收敛。
        # 这里的 hyp["lrf"] 是最终学习率（通常是较小的值），epochs 是总训练轮数。
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        # 没有启用余弦退火学习率调度（即 opt.cos_lr 为 False），则使用线性衰减的学习率调度。
        # lf(x) 是一个线性衰减学习率的计算函数，其中 x 表示当前的训练轮次（epoch）
        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            # 在训练开始时（x = 0），学习率是最大值 1.0 - hyp["lrf"]；随着训练进行（x 增加），学习率逐渐下降，直到最终接近 hyp["lrf"]。
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    # 创建一个学习率调度器（LambdaLR），它基于自定义的学习率调整函数（lf）来调整学习率。
    # lr_scheduler.LambdaLR:
    #   LambdaLR 是 PyTorch 中的一种学习率调度器，它使用一个自定义的函数 lr_lambda 来调整学习率。
    #   LambdaLR 通过在每个训练轮次（epoch）时调用 lr_lambda 函数来计算新的学习率。
    #   它非常灵活，可以让你定义任意形式的学习率衰减策略，只需提供一个根据当前轮次 x（或者其他标准）计算学习率的函数。
    # optimizer:
    #   optimizer 是你定义的优化器（如 torch.optim.SGD 或 torch.optim.Adam）。
    #   这个优化器包含了模型的所有参数以及相关的超参数（如学习率、动量等）。LambdaLR 会根据这个优化器中的学习率来调整参数。
    # lr_lambda=lf:
    #   lf 是一个函数，它定义了学习率如何随着训练轮次变化。lf 的输入通常是当前训练轮次 x，并且返回一个调整后的学习率。
    #   这个函数会根据训练轮次调整学习率，可以是余弦衰减、线性衰减等，具体取决于你在前面代码中定义的调度逻辑。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    # ModelEMA 是一个类，通常用于实现 指数加权移动平均（EMA）。
    # 它在训练过程中保持模型权重的一个平滑版本，以减少训练过程中权重波动的影响。
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    # best_fitness: 记录当前最好的训练性能，通常是指在验证集上模型的最优准确度（例如 mAP）。
    # start_epoch: 从哪里开始恢复训练，如果模型恢复自某个检查点（checkpoint），那么会用这个值来指定从哪个 epoch 开始训练。
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        # resume 为 True 表示需要从之前的训练状态恢复
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        # 删除不需要的变量,这些变量通常只在恢复时使用，不再需要，因此释放内存.
        del ckpt, csd

    # DP mode
    # 设置 DataParallel (DP) 模式来支持多 GPU 训练
    # cuda: 判断当前设备是否为 GPU。device.type != "cpu" 如果是 GPU，cuda 为 True。
    # RANK == -1: 这是在分布式训练中判断当前的进程是否是主进程（rank -1 表示单机训练或非分布式环境）。
    #   在分布式训练中，RANK 会被设置为大于等于 0 的值，表示当前进程的排名。
    # torch.cuda.device_count() > 1: 检查当前系统是否有多个 GPU。如果 GPU 数量大于 1，说明有多卡训练的条件。
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        # 如果条件成立，即存在多个 GPU，且没有使用分布式训练，程序会输出一条警告信息，
        # 提示用户 DataParallel (DP) 模式不是推荐的多 GPU 训练方式。
        # 推荐用户使用 Distributed Data Parallel (DDP)，因为 DDP 在多 GPU 训练中的性能和效率比 DP 更好。
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        # torch.nn.DataParallel 是 PyTorch 中的一种多 GPU 并行训练的方式。
        # 它会将模型的输入数据划分成多个子批次（sub-batch），然后在每个 GPU 上进行前向传播和反向传播，
        # 最后将梯度汇总并更新模型参数。
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 同步批归一化-在多 GPU 分布式训练时，确保所有 GPU 上的批归一化层使用同步方式，从而提升训练的稳定性和收敛效果
    if opt.sync_bn and cuda and RANK != -1:
        # 将模型中的所有 BatchNorm 层转换为同步批归一化层（SyncBatchNorm）。
        # 这意味着在多 GPU 环境下，所有 GPU 上的 BatchNorm 层将共享同一个均值和方差统计值，而不是每个 GPU 单独计算。
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path, # 训练数据的路径
        imgsz, # 训练数据的路径
        batch_size // WORLD_SIZE, # 每个 GPU 的批量大小
        gs, # grid size
        single_cls, # 是否为单类训练（用于处理只有一个类别的情况）
        hyp=hyp, # 超参数配置
        augment=True, # 是否进行数据增强
        cache=None if opt.cache == "val" else opt.cache, # 是否缓存数据,可以把所有的数据都缓存下来
        rect=opt.rect, # 是否进行矩形训练（保持原始长宽比）
        rank=LOCAL_RANK, # 当前进程的 rank（在分布式训练中使用）
        workers=workers, # 用于加载数据的线程数
        image_weights=opt.image_weights,  # 是否使用图像权重
        quad=opt.quad, # 是否启用四元组数据增强
        prefix=colorstr("train: "), # 用于日志输出的前缀
        shuffle=True, # 是否打乱数据
        seed=opt.seed, # 随机种子
    )
    # 将所有数据集中的标签合并成一个大的 NumPy 数组
    labels = np.concatenate(dataset.labels, 0)
    # 通过获取标签数组中最大值来获得数据集中最大的标签类别。
    # labels[:, 0] 选取标签数组中的第一列（即类别标签），然后调用 .max() 得到最大的类标签。
    # mlc 代表最大类标签，用于后续验证类标签的合法性。
    mlc = int(labels[:, 0].max())  # max label class
    # 这个断言确保数据集中的最大标签类不会超过模型定义的类别数
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
    Parse command-line arguments for YOLOv5 training, validation, and testing.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.

    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        print(opt)
        ```

    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Runs the main entry point for training or hyperparameter evolution with specified options and optional callbacks.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None

    Note:
        For detailed usage, refer to:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mosaic (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),  # segment copy-paste (probability)
        }

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def generate_individual(input_ranges, individual_length):
    """
    Generate an individual with random hyperparameters within specified ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene (hyperparameter).
        individual_length (int): The number of genes (hyperparameters) in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Example:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```

    Note:
        The individual returned will have a length equal to `individual_length`, with each gene value being a floating-point
        number within its specified range in `input_ranges`.
    """
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    """
    Execute YOLOv5 training with specified options, allowing optional overrides through keyword arguments.

    Args:
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-low.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a
            value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/ hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/train'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        ```python
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
