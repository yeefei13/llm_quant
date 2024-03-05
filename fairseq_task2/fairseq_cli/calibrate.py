#!/usr/bin/env python3 -u
import os
import sys
import logging
from argparse import Namespace

import torch
from fairseq_cli.omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.calibration")

def calibrate(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    reset_logging()

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=eval(cfg.common_eval.model_overrides),
    )
    model = models[0]  # For calibration, we'll focus on the first model

    model.eval()
    if use_cuda:
        model.cuda()

    # Setup data loading
    task.load_dataset(cfg.dataset.gen_subset)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(itr, log_format='tqdm', enable=not cfg.common.no_progress_bar)

    # Calibration loop
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _ = model(**sample['net_input'])

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), calibrate)

if __name__ == "__main__":
    cli_main()
