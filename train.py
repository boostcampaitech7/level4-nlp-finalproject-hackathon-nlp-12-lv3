# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import gc

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner
from cli_yaml import update_output_dir, update_dataset_paths

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["stage1", "stage2", "merged"], 
        required=True, 
        help="Choose training stage: stage1, stage2, or Merged"
    )
    
    parser.add_argument(
        "--quant", 
        type=str, 
        choices=["4bit", "8bit"], 
        required=False, 
        help="Choose 4bit quantization or 8bit quantization"
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def run_stage(cfg, model, datasets, job_id, dryrun, stage, base_output_dir):
    """
    단일 stage의 training을 실행하는 함수.
    output_dir과 datasets 경로 업데이트 후 Runner를 생성하고 train() 실행.
    """
    logging.info(f"Starting {stage.capitalize()} Training...")
    
    # output_dir 업데이트
    cfg = update_output_dir(cfg, stage, base_output_dir)
    # datasets 경로 업데이트 (필요한 경우)
    cfg = update_dataset_paths(cfg, stage)
    
    runner = Runner(cfg, model, datasets, job_id, dryrun, stage)
    runner.train()
    return runner

def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.'''

    # Wandb logger        
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank == 0:
        wandb.login()
        wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # build model

    if not args.dryrun:
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # build runner
    base_output_dir = cfg.config.run.output_dir  # 원본 output_dir 저장
    if args.stage == "Merged":
        logging.info("Starting Merged Training (Stage 1 + Stage 2)...")
        runner = run_stage(cfg, model, datasets, job_id, args.dryrun, stage="stage2", base_output_dir=base_output_dir)
        # Stage 1 Training
        ckpt_path = runner.best_path
        # 메모리 정리
        logging.info("Clearing memory after Stage 1...")
        del runner
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Stage 2를 위해 모델 재로딩 및 config 업데이트
        logging.info("Loading Stage 1 Model for Stage 2...")
        cfg.config.model.ckpt = ckpt_path
        model = load_model(model_config)  # 새 모델 생성
        
        # Stage 2 Training 실행
        runner = run_stage(cfg, model, datasets, job_id, args.dryrun, stage="stage2", base_output_dir=base_output_dir)

    else :
        cfg.config.run.output_dir = base_output_dir + "/" + args.stage
        logging.info(f"Starting Stage 1 Training...")
        runner = run_stage(cfg, model, datasets, job_id, args.dryrun, stage="stage1", base_output_dir=base_output_dir)


if __name__ == "__main__":
    main()