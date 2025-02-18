# cli_yaml.py
import json

train_template = """\
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

model:
  # paths
  llama_path: "{llama_path}"
  whisper_path: "{whisper_path}"
  beats_path: "{beats_path}"

  token: "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" # Use hf token to access gated repositories
  only_preprocessor: False

  ckpt: "{ckpt}" # model used for decoding

  freeze_whisper: True
  freeze_beats: True
  
  # window-level Q-Former
  use_speech_Qformer: True
  freeze_speech_QFormer: False
  window_level_Qformer: True
  num_speech_query_token: 1
  second_per_window: 0.333333
  second_stride: 0.333333

  speech_llama_proj_model: ""
  freeze_speech_llama_proj: False

  # LoRA
  lora: True
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.1
  
  #quantization
  quant_4bit : {quant_4bit}
  quant_8bit : {quant_8bit}

  multi_prompt: True
  prompt_template: "USER: {{}}\\nASSISTANT:"
  prompt_path: "prompts/train_prompt.json"
  test_prompt_path: "prompts/test_prompt.json"
  max_txt_len: 300
  end_sym: "{end_sym}"

datasets:
  prefix: "/data/home/datasets"
  train_ann_path: "data/stage1/train/train_100percent.json"
  valid_ann_path: "data/stage1/val/val.json"
  test_ann_path: "data/stage1/test/test.json"
  whisper_path: "{whisper_path}"

run:
  # log & settings
  seed: 42
  output_dir: "{output_dir}"
  evaluate: {evaluate} # if True, only evaluate model on test data
  exp_name: "{exp_name}"

  log_freq: 5
  epoch_based: {epoch_based}
  iters_per_epoch: {iters_per_epoch}
  accum_grad_iters: {accum_grad_iters}
  batch_size_train: {batch_size_train}
  batch_size_eval: {batch_size_eval}
  num_workers: 8

  device: "cuda"
  use_distributed: {use_distributed}
  run_device : {run_device}
  amp: True
  world_size: {world_size}
  dist_url: "env://"

  # optimizer & scheduler
  optims:
    max_epoch: {max_epoch}
    warmup_steps: {warmup_steps}
    warmup_start_lr: 1e-6
    init_lr: 3e-5
    min_lr: 1e-5
    weight_decay: 0.05
    beta2: 0.999
"""

eval_template = """\
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

model:
  llama_path: "{llama_path}"
  whisper_path: "{whisper_path}"
  beats_path: "{beats_path}"

  token: "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" # Use hf token to access gated repositories
  only_preprocessor: False

  ckpt: "{ckpt}" # model used for decoding

  freeze_whisper: True
  freeze_beats: True
  
  # window-level Q-Former
  use_speech_Qformer: True
  freeze_speech_QFormer: False
  window_level_Qformer: True
  num_speech_query_token: 1
  second_per_window: 0.333333
  second_stride: 0.333333

  speech_llama_proj_model: ""
  freeze_speech_llama_proj: False

  # LoRA
  lora: True
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.1
  
  #quantization
  quant_4bit : {quant_4bit}
  quant_8bit : {quant_8bit}

  multi_prompt: True
  prompt_template: "USER: {{}}\\nASSISTANT:"
  prompt_path: "prompts/train_prompt.json"
  test_prompt_path: "prompts/test_prompt.json"
  max_txt_len: 300
  end_sym: "{end_sym}"

generate:
  max_new_tokens: 200
  num_beams: 4
  do_sample: False
  min_length: 1
  temperature: 1.0
  top_p: 0.9
  repetition_penalty: 1.0
  length_penalty: 1.0
  end_sym: "{end_sym}"

datasets:
  prefix: "/data/home/datasets"
  train_ann_path: "data/stage1/train/train_100percent.json"
  valid_ann_path: "data/stage1/val/val.json"
  test_ann_path: "data/stage1/test/test.json"
  whisper_path: "{whisper_path}"

run:
  batch_size_eval: {batch_size_eval}
  num_workers: 8
  device: "cuda"
"""

def update_output_dir(cfg, stage, base_output_dir):
    """
    config의 output_dir을 stage에 따라 업데이트하는 함수.
    """
    cfg.config.run.output_dir = f"{base_output_dir}/{stage}"
    return cfg

def update_dataset_paths(cfg, stage):
    """
    config에 있는 datasets의 경로를 stage별로 업데이트하는 함수.
    예시에서는 "stage1"이 포함된 경로를 입력받은 stage로 치환.
    """
    if "stage1" in cfg.config.datasets.train_ann_path:
        cfg.config.datasets.train_ann_path = cfg.config.datasets.train_ann_path.replace("stage1", stage)
    if "stage1" in cfg.config.datasets.valid_ann_path:
        cfg.config.datasets.valid_ann_path = cfg.config.datasets.valid_ann_path.replace("stage1", stage)
    if "stage1" in cfg.config.datasets.test_ann_path:
        cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path.replace("stage1", stage)
    return cfg
  