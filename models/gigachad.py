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

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM, ATModel
from .modeling_whisper import WhisperModel
from .salmonn import SALMONN
from .utils import StoppingCriteriaSub


class GIGACHAD(SALMONN):
    def __init__(
        self,
        *args,
        n_layer=32, 
        rep_dim=1280, 
        tltr_mode='tl_down_tr_512_1_8',
        
        use_at_model=True,
        freeze_at_model = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        if use_at_model:
            self.at_model = ATModel(
                n_layer=n_layer, 
                rep_dim=rep_dim, 
                mode=tltr_mode
            )
        else:
            raise ValueError("ATModel is required for GIGACHAD")

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            # Whisper 중간 레이어 추출
            encoder_outputs = self.speech_encoder(
                spectrogram, 
                return_dict=True
            )
            speech_embeds = encoder_outputs.last_hidden_state  # 최종 레이어
            all_hidden = encoder_outputs.hidden_states  # 모든 레이어 [num_layers+1, B, T, D]

            # ATModel 입력 형식 맞춤 (B, num_layers, T, D)
            hidden_states = torch.stack(all_hidden[1:], dim=1)  # 임베딩 제외
            B, num_layers, T, D = hidden_states.shape
            audio_embeds = self.at_model(hidden_states)  # [B, T, D]
            audio_embeds = self.ln_audio(audio_embeds)

        return self._encode_auditory_feature(speech_embeds, audio_embeds)

    @classmethod
    def parse_config(cls, config):
        config_dict = super().parse_config(config)
        
        use_at_model = config.get("use_at_model", True)
        freeze_at_model = config.get("freeze_at_model", False)
        
        n_layer = config.get("n_layer", 32)
        rep_dim = config.get("rep_dim", 1280)  
        tltr_mode = config.get("tltr_mode", "tl_down_tr_512_1_8")

        config_dict.update({
            "use_at_model": use_at_model,
            "freeze_at_model": freeze_at_model,
            "n_layer": n_layer,
            "rep_dim": rep_dim,
            "tltr_mode": tltr_mode,
        })
        
        return config_dict