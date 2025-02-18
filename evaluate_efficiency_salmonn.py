
import sys
from pathlib import Path
import torch
import json
import time
import numpy as np
import argparse
import gc
import subprocess
from transformers import DynamicCache
from tqdm import tqdm

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import WhisperFeatureExtractor

# From trainer
# sys.path.append(str(Path().parent / "audiolm-trainer"))
from config import Config
from dataset import SALMONNDataset
from utils import get_dataloader, prepare_sample
from models.salmonn import SALMONN
from salmonn_utils import load_preprocessor, load_model

class MockDataset(SALMONNDataset):
    def __init__(self, cfg, sr, audio_length, dataset_length):
        self.sr = sr
        self.audio_length = audio_length
        self.dataset_length = dataset_length
        self.prefix = cfg.config.datasets.prefix
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(
            cfg.config.datasets.whisper_path
        )
        self.random_sample = np.random.randn(self.sr * self.audio_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        audio = self.random_sample.copy()
        spectrogram = self.wav_processor(
            audio, sampling_rate=self.sr, return_tensors="pt"
        )["input_features"].squeeze()
        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": "test",
            "task": "asr",
            "Q": "",
            "id": idx,
        }

    @staticmethod
    def make_mock_dataloader(cfg, sr, audio_length, dataset_length=100):
        dataset = MockDataset(cfg, sr, audio_length, dataset_length)
        return get_dataloader(
            dataset, cfg.config.run, is_train=False, use_distributed=False
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        help="path to configuration file",
        default="configs/salmonn_llama1b/evaluation.yaml",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--num_it", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=10)
    return parser.parse_args()


def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = int(result.strip().split("\n")[0])
    return gpu_memory

def format_time_memory(name, time, memory=None):
    if memory is not None:
        return f"{name}: {time:.4f} seconds, Memory: {memory/1024**3:.4f} GB"
    else :
        return f"{name}: {time:.4f} seconds"
    
def model_inference(cfg, samples, test_prompt, salmonn,llm):
    # TTFT
    start_time = time.time()

    batch_size = samples['spectrogram'].shape[0]
    spectrogram = samples["spectrogram"]
    raw_wav = samples.get("raw_wav", None)
    audio_padding_mask = samples.get("padding_mask", None)

    # Measure Speech Encoder Time
    gpu_memory_start = torch.cuda.memory_allocated()
    speech_encoder_start = time.time()
    """speech_embeds, speech_atts = salmonn.encode_speech(
        spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
    )
    """
    ## speech_embeds 좀 더 자세히 분석
    with salmonn.maybe_autocast():
        whisper_encoder_start = time.time()
        speech_embeds = salmonn.speech_encoder(spectrogram, return_dict=True).last_hidden_state
        whisper_encoder_time = time.time() - whisper_encoder_start


        if salmonn.beats_path and raw_wav is not None:
            beats_extract_start = time.time()
            audio_embeds, _ = salmonn.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            beats_extract_time = time.time() - beats_extract_start
        else:
            audio_embeds = None

        feature_merge_start = time.time()               
        speech_embeds, speech_atts =  salmonn._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)
        feature_merge_time = time.time() - feature_merge_start

    gpu_memory_end = torch.cuda.memory_allocated()
    speech_encoder_time = time.time() - speech_encoder_start
    speech_encoder_memory = gpu_memory_end - gpu_memory_start

    # Measure QFormer Time
    gpu_memory_start = torch.cuda.memory_allocated()
    qformer_start = time.time()
    prompts = [test_prompt[task] for task in samples["task"]]
    templated_prompts = [
        cfg.config.model.prompt_template.format(prompt) for prompt in prompts
    ]
    speech_embeds, speech_atts = salmonn.prompt_wrap(
        speech_embeds, speech_atts, templated_prompts, multi_prompt=True
    )
    qformer_time = time.time() - qformer_start
    gpu_memory_end = torch.cuda.memory_allocated()
    qformer_memory = gpu_memory_end - gpu_memory_start

    # Measure LLM Time (Forward Pass)
    gpu_memory_start = torch.cuda.memory_allocated()
    llm_start = time.time()
    bos = (
        torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        )
        * salmonn.llama_tokenizer.bos_token_id
    )
    bos_embeds = (
        llm.model.embed_tokens(bos)
        if not salmonn.lora
        else llm.model.model.embed_tokens(bos)
    )
    atts_bos = speech_atts[:, :1]

    speech_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
    speech_atts = torch.cat([atts_bos, speech_atts], dim=1)

    outputs = llm.model(
        inputs_embeds=speech_embeds,
        attention_mask=speech_atts,
    )

    llm_time = time.time() - llm_start
    gpu_memory_end = torch.cuda.memory_allocated()
    llm_memory = gpu_memory_end - gpu_memory_start

    end_time = time.time()
    ttft = end_time - start_time

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    # TPOT
    start_time = time.time()
    with torch.no_grad():
        _ = llm.model(next_token, past_key_values=past_key_values, use_cache=True)
    end_time = time.time()
    tpot = end_time - start_time

    inference_time = ttft + tpot

    generate_cfg = cfg.config.generate
    outputs = llm.model.generate(
        inputs_embeds=speech_embeds,
        pad_token_id=llm.config.eos_token_id[0],
        max_new_tokens=generate_cfg.get("max_new_tokens", 200),
        num_beams=generate_cfg.get("num_beams", 4),
        do_sample=generate_cfg.get("do_sample", False),
        min_length=generate_cfg.get("min_length", 1),
        temperature=generate_cfg.get("temperature", 1.0),
        top_p=generate_cfg.get("top_p", 0.9),
        repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
        length_penalty=generate_cfg.get("length_penalty", 1.0),
        attention_mask=speech_atts,
    )

    return (
        inference_time,
        ttft,
        tpot,
        speech_encoder_time,
        whisper_encoder_time,
        feature_merge_time,
        beats_extract_time,
        qformer_time,
        llm_time,
        speech_encoder_memory,
        qformer_memory,
        llm_memory,
        outputs
    )

def main(args):
    cfg = Config(args)

    print("Force batch size as 1")
    cfg.config.run.batch_size_eval = 1
    
    # Load model
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # Load dataset
    with open("prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    data_config = cfg.config.datasets
    valid_datasets = SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path)
    valid_loader = get_dataloader(valid_datasets, cfg.config.run, is_train = False, use_distributed=False)

    # Measure memory and latency
    memory_usages = []
    inference_times = []
    ttfts = []
    tpots = []
    speech_encoder_times = []
    whisper_encoder_times = []
    feature_merge_times = []
    beats_extract_times = []
    qformer_times = []
    llm_times = []
    speech_encoder_memories = []
    qformer_memories = []
    llm_memories = []
    data_load_times = []
    data_load_memories = []
    outputs = []
    batch_size = 1 ## hyperparameter
    cfg.config.run.bathc_size_eval = batch_size 


    data_load_start = time.time()
    iter = 0
    warmup_iter = 10
    for sample in tqdm(valid_loader):
        sample = prepare_sample(sample)
        data_load_time = time.time() - data_load_start
        torch.cuda.synchronize()
        with torch.no_grad():
            (
                inference_time,
                ttft,
                tpot,
                speech_encoder_time,
                whisper_encoder_time,
                feature_merge_time,
                beats_extract_time,
                qformer_time,
                llm_time,
                speech_encoder_memory,
                qformer_memory,
                llm_memory,
                output
            ) = model_inference(
                cfg,
                sample,
                test_prompt,
                salmonn_preprocessor,
                llama_model
            )
        torch.cuda.synchronize()
        after_memory_allocated = torch.cuda.max_memory_allocated()

        output = output.detach().cpu()
        torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
        gc.collect()

        if iter>=warmup_iter:
            memory_usages.append(after_memory_allocated)
            inference_times.append(inference_time)
            ttfts.append(ttft)
            tpots.append(tpot)
            speech_encoder_times.append(speech_encoder_time)
            whisper_encoder_times.append(whisper_encoder_time)
            feature_merge_times.append(feature_merge_time)
            beats_extract_times.append(beats_extract_time)
            speech_encoder_memories.append(speech_encoder_memory)
            qformer_times.append(qformer_time)
            qformer_memories.append(qformer_memory)
            llm_times.append(llm_time)
            llm_memories.append(llm_memory)
            data_load_times.append(data_load_time)
            outputs.append(output)

            average_memory_usage = np.mean(memory_usages)/batch_size
            average_inference_time = np.mean(inference_times)/batch_size
            average_ttft = np.mean(ttfts)/batch_size
            average_tpot = np.mean(tpots)/batch_size

            average_data_load_time = np.mean(data_load_times)/batch_size
            average_speech_encoder_time = np.mean(speech_encoder_times)/batch_size
            average_whisper_encoder_time = np.mean(whisper_encoder_times)/batch_size
            average_feature_merge_time = np.mean(feature_merge_times)/batch_size
            average_beats_extract_time = np.mean(beats_extract_times)/batch_size

            average_sppech_encoder_memory = np.mean(speech_encoder_memories)/batch_size
            average_qformer_time = np.mean(qformer_times)/batch_size
            average_qformer_memory = np.mean(qformer_memories)/batch_size
            average_llm_time = np.mean(llm_times)/batch_size
            average_llm_memory = np.mean(llm_memories)/batch_size

            if iter%10 == 0 :
                print("\n=== Performance Metrics ===")
                print(f"├─ Average memory used during inference: {average_memory_usage/1024**3:.4f} GB")
                print(f"├─ {format_time_memory('Total Inference Time', average_inference_time)}")
                print(f"│  ├─ {format_time_memory('Time to First Token (TTFT)', average_ttft)}")
                print(f"│  └─ {format_time_memory('Time per Output Token (TPOT)', average_tpot)}")
                print(f"│")
                print(f"├─ {format_time_memory('Data Loading',average_data_load_time)}")
                print(f"├─ {format_time_memory('Speech Encoder', average_speech_encoder_time, average_sppech_encoder_memory)}")
                print(f"│  ├─ {format_time_memory('Whisper Encoder', average_whisper_encoder_time)}")
                print(f"│  ├─ {format_time_memory('Beats Extract', average_beats_extract_time)}")
                print(f"│  └─ {format_time_memory('Feature Merge', average_feature_merge_time)}")
                print(f"│")
                print(f"├─ {format_time_memory('QFormer', average_qformer_time, average_qformer_memory)}")
                print(f"└─ {format_time_memory('LLM', average_llm_time, average_llm_memory)}")

        iter+=1
        data_load_start = time.time()


if __name__ == "__main__":
    args = parse_args()
    main(args)
