import logging
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
# import datasets
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from utils import get_last_non_padded_tokens, compute_block_distances
from typing import Optional
from peft import PeftModel

logging.basicConfig(level=logging.INFO)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

def load_dataset_from_tensor(dir_path, padding_value=0.0):
    '''
    .pt파일이 들어있는 디렉토리를 검색하여 .pt 파일을 불러오기
    '''
    if os.path.isdir(dir_path):
        print(f"디렉토리에서 모든 .pt 파일을 불러옵니다: {dir_path}")
        tensor_data = []
        max_length = 0
        for file_name in sorted(os.listdir(dir_path)):
            if file_name.endswith(".pt"):
                full_path = os.path.join(dir_path, file_name)
                tensor = torch.load(
                    full_path, map_location=torch.device('mps'))
                seq_length = tensor.size(1)    
                max_length = max(max_length, seq_length)
                tensor_data.append(tensor)
                
        print(f"{len(tensor_data)}개의 아이템들을 디렉토리에서 불러왔습니다.")
        print(f"최대 길이: {max_length}")
        
        padded_tensors = []
        
        for tensor in tensor_data:
            if len(tensor.size()) == 2:
                seq_length = tensor.size(1)
                padding = (0, max_length - seq_length)
                padded_tensors.append(F.pad(tensor, padding, value=padding_value))
            elif len(tensor.size()) == 3:
                seq_length = tensor.size(1)
                padding = (0, 0, 0, max_length - seq_length)
                padded_tensors.append(F.pad(tensor, padding, value=padding_value))
        
        return torch.stack(padded_tensors)
    else:
        raise FileNotFoundError(f"디렉토리 찾을 수 없음: {dir_path}")


def main(model_path: str, dataset: str, dataset_column: str, batch_size: int, max_length: int,
         layers_to_skip: int, dataset_size: Optional[int] = None, dataset_subset: Optional[str] = "eval", tensor_file: Optional[str] = None, attention_file: Optional[str] = None):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")

    # if resource is a problem
    # 일단 BitsAndBytesConfig는 MacOS에서는 지원하지 않기에 주석 처리
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    #                                         bnb_4bit_use_double_quant=True,
    #                                         bnb_4bit_quant_type="nf4",
    #                                         bnb_4bit_compute_dtype=torch.bfloat16)
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=True, 
        r=4, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 device_map="auto",
                                                 offload_folder="offload", 
                                                 offload_state_dict=True,
                                                 torch_dtype=torch.float32,
                                                 #  quantization_config=
                                                 # quantization_config,
                                                 # # macOS에서는 지원하지 않음
                                                 low_cpu_mem_usage=True,
                                                 output_hidden_states=True)
    lora_weights_path = "../pth_weight/salmonn_3b_nota.pth"
    state_dict = torch.load(lora_weights_path, map_location="cpu")
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
        
    base_model.load_state_dict(state_dict, strict=False)
    
    peft_model = get_peft_model(base_model, config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    peft_model.eval()

    input_embeds = load_dataset_from_tensor(tensor_file)
    attention = load_dataset_from_tensor(attention_file)
    
    dataloader = DataLoader(
        dataset=TensorDataset(input_embeds, attention),
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize a list to store distances for each block across the dataset
    all_distances = [[] for _ in range(base_model.config.num_hidden_layers - layers_to_skip)]
    # hidden_size_mapping = nn.Linear(2048, 3072).to(device)


    for batch in tqdm(dataloader, desc="Processing batches"):
        input_embeds = batch[0].to(device).to(torch.float32)
        # input_embeds = hidden_size_mapping(input_embeds)
        input_embeds = input_embeds.squeeze(1)
        attention = batch[1].squeeze(1)
        
        with torch.no_grad():
            outputs = peft_model(input_ids=None, inputs_embeds=input_embeds, attention_mask=attention)
        hidden_states = outputs.hidden_states
        last_non_padded_hidden_states = get_last_non_padded_tokens(hidden_states, attention)
        last_non_padded_hidden_states = last_non_padded_hidden_states[1:]
        
        assert len(last_non_padded_hidden_states) == peft_model.config.num_hidden_layers, "Length of last_non_padded_hidden_states  \
        does not match expected number of hidden layers."
        
        distances = compute_block_distances(last_non_padded_hidden_states, layers_to_skip)
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

    # Calculate average distances for each block
    average_distances = [np.mean(block_distances) for block_distances in all_distances]

    # Write the average distances to a CSV file and compute the minimum average distance
    min_distance = float('inf')  # Initialize with infinity
    min_distance_layer = 0  # Initialize with an impossible value

    with open('layer_distances.csv', 'w', newline='') as csvfile:
        fieldnames = ['block_start', 'block_end', 'average_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, avg_dist in enumerate(average_distances):
            # Write each row to the CSV
            writer.writerow({
                'block_start': i + 1,  # layer indices are 1-based in the paper
                'block_end': i + 1 + layers_to_skip,
                'average_distance': avg_dist
            })
            
            if avg_dist < min_distance:
                min_distance = avg_dist
                min_distance_layer = i + 1  

    # Log the layer with the minimum average distance
    logging.info(f"Layer {min_distance_layer} to {min_distance_layer + layers_to_skip} has the minimum average distance of {min_distance}. Consider examining this layer more closely for potential optimization or removal.")
    logging.info("Layer distances written to layer_distances.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model analysis.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--dataset_column", type=str, required=True, help="The specific column of the dataset to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of the tokenized input.")
    parser.add_argument("--layers_to_skip", type=int, required=True, help="Number of layers to skip.")
    parser.add_argument("--dataset_size", type=int, help="Optional argument to specify the size of the dataset.")
    parser.add_argument("--dataset_subset", type=str, default="eval", help="Subset of the dataset to use (e.g., 'train', 'eval').")
    parser.add_argument("--device", type=str, help="Device to run the model on ('cpu', 'cuda').")
    parser.add_argument("--tensor_file", type=str, help="Path to the tensor file.")
    parser.add_argument("--attention_file", type=str, help="Path to the attention file.")

    args = parser.parse_args()

    main(args.model_path, args.dataset, args.dataset_column, args.batch_size,
         args.max_length, args.layers_to_skip, args.dataset_size, args.dataset_subset, args.tensor_file, args.attention_file)
