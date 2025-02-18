import os
import torch

directory = './attention_mask_embeds'

for file in os.listdir(directory):
  file_path = os.path.join(directory, file)
  try:
    tensor = torch.load(file_path, map_location=torch.device("cpu"))
  except Exception as e:
    print(f"Failed to load {file_path}: {e}")