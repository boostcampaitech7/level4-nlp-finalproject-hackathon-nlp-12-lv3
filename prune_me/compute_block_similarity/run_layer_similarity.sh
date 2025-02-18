#!/bin/bash

# This Bash script runs the Python script with arguments

# Run the Python script with command-line arguments
python layer_similarity.py --model_path "meta-llama/Llama-3.2-1B-Instruct" \
                      --dataset "arcee-ai/sec-data-mini" \
                      --dataset_column "text" \
                      --batch_size 1 \
                      --max_length 1024 \
                      --layers_to_skip 8 \
                      --dataset_size 4000 \
                      --dataset_subset "train" \
                      --tensor_file "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/datasets/output_inps_1" \
                      --attention_file "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/datasets/output_attns_1"