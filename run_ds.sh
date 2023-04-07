#!/bin/bash

# pip install flash_attn
# pip install -e /home/quansun84/transfomers
# pip install datasets
# pip install openai
# pip install zstandard

torchrun --nproc_per_node=2 --master_port=12355 train_mem.py \
    --model_name_or_path="/share/project/qiying/model_cache/LLaMA/hf/llama-7b" \
    --data_path /share/project/quansun/Pile/ \
    --data_source pretrain \
    --bf16 True \
    --report_to="tensorboard" \
    --output_dir="/home/quansun84/stanford_alpaca/output/test_pile" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config.json \
    --tf32 True \
    --model_max_length 1024 \
    --max_steps 10000 \
    --gradient_checkpointing True \
    --run_name "test_pile"

    # --max_steps 1000000 \
    # --data_path /share/project/quansun/Pile/
    # --data_path /home/quansun84/stanford_alpaca/alpaca_data_cleaned.json \

    # --report_to="wandb,tensorboard" \
    # --data_path ./alpaca_data.json \