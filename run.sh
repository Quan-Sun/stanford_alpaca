#!/bin/bash

# pip install flash_attn
# pip install -e /home/quansun84/transformers
# pip install datasets
# pip install openai
# pip install zstandard

torchrun --nproc_per_node=2 --master_port=12355 train_mem.py \
    --model_name_or_path="/share/project/qiying/model_cache/LLaMA/hf/llama-7b" \
    --data_path /home/quansun84/stanford_alpaca/alpaca_gpt4_data.json \
    --data_source instruction \
    --report_to="tensorboard" \
    --bf16 True \
    --output_dir="/home/quansun84/stanford_alpaca/output/test_gpt4_data" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --torch_compile True \
    --run_name "test_gpt4_data"

    # --num_train_epochs 3 \
    # --save_steps 2000 \
    # --learning_rate 2e-5 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \

    # --max_steps 1000000 \
    # --data_path /share/project/quansun/Pile/
    # --data_path /home/quansun84/stanford_alpaca/alpaca_data_cleaned.json \

    # --report_to="wandb,tensorboard" \
    # --dat