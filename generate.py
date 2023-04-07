import copy
import logging
from argparse import ArgumentParser
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import pathlib
import sys
import yaml

import torch
import transformers
from transformers import GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def read_config(path):
    # read yaml and return contents 
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def setup_model(model_name_or_path, cache_dir, bf16):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        # device_map="auto", 
        # torch_dtype=torch.float16 if not bf16 else torch.bfloat16,
    )
    # model to device
    model = model.to(device)
    model = model.to(torch.bfloat16) if bf16 else model.to(torch.float16)

    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        # model_max_length=model_max_length,
    )

    model = model.to(torch.bfloat16)
        
    # if "llama" in model_name_or_path:
    added_tokens = tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    print(f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Detecting torch.compile(model)...")
        model = torch.compile(model)
        
    return model, tokenizer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate(tokenizer, 
             prompt, 
             model,
             config):
    if config["add_instruction"]:
        prompt = generate_prompt(prompt, input=None)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    generation_config = GenerationConfig(**config)

    print(generation_config)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = outputs.sequences[0]
    decoded = tokenizer.decode(s,skip_special_tokens=True).strip()

    return decoded

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--decode_config", type=str, default='./decode_config.yaml') 
    
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--bf16", action="store_true", default=False)

    args = parser.parse_args()
    args.cache_dir = args.model

    if args.prompt is None:
        raise ValueError("Prompt is required either in config or as argument")

    config = read_config(args.decode_config)

    print("Setting up model")
    model, tokenizer = setup_model(args.model, args.cache_dir, args.bf16)

    print("Generating")
    start = time.time()
    generation = generate(
        tokenizer, 
        args.prompt, 
        model, 
        config
        )
    print(f"Done in {time.time() - start:.2f}s")
    print(generation)