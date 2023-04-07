#!/bin/bash

# model_name_or_path="/home/quansun84/stanford_alpaca/output/alpaca-2/checkpoint-2000"
# model_name_or_path="/home/quansun84/stanford_alpaca/output/alpaca-4/checkpoint-4000"
model_name_or_path="/home/quansun84/stanford_alpaca/output/test_clean_alpaca/checkpoint-2000"
# model_name_or_path="/share/project/qiying/model_cache/LLaMA/hf/llama-7b"
# PROMPT="I am a student. I am interested in learning about the history of Stanford University. What is the history of Stanford University"
# PROMPT="Provide instructions for the given exercise. Leg Raises"
# PROMPT="A color description has been provided. Find the CSS code associated with that color. A light red color with a medium light shade of pink"
# PROMPT="Come up with an interesting idea for a new movie plot. Your plot should be described with a title and a summary."
# PROMPT="Reverse a string in python."
PROMPT="Write a script to reverse a string in Python."
# PROMPT="Write me a poem about the fall of Julius Ceasar into a ceasar salad in iambic pentameter."
# PROMPT="What is a three word topic describing the following keywords: baseball, football, soccer:"
# PROMPT="List 10 dogs."


python generate.py \
    --model $model_name_or_path \
    --prompt "$PROMPT"
