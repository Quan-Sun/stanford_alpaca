"""
LLaMA model from transformers, following stanford's Alpaca
"""

from typing import Dict
import sys
import os

import torch
from apex.normalization import FusedLayerNorm

from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import torch.nn as nn
import transformers
# from src.open_clip.transformer import AttentionalPooler

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def llama_model(model_max_length=512, lm=True):

    if os.getenv('MLP_WORKER_NUM') is None:  # 九鼎 Platform
        model_path = "/share/project/qiying/model_cache/LLaMA/hf/llama-7b"
    else:  # 火山
        model_path = "/baai-health/baai-vision/qiying/models/llama/hf/llama-7b"

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length, # 512
        padding_side="right",
        use_fast=False,
    )

    if lm:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_path,
        )
    else:
        model = transformers.LlamaModel.from_pretrained(
            model_path,
        )

    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            if lm:
                output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            if lm:
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            if lm:
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    return model, tokenizer


class LLaMA_LM(nn.Module):

    def __init__(self):
        super(LLaMA_LM, self).__init__()

        self.lm, self.tokenizer = llama_model(lm=True)

        self.config = self.lm.config
        self.lm.config.d_model = self.lm.config.hidden_size

        self.lm.half()

        self.prompt = None

    def forward(self, image_embeds, text_input, input_mask, text_output=None, output_mask=None):
        """

        :param image_embeds: [B, n_query, C], after projected into Language shape
        :param text_input: [B, seq_len]
        :param input_mask: [B, seq_len]
        :return:
        """
        # img attn mask
        attn_img = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        targets = text_input.masked_fill(
            text_input == self.tokenizer.pad_token_id, -100
        )

        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(attn_img.size(), dtype=torch.long).to(image_embeds.device).fill_(-100)
        )  # img targets
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.lm.model.embed_tokens(text_input)
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([attn_img, input_mask], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.lm.gradient_checkpointing_enable()


# class LLaMA_AttnPool(nn.Module):
#     """
#     LLaMA model with an attention pool
#     """
    
#     def __init__(self, embed_dim, attn_pooler_heads, norm_layer=FusedLayerNorm, xattn=True):
#         super(LLaMA_AttnPool, self).__init__()

#         self.llama, self.tokenizer = llama_model(lm=False)

#         self.llama_d_model = self.llama.embed_tokens.weight.shape[-1]

#         self.attn_pool = AttentionalPooler(self.llama_d_model, self.llama_d_model,
#                                            n_head=attn_pooler_heads, n_queries=1,
#                                            norm_layer=norm_layer,
#                                            xattn=xattn
#                                            )

#         self.head = nn.Linear(self.llama_d_model, embed_dim)

#     def forward(self, text, text_attn_mask):
#         """

#         :param text: [B, seq_len]
#         :param text_attn_mask: [B, seq_len]
#                 Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
#                 - 1 for tokens that are **not masked**,
#                 - 0 for tokens that are **masked**.
#         :return:
#         """
#         # [B, seq_len] --> [B, seq_len, d_model (4096 for llama)]
#         text_features = self.llama(input_ids=text, attention_mask=text_attn_mask).last_hidden_state
#         # [B, seq_len, d_model (4096 for llama)] --> [B, 1, d_model]
#         pooled = self.attn_pool(text_features, text_attn_mask)
#         # [B, 1, d_model] --> [B, 1, embed_dim] (for contrasting)
#         pooled = self.head(pooled)
#         return pooled.squeeze(1)

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.llama.gradient_checkpointing_enable()
