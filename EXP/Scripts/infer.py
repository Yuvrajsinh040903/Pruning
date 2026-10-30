import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
import torch
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from pathlib import Path
import numpy as np
import re
import random
from huggingface_hub import login
from transformers import AutoTokenizer


login(token="")

infer_folder = """/home/priyanka/Pruning/EXP/Data/Flores_Devtest.tsv"""

model_path = "/home/priyanka/Pruning/EXP/Experiments/EXP_CKA_Wed_Sep_16_00-30-28_2026_wo_rec/Pruned_Model_28_to_24_CKA/pruning_recovery_oft/final_recovery_bpcc_oft"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

save_tsv_file = Path(model_path).parents[0]/"inference.tsv"

# For base
# save_tsv_file = f"/home/priyanka/Pruning/EXP/Experiments/Base/{Path(model_path).name}_inference.tsv"

LANG_TO_CODE = {
    "English":"en",
    "Norwegian":"nn",
    "Oriya":"or",
    "Nepali":"ne",
    "eng_Latn":"English",
    "hin_Deva":"Hindi"
}

def get_vllm_model(model_path, max_model_len=4096, dtype=torch.float16, gpu_memory_utilization=0.9, tensor_parallel_size=2, enforce_eager=False):
    
    llm = LLM(
    model=model_path,
    max_model_len=max_model_len,
    dtype=dtype,
    gpu_memory_utilization=gpu_memory_utilization,
    trust_remote_code=True,
    tensor_parallel_size=tensor_parallel_size,
    enforce_eager=enforce_eager,

    #Sarvam
    # attention_backend="FLEX_ATTENTION"
    limit_mm_per_prompt={"image": 0},
)

    return llm

def make_prompts(example):
    src_lang = LANG_TO_CODE[example['src_lang']]
    tgt_lang = LANG_TO_CODE[example['tgt_lang']]
    src = example['src']
    tgt = example['tgt']
    sarvam_system_prompt = f"Translate the text below to {tgt_lang}."
    system_prompt = f"""Translate following sentence from {src_lang} to {tgt_lang}.\n{src}\nRESPONSE FORMAT:<ONLY_TRANSLATED_TEXT>"""

    messages = [{
        "role":"user",
        "content":system_prompt
    }]

    return {"messages":messages}

def make_prompts_tg(example):
    src_lang = example['src_lang'].strip()
    tgt_lang = example['tgt_lang'].strip()
    src = example['src'].strip()
    if LANG_TO_CODE[src_lang]=='' or LANG_TO_CODE[tgt_lang]=='':
        print(src)
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": LANG_TO_CODE[src_lang],
                "target_lang_code": LANG_TO_CODE[tgt_lang],
                "text": src,
            }
        ]
    }
]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"messages":prompt}

def main():

    llm = get_vllm_model(model_path=model_path, gpu_memory_utilization=0.25, dtype=torch.bfloat16,tensor_parallel_size=1)
        
    # for tsv_file in Path(infer_folder).glob("*.tsv"):
    # print(f"|> Inferring {tsv_file.name} file...")
    ds = load_dataset("csv", data_files=str(infer_folder), delimiter='\t',split='train')
    df = ds.to_pandas()
    
    prompts = ds.map(make_prompts)
    prompts = prompts.remove_columns(ds.column_names)

    # For Other
    outputs=llm.chat(list(prompts['messages']),
                     SamplingParams(max_tokens=512,temperature=0,stop=["</s>", "<|im_end|>", "<|endoftext|>"]),
                     chat_template_kwargs={"enable_thinking": False},
                    )

    # For TGemma
    # outputs=llm.generate(list(prompts['messages']),
    #                  SamplingParams(max_tokens=512,temperature=0,stop=["</s>", "<|im_end|>", "<|endoftext|>"]),)

    mt = [i.outputs[0].text.strip() for i in outputs]

    df['mt'] = mt

    # save_file_name = Path(f"{save_tsv_file}/{Path(model_path).name}")
    # save_file_name.mkdir(exist_ok=True, parents=True)

    df.to_csv(save_tsv_file, sep='\t')

if __name__ == "__main__":
    main()