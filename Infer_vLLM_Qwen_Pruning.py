import os
os.environ['HF_HOME']="/scratch/yuvraj.bodana"
os.environ['HF_DATASETS_CACHE']="/scratch/yuvraj.bodana"

from vllm import LLM, SamplingParams
import torch
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import numpy as np
from huggingface_hub import login

login(token="")

#### Variables ####
dataset_path = './Opus_Test_Data_en_hi'
model_path = "/scratch/yuvraj.bodana/Llama_1B_inst_in22_flores_ft/merged-checkpoint-280/"
# "Qwen/Qwen3-4B", "Yuvrajsinh0409/Qwen_4B_LayerWise_Pruned_28_Layers", "meta-llama/Llama-3.2-1B-Instruct"
#### Load Dataset ####
data = load_from_disk(dataset_path)
# messages = data['prompt'][:10]

def format_chat(example):
    return {
        "messages":[{"role":"user","content":example['prompt']}]
    }
messages = data.map(format_chat)['messages'][:]
#### Load Model ####
llm = LLM(
    model=model_path,
    max_model_len=512,
    # dtype="auto",
    # gpu_memory_utilization=0.3,
    # trust_remote_code=True,
    tensor_parallel_size=2,
    enforce_eager=True,
    
)

#### Generate ####
outputs = llm.chat(
    messages,
    SamplingParams(max_tokens=256, temperature=0,stop=["</s>", "<|im_end|>", "<|endoftext|>"]),
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}  
    # default_chat_template_kwargs = {"enable_thinking": False}
    chat_template_kwargs={"enable_thinking": False}  

)

data = {
 "Source_Dialogue" : data['source'],
    "Reference": data['target'],
    "MT": [i.outputs[0].text.strip() for i in outputs]
}

infer_df = pd.DataFrame(data)
infer_df.to_csv(f"Inference_vllm_Llama_3.2_1B_FT.tsv", sep='\t')