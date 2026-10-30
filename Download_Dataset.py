import os
# os.environ['HF_HOME'] = "/scratch/yuvraj.bodana/"
# os.environ['HF_DATASETS_CACHE'] = "/scratch/yuvraj.bodana/"

from datasets import load_dataset, load_from_disk
from huggingface_hub import login

login(token="")

dataset_name = "openlanguagedata/flores_plus"

ds = load_dataset(dataset_name, "npi_Deva")

print(f"Dataset Downloaded Successfully..!!\n{ds}")