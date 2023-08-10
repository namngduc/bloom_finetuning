#!/usr/bin/env python3

import os
import torch
from peft import PeftModel, PeftConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


# HUGGING_FACE_USER_NAME = "namngduc"

# peft_model_id = f"{HUGGING_FACE_USER_NAME}/GenerationText-Bloom"
base_dir = os.path.abspath(os.getcwd())
peft_model_id = os.path.join(base_dir, "bloom_ai")
config = PeftConfig.from_pretrained(peft_model_id)

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
elif device == "mps":
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, device_map=device, low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

# Load the LoRa model
model = PeftModel.from_pretrained(model, peft_model_id, 
                                offload_folder="offload") # prevent offload_dir ValueError

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)