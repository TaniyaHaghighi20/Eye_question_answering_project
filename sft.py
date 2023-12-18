# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer, is_xpu_available


tqdm.pandas()
model_name ='path/to/model'
dataset_name='finetune_data/path/to/data'
peft_lora_r=16
peft_lora_alpha=64
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
seq_length=1024
dataset_text_field='text'
output_dir= 'run4/llama2-7b-chat/finetune'
train_split= 'train'
val_split= 'test'

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    ) 

# Step 1: Load the model
# quantization_config = BitsAndBytesConfig(load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16)
# Copy the model to each device
# device_map = (
#     {"": f"xpu:{Accelerator().local_process_index}"}
#     if is_xpu_available()
#     else {"": Accelerator().local_process_index}
# )
torch_dtype = torch.bfloat16


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    load_in_8bit=True,
    use_flash_attention_2=True,
    # device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
)
print_trainable_parameters(model)

# Step 2: Load the dataset
dataset = load_from_disk(dataset_name)
train_dataset = dataset[train_split]
validation_dataset = dataset[val_split]
#train_dataset = train_dataset.select(range(500))
#validation_dataset = validation_dataset.select(range(50))

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=64,
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=40,
    report_to='tensorboard',
    save_steps=200,
    gradient_checkpointing=False,
    evaluation_strategy="steps",  # Evaluate the model every logging step
    logging_dir= output_dir+"/logs",  # Directory for storing logs
    save_strategy="steps",  # Save the model checkpoint every logging step
    eval_steps=200,  # Evaluate and save checkpoints every 10 steps
    do_eval=True,
    warmup_steps=600,
    weight_decay=1e-3,
    lr_scheduler_type='cosine',
    load_best_model_at_end=True,
    max_grad_norm=1.0,
    metric_for_best_model='eval_loss',
    save_total_limit=10
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)

# Step 4: Define the LoraConfig

peft_config = LoraConfig(
    r=peft_lora_r,
    lora_alpha=peft_lora_alpha,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)


# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    # max_seq_length=seq_length,
    packing=True,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    dataset_text_field=dataset_text_field,
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(output_dir)
