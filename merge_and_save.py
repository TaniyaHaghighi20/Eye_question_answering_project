
path_to_checkpoint = "/data1/sina/eyeGPT/eyeGPT2/LLM/gpt2-model/checkpoint-12400"


from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

model_name = "gpt2-xl"
path_to_whole_model = "/data1/sina/eyeGPT/eyeGPT2/LLM/gpt2-model/pretrained"
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_model = PeftModel.from_pretrained(model, path_to_checkpoint)

# Merge the PEFT model with the base model
merged_model = peft_model.merge_and_unload()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)

# Save the tokenizer
tokenizer.save_pretrained(path_to_whole_model)


# Save the merged model
merged_model.save_pretrained(path_to_whole_model)
