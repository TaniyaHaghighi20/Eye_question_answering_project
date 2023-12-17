import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

#peft_model_id = "/data1/sina/eyeGPT/eyeGPT2/LLM/run3/QA_finetune/checkpoint-300"
peft_model_id = "/data1/sina/eyeGPT/eyeGPT2/LLM/run3/mcqa_finetune/checkpoint-700"
#peft_model_id = "/data1/sina/eyeGPT/eyeGPT2/LLM/run3/output1/checkpoint-4000"
#peft_model_id = "/data1/sina/eyeGPT/eyeGPT2/LLM/run4/llama2-7b/checkpoint-7600"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.pad_token = tokenizer.eos_token
# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
print(type(model))
merged_model = model.merge_and_unload()
print(type(merged_model))
merged_model.to('cuda')
input_file_path = "run3/inference/medQA.x.txt"  # Replace with the path to your input text file
output_file_path = "run3/inference/MedQA.Mistral-mcqa_finetune.generated.y.txt"  # Replace with the desired path for the output file
count=0
try:
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Read each line from the input file, add "s", and write to the output file
        for line in input_file:
            count+=1
            print(f"Q number: {count}")
            #line= "Choose one option with one word "+line
            messages = [{"role": "user", "content": ' '.join(line.split())}]
            
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
            inputs = inputs.to('cuda')

            outputs = merged_model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95,pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_file.write(' '.join(decoded) + '\n-----\n')

except FileNotFoundError:
    print(f"Input file not found: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")