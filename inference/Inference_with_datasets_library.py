import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


peft_model_id = "/data1/sina/eyeGPT/eyeGPT2/LLM/run4/llama2-7b/checkpoint-20000"
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
test_set = load_from_disk('inference/medmcqa/test')
output_file_path = "run4/inference/medmcqa.llama2.generated.y.txt"  # Replace with the desired path for the output file
count=0
try:
    with open(output_file_path, 'w') as output_file:
        for row in range(len(test_set)):
            print(f"Q number: {row}")
            line= f"{test_set[row]['input'].replace('###','')}"
            line= line+ "\nChoose one option regarding the question. write your choice at the end of this sentence.\nsentence: my choice is"
            
            messages = [{
                "role": "system","content": test_set[row]['instruction']},
                {"role": "user", "content": line}]
            print(messages)
            
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
            inputs = inputs.to('cuda')

            outputs = merged_model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95,pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_file.write(' '.join(decoded) + '\n-----\n')

except Exception as e:
    print(f"An error occurred: {e}")