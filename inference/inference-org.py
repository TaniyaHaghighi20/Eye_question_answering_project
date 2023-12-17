import torch
model_id = "meta-llama/Llama-2-13b-chat-hf"
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_file_path = "inference/golden_set.txt"  # Replace with the path to your input text file
output_file_path = "run4/inference/golden_set.llama13b.org.generated.y.txt"  # Replace with the desired path for the output file
count=0
try:
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
        # Read each line from the input file, add "s", and write to the output file
        for line in input_file:
            count+=1
            print(f"Q number: {count}")
            #line= "Choose one option with one word "+line
            messages = [{"role": "user", "content": ' '.join(line.split())}]
            
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
            

            outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95,pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_file.write(' '.join(decoded) + '\n-----\n')

except FileNotFoundError:
    print(f"Input file not found: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
