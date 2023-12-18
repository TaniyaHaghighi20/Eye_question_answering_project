import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
model_id = "meta-llama/Llama-2-13b-chat-hf"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
test_set = load_from_disk('inference/medmcqa/test')
output_file_path = "run4/inference/golden_set.llama13b.org.generated.y.txt"  # Replace with the desired path for the output file
try:
    with open(output_file_path, 'a') as output_file:
        # Read each line from the input file, add "s", and write to the output file
        for row in range(len(test_set)):
            print(f"Q number: {row}")
            line= f"{test_set[row]['input'].replace('###','')}"
            #medmcqa
            line= line+ "\nChoose one option regarding the question. write your choice at the end of this sentence.\nsentence: my choice is"
            #pubmedqa
            #line= line+ "\nwrite yes or no at the end of the sentence.\nsentence: the answer is"
            
            messages = [{
                "role": "system","content": test_set[row]['instruction']},
                {"role": "user", "content": line}]
            print(messages)
            
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
            

            outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95,pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_file.write(' '.join(decoded) + '\n-----\n')

except Exception as e:
    print(f"An error occurred: {e}")
