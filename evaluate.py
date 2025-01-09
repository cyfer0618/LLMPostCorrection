import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from tqdm import tqdm
import argparse
import random
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_name',
    type=str,
    required=True
)
parser.add_argument(
    "--folder",
    type=str,
    required=True
)
parser.add_argument(
    "--hypothesis", 
    type=str,
    required=True
)

args = parser.parse_args()

test_data = pd.read_csv(f'Dataset/Test/{args.folder}/combined_output.csv')
test_data = test_data.fillna('')

data_len = len(test_data)

system = [" Here is an incorrect Hindi ASR transcription. Please correct any spelling, grammatical, or meaning errors and provide the corrected version without explanation."] * data_len
user = [sentence.strip() for sentence in test_data[f'hyp_{args.hypothesis}'].tolist() if type(sentence) == str]

inputs = []

for idx in range(len(system)):
    sys = system[idx]
    usr = user[idx]
    if usr != '':
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>{usr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  
        inputs.append(prompt)


tokenizer = AutoTokenizer.from_pretrained("models/MBZUAI/Llama-3-Nanda-10B-Chat_epochs=2_bs=32_task=17-hindi-num/final_checkpoint", device_map = "auto")
model = AutoModelForCausalLM.from_pretrained("models/MBZUAI/Llama-3-Nanda-10B-Chat_epochs=2_bs=32_task=17-hindi-num/final_checkpoint", device_map = "auto")
tokenizer.pad_token = tokenizer.eos_token


with open(f'models/outputs/{args.output_name}_hyp_{args.hypothesis}.txt', 'a', encoding='utf-8') as file:
    for idx in tqdm(range(0, len(inputs), 8), desc='Processed batch'):
        input_ids = tokenizer(inputs[idx:idx+8], return_tensors="pt", padding=True, truncation=True)
        input_ids = {key: value.to('cuda') for key, value in input_ids.items()}
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=256)])
        outputs = model.generate(**input_ids, max_new_tokens=128)
        output_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for output in output_list:
            file.write(output + '\n\n')
