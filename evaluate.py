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
import time
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_name',
    type=str,
    required=True,
    help="Specify output file name"
)
parser.add_argument(
    "--folder",
    type=str,
    required=True,
    choices=["WS", "CW", "EN", "EW", "HN", "AllError"],
    help="Specify the folder. Must be one of: WS, CW, EN, EW, HN, AllError"
)
parser.add_argument(
    "--hypothesis", 
    type=str,
    required=True,
    choices=['ic', 'w2v'],
    help="Specify the hypothesis. Must be one of: ic, w2v"
)

args = parser.parse_args()

test_data = pd.read_csv(f'Dataset/Test/{args.folder}/combined_output.csv')
test_data = test_data.fillna('')

data_len = len(test_data)

if args.folder == 'WS':
    prompt = "Here is a Hindi ASR transcription with incorrect word segmentation (e.g., रहीहै instead of रही है or दु कान instead of दुकान). Please provide the corrected version without any explanation."
elif args.folder == 'CW':
    prompt = "Here is a Hindi ASR transcription with incorrect split of compound word (e.g., रथयात्रा as रथ यात्रा). Please provide the corrected version without any explanation."
elif args.folder == 'EW':
    prompt = "Here is a Hindi ASR transcription with incorrect transliterations of English words (e.g., 'concept' as कांसेकत instead of कॉन्सेप्ट). Please provide the corrected version without any explanation."
elif args.folder == 'EN':
    prompt = "Here is a Hindi ASR transcription with incorrect transliterations of English number (e.g., 'one' as वान instead of वन). Please provide the corrected version without any explanation."
elif args.folder == 'HN':
    prompt = "Here is a Hindi ASR transcription with incorrect transliterations of English number (e.g., 'one' as वान instead of वन). Please provide the corrected version without any explanation."
else:
    prompt = "Here is an incorrect Hindi ASR transcription. Please correct any spelling, grammatical, or meaning errors and provide the corrected version without explanation."
    
system = [prompt] * data_len
user = [sentence.strip() for sentence in test_data[f'hyp_{args.hypothesis}'].tolist() if type(sentence) == str]

inputs = []

for idx in range(len(system)):
    sys = system[idx]
    usr = user[idx]
    if usr != '':
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>{usr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"  
        inputs.append(prompt)


tokenizer = AutoTokenizer.from_pretrained("models/MBZUAI/Llama-3-Nanda-10B-Chat_epochs=2_bs=32_task=14-compound-new/final_checkpoint", device_map = "auto") # change model path
model = AutoModelForCausalLM.from_pretrained("models/MBZUAI/Llama-3-Nanda-10B-Chat_epochs=2_bs=32_task=14-compound-new/final_checkpoint", device_map = "auto") # change model path
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