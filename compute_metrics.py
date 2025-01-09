import pandas as pd
from jiwer import wer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_name',
    type=str,
    required=True,
    help="Specify output file name"
)
parser.add_argument(
    "--folder_name",
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

test_data = pd.read_csv(f"Dataset/Test/{args.folder_name}/combined_output.csv")
test_data = test_data.fillna('')
data_len = len(test_data)

with open(f'models/outputs/{args.output_name}_hyp_{args.hypothesis}.txt', 'r', encoding='utf-8') as file:
    predictions = file.readlines()

predictions = ''.join(predictions).split('\n\n')[:-1]

predictions = [sentence.split("assistant")[1].strip("'").strip() for sentence in predictions]

references = [sentence.strip() for idx, sentence in enumerate(test_data['ground_truth'].tolist()) if type(sentence) == str and test_data[f'hyp_{args.hypothesis}'][idx] != '']

with open('log.txt', 'w', encoding='utf-8') as file:
    for idx, reference in enumerate(references):
        print(references[idx],' | ' , predictions[idx], file=file)

word_error_rate = wer(references, predictions)

print(word_error_rate)





