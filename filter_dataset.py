import pandas as pd
from datasets import DatasetDict, Dataset

split = ['train', 'val']

system = []
user = []
assistant = []
eval_system = []
eval_user = []
eval_assistant = []

for s in split:
    data = pd.read_json(f'Dataset/15/{s}.jsonl', lines=True)

    for row in data['text']:
        if 'INPUT:' not in row:
            row = 'INPUT:' + row
        if s == 'train':
            system.append(row.split('\n')[0].split('INPUT:')[1].strip())
            user.append(row.split('\n')[1].strip())
            assistant.append(row.split('\n')[2].split('OUTPUT:')[1].strip())
        else:
            eval_system.append(row.split('\n')[0].split('INPUT:')[1].strip())
            eval_user.append(row.split('\n')[1].strip())
            eval_assistant.append(row.split('\n')[2].split('OUTPUT:')[1].strip())

    # data['system'] = system
    # data['user'] = user
    # data['assistant'] = assistant

    # data = data.drop(['text'], axis=1)

    # data.to_csv(f'Dataset/7/{s}.csv', index=False)
print(len(system), len(eval_system))

hf_data = {
    "train": {'system': system, 'user': user, 'assistant': assistant},
    "validation": {'system': eval_system, 'user': eval_user, 'assistant': eval_assistant}
}

custom_dataset = DatasetDict()

for split in hf_data:
    custom_dataset[split] = Dataset.from_dict(hf_data[split])

custom_dataset.save_to_disk('Dataset/15/hf_format')