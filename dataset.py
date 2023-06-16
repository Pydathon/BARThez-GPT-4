from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.body[index]
        summary = self.data.resume[index]

        # Tokenize the text and summary
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        targets = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),  # Squeeze to remove the batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
        }
