from torch.utils.data import Dataset
import torch
from .data_utils import clean_data


class CustomTranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, min_length=5, max_length=64, max_ratio=1.5):
        # Clean and filter the dataset using the clean_data function from data_utils.py
        self.data = clean_data(dataset, min_length=min_length, max_length=max_length, max_ratio=max_ratio)

        # Initialize the tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the source and target sentence pair
        source, target = self.data[idx]

        # Encode source and target using the tokenizer
        source_ids = self.tokenizer.encode(source)
        target_ids = self.tokenizer.encode(target)

        # Convert to tensors and return
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    """
    # Extract the pad token ID from one of the dataset items
    pad_token_id = batch[0]['source_ids'].new_full((1,), batch[0]['source_ids'][0].pad_token_id).item()

    # Separate source and target sequences
    source_ids = [item['source_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad sequences
    source_ids = torch.nn.utils.rnn.pad_sequence(source_ids, batch_first=True, padding_value=pad_token_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)

    return {'source_ids': source_ids, 'target_ids': target_ids}
