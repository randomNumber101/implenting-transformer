import random
import re
from datasets import load_dataset
from torch.utils.data import DataLoader
from .tokenizer import BPETokenizer
import torch


def clean_data(data, min_length=5, max_length=64, max_ratio=1.5):
    # Whitelist of allowed characters
    whitelist = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    allowed_chars = set(whitelist)

    def preprocess(text):
        text = bytes(text, 'utf-8').decode('utf-8', 'ignore')
        text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
        text = ''.join([ch for ch in text if ch in allowed_chars])
        return text

    cleaned_data = []
    for source, target in data:
        # Remove characters not in the whitelist
        source = preprocess(source)
        target = preprocess(target)

        # Filter by length
        if len(source.split()) < min_length or len(source.split()) > max_length:
            continue
        if len(target.split()) < min_length or len(target.split()) > max_length:
            continue

        # Check source/target length ratio
        ratio = len(source.split()) / len(target.split())
        if ratio > max_ratio or ratio < (1 / max_ratio):
            continue

        # Append cleaned and filtered pair
        cleaned_data.append((source, target))

    return cleaned_data


def load_wmt17(split="train[:1%]") -> list:
    # Load WMT17 German-English dataset from Hugging Face
    dataset = load_dataset("wmt17", "de-en", split=split)  # Using 1% subset for quick testing
    data = [(item['translation']['de'], item['translation']['en']) for item in dataset]

    # Clean the data using the clean_data function
    cleaned_data = clean_data(data, min_length=5, max_length=64, max_ratio=1.5)
    return cleaned_data


def tokenize_data(data, tokenizer, max_len=50):
    """
    Tokenize data and prepare it for batching.

    Parameters:
        data: List[Tuple[str, str]] - List of (source, target) sentence pairs.
        tokenizer: BPETokenizer - Trained tokenizer.
        max_len: int - Maximum length for truncation.

    Returns:
        List[Dict]: Tokenized data with 'source_ids' and 'target_ids'.
    """
    tokenized_data = []
    for source, target in data:
        source_ids = tokenizer.encode(source)[:max_len]
        target_ids = tokenizer.encode(target)[:max_len]
        tokenized_data.append({'source_ids': source_ids, 'target_ids': target_ids})
    return tokenized_data


def collate_fn(batch, pad_token_id, max_len=50):
    # Separate source and target sequences
    source_ids = [torch.tensor(item['source_ids'][:max_len], dtype=torch.long) for item in batch]  # Convert to tensor
    target_ids = [torch.tensor(item['target_ids'][:max_len], dtype=torch.long) for item in batch]  # Convert to tensor

    # Find maximum length in the batch for both source and target
    max_source_len = min(max(len(s) for s in source_ids), max_len)
    max_target_len = min(max(len(t) for t in target_ids), max_len)

    # Pad sequences to max length
    padded_source_ids = [
        torch.cat([s, torch.full((max_source_len - len(s),), pad_token_id, dtype=torch.long)])
        for s in source_ids
    ]
    padded_target_ids = [
        torch.cat([t, torch.full((max_target_len - len(t),), pad_token_id, dtype=torch.long)])
        for t in target_ids
    ]

    # Create attention masks for source and target
    source_mask = [
        torch.cat([torch.ones(len(s), dtype=torch.long), torch.zeros(max_source_len - len(s), dtype=torch.long)])
        for s in source_ids
    ]
    target_mask = [
        torch.cat([torch.ones(len(t), dtype=torch.long), torch.zeros(max_target_len - len(t), dtype=torch.long)])
        for t in target_ids
    ]

    # Stack into a single tensor
    source_ids_batch = torch.stack(padded_source_ids)
    target_ids_batch = torch.stack(padded_target_ids)
    source_mask_batch = torch.stack(source_mask)
    target_mask_batch = torch.stack(target_mask)

    return {
        'source_ids': source_ids_batch,
        'target_ids': target_ids_batch,
        'source_mask': source_mask_batch,
        'target_mask': target_mask_batch
    }


def prepare_test_data(test_data, tokenizer, max_len=50):
    """
    Tokenize and prepare the test data.

    Parameters:
        test_data: List[Tuple[str, str]] - List of (source, target) sentence pairs.
        tokenizer: BPETokenizer - Trained tokenizer.
        max_len: int - Maximum length for truncation.

    Returns:
        List[Dict]: Tokenized test data with 'source_ids', 'target_ids', 'source_mask', and 'target_mask'.
    """
    tokenized_data = []
    for source, target in test_data:
        source_ids = tokenizer.encode(source)[:max_len]
        target_ids = tokenizer.encode(target)[:max_len]

        # Create masks
        source_mask = [1] * len(source_ids) + [0] * (max_len - len(source_ids))
        target_mask = [1] * len(target_ids) + [0] * (max_len - len(target_ids))

        # Pad sequences to max_len
        source_ids = source_ids + [tokenizer.get_pad_token_id()] * (max_len - len(source_ids))
        target_ids = target_ids + [tokenizer.get_pad_token_id()] * (max_len - len(target_ids))

        tokenized_data.append({
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long),
        })

    return tokenized_data

def train_test_split(data, test_portion=0.3):
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_portion))
    return data[:split_idx], data[split_idx:]

def train_tokenizer_on_dataset(data, vocab_size=50000) -> BPETokenizer:
    # Extract sentences for tokenizer training (concatenate source and target)
    corpus = [source for source, _ in data] + [target for _, target in data]

    # Initialize and train the CustomBPETokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus)
    return tokenizer


def create_dataloader(cleaned_data, tokenizer: BPETokenizer, batch_size=16, max_len=256):
    from .dataloader import CustomTranslationDataset
    # Initialize the CustomTranslationDataset with cleaned data and trained tokenizer
    translation_dataset = CustomTranslationDataset(cleaned_data, tokenizer)

    # Create a DataLoader for batch processing
    dataloader = DataLoader(translation_dataset,
                            batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_token_id(), max_len=max_len))
    return dataloader