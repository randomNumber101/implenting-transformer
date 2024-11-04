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


def load_wmt17():
    # Load WMT17 German-English dataset from Hugging Face
    dataset = load_dataset("wmt17", "de-en", split="train[:1%]")  # Using 1% subset for quick testing
    data = [(item['translation']['de'], item['translation']['en']) for item in dataset]

    # Clean the data using the clean_data function
    cleaned_data = clean_data(data, min_length=5, max_length=64, max_ratio=1.5)
    return cleaned_data


def train_tokenizer_on_dataset(data, vocab_size=50000):
    # Extract sentences for tokenizer training (concatenate source and target)
    corpus = [source for source, _ in data] + [target for _, target in data]

    # Initialize and train the CustomBPETokenizer
    tokenizer = BPETokenizer(vocab_size=50000)
    tokenizer.train(corpus)
    return tokenizer


def collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad sequences in the batch to the same length.
    """


    # Separate source and target sequences
    source_ids = [item['source_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Find maximum length in the batch for both source and target
    max_source_len = max(len(s) for s in source_ids)
    max_target_len = max(len(t) for t in target_ids)

    # Pad sequences to max length
    padded_source_ids = [
        torch.cat([s, torch.full((max_source_len - len(s),), pad_token_id, dtype=torch.long)])
        for s in source_ids
    ]
    padded_target_ids = [
        torch.cat([t, torch.full((max_target_len - len(t),), pad_token_id, dtype=torch.long)])
        for t in target_ids
    ]

    # Stack into a single tensor
    source_ids_batch = torch.stack(padded_source_ids)
    target_ids_batch = torch.stack(padded_target_ids)

    return {'source_ids': source_ids_batch, 'target_ids': target_ids_batch}


def create_dataloader(cleaned_data, tokenizer: BPETokenizer, batch_size=16):
    from .dataloader import CustomTranslationDataset
    # Initialize the CustomTranslationDataset with cleaned data and trained tokenizer
    translation_dataset = CustomTranslationDataset(cleaned_data, tokenizer)

    # Create a DataLoader for batch processing
    dataloader = DataLoader(translation_dataset,
                            batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer.get_pad_token_id()))
    return dataloader


