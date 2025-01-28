import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from transformer.modelling.preprocessor.data_utils import train_tokenizer_on_dataset, load_wmt17, create_dataloader, \
    train_test_split
from transformer.run.optimizers import initialize_optimizer_and_scheduler
from utils import TrainingConfig, load_or_create_model, get_model_path


def train_epoch(model, dataloader, criterion, optimizer, scheduler, vocab_size):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch['source_ids'].to(model.device)
        tgt = batch['target_ids'][:, :-1].to(model.device)  # Exclude last token for input
        tgt_labels = batch['target_ids'][:, 1:].to(model.device)  # Shifted target for labels

        # Get masks from the batch
        src_mask = batch['source_mask'].to(model.device)
        tgt_mask = batch['target_mask'][:, :-1].to(model.device)  # Exclude last token for mask

        optimizer.zero_grad()
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = criterion(output.reshape(-1, vocab_size), tgt_labels.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch['source_ids'].to(model.device)
            tgt = batch['target_ids'][:, :-1].to(model.device)
            tgt_labels = batch['target_ids'][:, 1:].to(model.device)

            # Get masks from the batch
            src_mask = batch['source_mask'].to(model.device)
            tgt_mask = batch['target_mask'][:, :-1].to(model.device)

            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, vocab_size), tgt_labels.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(config: TrainingConfig):
    # Load dataset and tokenizer
    cleaned_data = load_wmt17()
    tokenizer = train_tokenizer_on_dataset(cleaned_data, vocab_size=config.vocab_size)

    # Create dataloaders
    data = cleaned_data[:int(config.dataset_portion * len(cleaned_data))]
    train_data, val_data = train_test_split(data, test_portion=config.validation_portion)
    train_dataloader = create_dataloader(train_data, tokenizer,
                                         batch_size=config.batch_size,
                                         max_len=config.max_len)
    val_dataloader = create_dataloader(val_data, tokenizer,
                                       batch_size=config.batch_size,
                                       max_len=config.max_len)

    # Initialize model
    model = load_or_create_model(config)
    model.to(model.device)

    # Initialize training components
    criterion = CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    warmup_steps = int(config.warmup_portion * len(train_dataloader) * config.num_epochs)

    optimizer, scheduler = initialize_optimizer_and_scheduler(
        model,
        d_model=config.d_model,
        warmup_steps=warmup_steps,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Training loop
    best_val_loss = float('inf')
    model_path = get_model_path(config)

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_epoch(model, train_dataloader, criterion,
                                 optimizer, scheduler, config.vocab_size)
        val_loss = eval_epoch(model, val_dataloader, criterion, config.vocab_size)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    config = TrainingConfig() # Use default params
    train_model(config)