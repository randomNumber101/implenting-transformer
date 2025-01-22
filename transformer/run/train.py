import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, scheduler, vocab_size):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch['source_ids']
        tgt = batch['target_ids'][:, :-1]  # Exclude last token for input
        tgt_labels = batch['target_ids'][:, 1:]  # Shifted target for labels

        optimizer.zero_grad()
        output = model(src, tgt)
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
            src = batch['source_ids']
            tgt = batch['target_ids'][:, :-1]
            tgt_labels = batch['target_ids'][:, 1:]

            output = model(src, tgt)
            loss = criterion(output.reshape(-1, vocab_size), tgt_labels.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, vocab_size, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, scheduler, vocab_size)
        val_loss = eval_epoch(model, val_dataloader, criterion, vocab_size)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    from transformer.modelling.functional.Transformer import Transformer
    from transformer.modelling.preprocessor.data_utils import load_wmt17, create_dataloader, train_tokenizer_on_dataset
    from optimizers import initialize_optimizer_and_scheduler

    # Parameters
    VOCAB_SIZE = 50000
    D_MODEL = 64
    N_HEADS = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 128
    DROPOUT = 0.1
    MAX_LEN = 128
    DATASET_PORTION = 0.25
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 1e-4
    WARMUP_STEPS = 4000
    WEIGHT_DECAY = 0.01

    # Load dataset
    cleaned_data = load_wmt17()
    print("Data loaded.")

    # Train tokenizer
    tokenizer = train_tokenizer_on_dataset(cleaned_data, vocab_size=VOCAB_SIZE)
    print("Tokenizer trained.")

    # Create dataloaders
    train_data = cleaned_data[:int(DATASET_PORTION * len(cleaned_data))]
    val_data = cleaned_data[int(DATASET_PORTION * len(cleaned_data)):]
    train_dataloader = create_dataloader(train_data, tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    val_dataloader = create_dataloader(val_data, tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN)

    # Initialize model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LEN
    )

    # Initialize loss, optimizer, and scheduler
    criterion = CrossEntropyLoss(ignore_index=tokenizer.get_pad_token_id())
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, d_model=D_MODEL, warmup_steps=WARMUP_STEPS, lr=LR,
                                                              weight_decay=WEIGHT_DECAY)

    # Train model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        vocab_size=VOCAB_SIZE,
        num_epochs=NUM_EPOCHS
    )
