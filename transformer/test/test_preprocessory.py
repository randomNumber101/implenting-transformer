from ..modelling.preprocessor.data_utils import load_wmt17, train_tokenizer_on_dataset, create_dataloader


def test_translation_dataloader():
    # Step 1: Load and clean the WMT17 dataset
    cleaned_data = load_wmt17()

    # Step 2: Train the tokenizer on the cleaned dataset
    tokenizer = train_tokenizer_on_dataset(cleaned_data, vocab_size=50000)

    # Step 3: Create a DataLoader for the prepared data
    dataloader = create_dataloader(cleaned_data, tokenizer, batch_size=16)

    # Step 4: Test the DataLoader by fetching a batch
    for batch in dataloader:
        print("Source IDs:", batch['source_ids'])
        print("Target IDs:", batch['target_ids'])
        break  # Print one batch for testing
