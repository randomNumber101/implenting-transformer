# generate.py
import os

import torch
import evaluate
from tqdm import tqdm
from utils import TrainingConfig, load_or_create_model, get_model_path
from transformer.modelling.preprocessor.data_utils import load_wmt17, prepare_test_data, train_tokenizer_on_dataset


def generate(config: TrainingConfig):
    # Load dataset and tokenizer
    cleaned_data = load_wmt17()
    tokenizer = train_tokenizer_on_dataset(cleaned_data, vocab_size=config.vocab_size)

    # Initialize model
    model = load_or_create_model(config)
    model_path = get_model_path(config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise ValueError(f"No trained model found at {model_path}")

    # Prepare test data
    test_data = load_wmt17(split="test[:1%]")
    prepared_test_data = prepare_test_data(test_data, tokenizer, max_len=config.max_len)

    # Evaluation logic
    bleu = evaluate.load("bleu")
    references = []
    predictions = []

    for idx, example in tqdm(enumerate(prepared_test_data), desc="Evaluating"):
        src = example['source_ids'].unsqueeze(0).to(model.device)
        src_mask = example['source_mask'].unsqueeze(0).to(model.device)

        generated_ids = model.generate(src, tokenizer, max_len=config.max_len, src_mask=src_mask)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        reference_text = tokenizer.decode(example['target_ids'], skip_special_tokens=True)

        if idx % 50 == 0:
            print(f"Example {idx + 1}: {generated_text}")

        references.append([reference_text])
        predictions.append(generated_text)

    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_score['bleu']}")


if __name__ == "__main__":
    config = TrainingConfig(
        # Match the training config parameters
        # num_epochs=10,
        # lr=2e-3,
        # ...
    )
    generate(config)