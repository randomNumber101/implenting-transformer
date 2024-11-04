from transformer.modelling.preprocessor.tokenizer import CustomCharBPETokenizer as CustomTokenizer

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

TEST_DATA = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]

INPUT_SENTENCE = "Machine learning is a subset of artificial intelligence."


def get_custom():
    tokenizer = CustomTokenizer()
    tokenizer.train(TEST_DATA, max_tokens=64)
    return tokenizer


def get_huggingface():
    # Save corpus to a temporary file
    with open("corpus.txt", "w") as f:
        for line in TEST_DATA:
            f.write(line + "\n")

    # Initialize the Byte-Pair Encoding (BPE) tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Train the tokenizer with the specified vocabulary size

    trainer = trainers.BpeTrainer(
        vocab_size=295,
        min_frequency=2
    )

    tokenizer.train([".\corpus.txt"], trainer=trainer)
    return tokenizer


def test_custom():
    t = get_custom()
    assert t is not None


def test_huggingface():
    t = get_huggingface()
    assert t is not None


def test_both():
    custom = get_custom()
    hf = get_huggingface()

    print(f"Input:  {INPUT_SENTENCE}")
    print(f"Custom: {custom.tokenize(INPUT_SENTENCE)}")
    print(f"HF Tok: {hf.encode(INPUT_SENTENCE).tokens}")
