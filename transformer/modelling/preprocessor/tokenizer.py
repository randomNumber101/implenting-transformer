import copy
from typing import List, Tuple, Set, Union
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from transformers import GPT2Tokenizer
import os
import json
from tokenizers.models import BPE

class Pair:
    def __init__(self, l: str, r: str, start: int, end: int):
        self.l = l
        self.r = r
        self.start = start
        self.end = end


class PairCounter:
    def __init__(self, base_tokens: List[str], EOW = "<EOW>"):
        self.base_tokens = base_tokens
        self.pairs = {}
        self.tok_freqs = {tok: 0 for tok in base_tokens}
        self.EOW = EOW
        for tok in base_tokens:
            self.tok_freqs[tok] += 1

    def add_pair(self, pair: Pair):
        if (pair.l, pair.r) not in self.pairs:
            self.pairs[(pair.l, pair.r)] = [pair]
        else:
            self.pairs[(pair.l, pair.r)].append(pair)

    def add(self, l: str, r: str, start: int, end: int):
        pair = Pair(l, r, start, end)
        self.add_pair(pair)

    def remove(self, pair: Pair):
        if (pair.l, pair.r) in self.pairs:
            self.pairs[(pair.l, pair.r)].remove(pair)
            if not self.pairs[(pair.l, pair.r)]:
                del self.pairs[(pair.l, pair.r)]

    def merge(self, x: str, y: str):
        if (x, y) not in self.pairs:
            raise ValueError(f"Pair {x, y} not found.")

        merged_token = x + y
        self.tok_freqs[merged_token] = len(self.pairs[(x, y)])

        newpairs = []
        for pair in self.pairs[(x, y)]:
            if pair.start > 0:
                l = self.base_tokens[pair.start - 1]
                if l != self.EOW:
                    newpairs.append(Pair(l, merged_token, pair.start - 1, pair.end))
            if pair.end < len(self.base_tokens) - 1:
                r = self.base_tokens[pair.end + 1]
                if r != self.EOW:
                    newpairs.append(Pair(merged_token, r, pair.start, pair.end + 1))

        for p in newpairs:
            self.add_pair(p)

        del self.pairs[(x, y)]

    def get_most_freq(self) -> Tuple[str, str]:
        return max(self.pairs.keys(), key=lambda k: len(self.pairs[k]), default=None)

    def get_freqs(self):
        return self.tok_freqs


class PairIndex:
    def __init__(self, text: str, EOW="<EOW>"):
        self.base_tokens = [tok for word in text.strip().split() for tok in list(word) + [EOW]]
        self.tokens = set(self.base_tokens + [EOW])
        self.merges = []
        self.EOW = EOW
        self.pairs = {}
        self.counter = PairCounter(self.base_tokens)
        self._build()

    def _build(self):
        for i, tok in enumerate(self.base_tokens):
            if i == len(self.base_tokens) - 1:
                break
            r = self.base_tokens[i + 1]
            if tok == self.EOW or r == self.EOW:
                continue
            self.counter.add(tok, r, i, i + 1)

    def get_freqs(self):
        return self.counter.tok_freqs

    def do_iteration(self):
        pair = self.counter.get_most_freq()
        if pair:
            self.counter.merge(*pair)
            self.merges.append(pair)
            self.tokens.add(pair[0] + pair[1])


class CustomCharBPETokenizer:
    def __init__(self):
        self.merges = []
        self.tokens = set()
        self.freqs = {}

    def train(self, train_data: Union[str, List[str]], max_tokens=500, max_iterations=1000):
        if isinstance(train_data, list):
            train_data = " ".join(train_data)

        index = PairIndex(train_data)
        num_tokens = len(index.tokens)
        for _ in range(max_iterations):
            if num_tokens >= max_tokens:
                break
            index.do_iteration()
            num_tokens = len(index.tokens)

        # Save the merges, tokens, and frequency counts after training
        self.merges = index.merges
        self.tokens = index.tokens
        self.freqs = index.get_freqs()

        print("Merges:", self.merges)
        print("Tokens:", self.tokens)
        print("Token Frequencies:", self.freqs)

    def tokenize(self, text: str):
        words = text.strip().split()
        tokens = [list(word) + ["<EOW>"] for word in words]
        for x, y in self.merges:
            new_token = x + y
            for word in tokens:
                i = 0
                while i < len(word) - 1:
                    if word[i] == x and word[i + 1] == y:
                        word[i] = new_token
                        del word[i + 1]
                    i += 1
        return tokens







class BPETokenizer:
    def __init__(self, vocab_size=295, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        self.gpt2_tokenizer: GPT2Tokenizer = None
        # Define the special tokens to use
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

    def train(self, corpus, output_dir="tokenizer_files"):

        tmp_path = os.path.join(output_dir, "save.json")
        vocab_path = os.path.join(output_dir, "vocab.json")
        merges_path = os.path.join(output_dir, "merges.txt")

        if not os.path.isfile(vocab_path) or not os.path.isfile(merges_path):
            # Save the corpus temporarily for training with UTF-8 encoding
            with open("corpus.txt", "w", encoding="utf-8") as f:
                for line in corpus:
                    f.write(line + "\n")

            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            tokenizer.decoder = decoders.ByteLevel()

            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens
            )

            # Train the tokenizer
            tokenizer.train(["corpus.txt"], trainer=trainer)

            # Save vocabulary and merges in GPT2-compatible format
            os.makedirs(output_dir, exist_ok=True)


            # Write the vocab.json file
            vocab = tokenizer.get_vocab()
            sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
            with open(vocab_path, 'w', encoding="utf-8") as vocab_file:
                json.dump(sorted_vocab, vocab_file)

            # Write the merges.txt file
            tokenizer.save(tmp_path)
            with open(tmp_path, 'r', encoding="utf-8") as file:
                merges = json.load(file)["model"]["merges"]

            with open(merges_path, 'w', encoding="utf-8") as merges_file:
                merges_file.write("#version: 0.2\n")
                for merge in merges:
                    merges_file.write(" ".join(merge) + "\n")

        # Load the GPT2Tokenizer from the saved files, specifying the special tokens
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            output_dir,
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>"
        )

    def encode(self, text):
        # Encode text using the GPT2-compatible tokenizer
        return self.gpt2_tokenizer.encode(text)

    def decode(self, token_ids, **kwargs):
        # Decode token IDs using the GPT2-compatible tokenizer
        return self.gpt2_tokenizer.decode(token_ids, **kwargs)

    def get_pad_token(self):
        return self.gpt2_tokenizer.pad_token

    def get_pad_token_id(self):
        return self.gpt2_tokenizer.pad_token_id



