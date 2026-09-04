from collections import defaultdict
from tqdm import tqdm


class PepeBPE:
    def __init__(self):
        self.vocab: dict[bytes, int] = dict()
        self.frequencies: dict[tuple[bytes], int] = defaultdict(int)
        self.last_free_index: int = 0

    def _add_bytes_token_to_vocab(self, token: bytes):
        if token not in self.vocab:
            self.vocab[token] = self.last_free_index
            self.last_free_index += 1

    def _pre_tokenize_piece(self, text: str):
        for word in text.split(" "):
            self.frequencies[tuple(bytes([w, ]) for w in word.encode())] += 1

    def _merge_most_frequent_pair_with_replacement(self):
        if not self.frequencies:
            return

        pairs_frequencies: dict[bytes, int] = defaultdict(int)
        final_merge = None
        for word in self.frequencies.keys():
            for i in range(len(word) - 1):
                pair = word[i] + word[i + 1]
                pairs_frequencies[pair] += self.frequencies[word]
                if final_merge is None or pairs_frequencies[pair] > pairs_frequencies[final_merge]:
                    final_merge = pair
                    continue
                if pairs_frequencies[pair] == pairs_frequencies[final_merge] and pair > final_merge:
                    final_merge = pair

        self._add_bytes_token_to_vocab(final_merge)

        new_frequencies: dict[tuple[bytes], int] = defaultdict(int)
        for word in self.frequencies.keys():
            if len(word) < 2:
                continue
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] + word[i + 1] == final_merge:
                    new_word.append(final_merge)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_frequencies[tuple(new_word)] += self.frequencies[word]

        self.frequencies = new_frequencies


    def pre_tokenize_corpus(self, corpus: list[str]):
        for text in corpus:
            self._pre_tokenize_piece(text)

    def train_bpe_from_scratch(self, corpus: list[str], vocab_size: int):
        # init the vocab
        self.vocab.clear()
        self.frequencies.clear()
        self.last_free_index = 0
        for i in range(256):
            self._add_bytes_token_to_vocab(bytes([i,]))
        self._add_bytes_token_to_vocab("<|endoftext|>".encode())

        self.pre_tokenize_corpus(corpus)

        with tqdm(total=vocab_size - len(self.vocab), desc="Merging pairs") as pbar:
            while len(self.vocab) < vocab_size:
                self._merge_most_frequent_pair_with_replacement()
                pbar.update(1)

if __name__ == "__main__":
    tokenizer = PepeBPE()
    EXAMPLE_CORPUS = [
        "low low low low low",
        "lower lower widest widest widest",
        "newest newest newest newest newest newest"
    ]

    tokenizer.train_bpe_from_scratch(EXAMPLE_CORPUS, vocab_size=263)
    print(tokenizer.frequencies)
    print(tokenizer.vocab)

