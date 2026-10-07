import os
import json
import regex as re
from concurrent.futures import ProcessPoolExecutor

from typing import BinaryIO, Iterator, Iterable
from collections import defaultdict
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4 * 1024  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = re.search(split_special_token, mini_chunk)
            if found_at is not None:
                chunk_boundaries[bi] = initial_position + found_at.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(text: bytes, pat: bytes, special_tokens_pattern: bytes) -> Iterator[tuple[tuple[bytes, ...], bool]]:
    for t in re.split(special_tokens_pattern, text):
        if re.fullmatch(special_tokens_pattern, t) is not None:
            yield (t, ), True
        else:
            for word in re.finditer(pat, t):
                yield tuple(bytes([w,]) for w in word.group()), False

def _find_index(lst, element):
    try:
        return lst.index(element)
    except ValueError:
        return -1

def _pretokenize_chunk_w_frequencies(text: bytes, pat: bytes, special_tokens_pattern: bytes) -> dict[tuple[bytes], int]:
    frequencies: dict[tuple[bytes], int] = defaultdict(int)
    for token, is_special in _pretokenize_chunk(text, pat, special_tokens_pattern):
        if not is_special:
            frequencies[token] += 1
    return frequencies

def _create_special_tokens_pattern(special_tokens: list[bytes]) -> bytes:
    if not special_tokens:
        return b"(?!)"

    tokens = sorted(set(special_tokens), key=len, reverse=True)
    return b"(" + b"|".join(re.escape(token) for token in tokens) + b")"

def _pretokenize_chunk_from_file(file: str, start: int, end: int, pat: str, special_tokens_pattern: bytes) -> dict[
    tuple[bytes], int]:
    with open(file, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
    return _pretokenize_chunk_w_frequencies(text, pat.encode(), special_tokens_pattern)

def _add_pair_frequencies(pairs_frequencies: defaultdict[tuple[bytes, bytes], int], frequencies: dict[tuple[bytes, ...], int]):
    for word in frequencies.keys():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs_frequencies[pair] += frequencies[word]

def _subtract_pair_frequencies(pairs_frequencies: defaultdict[tuple[bytes, bytes], int], frequencies: dict[tuple[bytes, ...], int]):
    for word in frequencies.keys():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs_frequencies[pair] -= frequencies[word]

def _replace_pair_in_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[tuple[bytes, ...], bool]:
    if len(word) < 2:
        return word, False
    new_word = []
    word_changed = False
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(pair[0] + pair[1])
            i += 2
            word_changed = True
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word), word_changed

def _replace_pair_in_frequencies(frequencies: dict[tuple[bytes, ...], int], pair_to_replace: tuple[bytes, bytes]) -> tuple[list[tuple[bytes, ...]], list[tuple[bytes, ...]]]:
    words_to_change: list[tuple[tuple[bytes, ...], tuple[bytes, ...], int]] = []
    for word in frequencies.keys():
        new_word, word_changed = _replace_pair_in_word(word, pair_to_replace)
        if word_changed:
            words_to_change.append((word, tuple(new_word), frequencies[word]))
    for word, new_word, count in words_to_change:
        frequencies[new_word] = count
    return [w[0] for w in words_to_change], [w[1] for w in words_to_change]

class PepeBPEFull:

    def __init__(self):
        self.vocab: dict[int, bytes] = dict()
        self.reverse_vocab: dict[bytes, int] = dict()
        self.special_tokens: list[str] = list()
        self.merges: dict[tuple[bytes, bytes], int] = dict()
        self.last_free_index: int = 0

    def _merge_most_frequent_pair(self, pairs_frequencies: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        final_pair = max(pairs_frequencies, key=lambda p: (pairs_frequencies[p], p))

        self.vocab[self.last_free_index] = final_pair[0] + final_pair[1]
        self.last_free_index += 1
        ind = len(self.merges)
        self.merges[(final_pair[0], final_pair[1])] = ind
        return final_pair

    def train_bpe_from_scratch(self,
                               input_path: str | os.PathLike,
                               vocab_size: int,
                               special_tokens: list[str],
                               num_chunks: int = 128,
                               num_processes: int = 8,
                               **kwargs):
        for token in special_tokens:
            self.vocab[self.last_free_index] = token.encode()
            self.special_tokens.append(token)
            self.last_free_index += 1

        for i in range(256):
            self.vocab[self.last_free_index] = bytes([i])
            self.last_free_index += 1

        with open(input_path, "rb") as f:
            special_tokens_pattern = _create_special_tokens_pattern([s.encode("utf-8") for s in special_tokens])
            chunks = find_chunk_boundaries(f, num_chunks, special_tokens_pattern)

        frequencies: dict[tuple[bytes, ...], int] = defaultdict(int)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(chunks[:-1], chunks[1:]):
                futures.append(executor.submit(_pretokenize_chunk_from_file, input_path, start, end, PAT, special_tokens_pattern))

            for future in tqdm(futures, desc="Pre-tokenizing chunks"):
                chunk_vocab = future.result()
                for token, count in chunk_vocab.items():
                    frequencies[token] += count

        pairs_frequencies: dict[tuple[bytes, bytes], int] = defaultdict(int)
        frequencies_to_pass = frequencies
        # Calculate pair frequencies
        with tqdm(total=vocab_size - len(self.vocab), desc="Merging pairs") as pbar:
            while len(self.vocab) < vocab_size:
                _add_pair_frequencies(pairs_frequencies, frequencies_to_pass)
                pair = self._merge_most_frequent_pair(pairs_frequencies)
                old_words, new_words = _replace_pair_in_frequencies(frequencies, pair)
                frequencies_to_pass = {w: frequencies[w] for w in old_words}
                _subtract_pair_frequencies(pairs_frequencies, frequencies_to_pass)
                for word in old_words:
                    del frequencies[word]
                frequencies_to_pass = {w: frequencies[w] for w in new_words}
                pbar.update(1)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def save_vocab(self, vocab_path: str):
        os.makedirs(vocab_path, exist_ok=True)
        with open(os.path.join(vocab_path, "tokenizer.json"), "w") as f:
            f.write(
                json.dumps(
                    {"tokens": {
                        "trained": {token.decode("Latin-1"): i for i, token in self.vocab.items()},
                        "special_tokens": self.special_tokens,
                    },
                     "merges": [[pair[0].decode("Latin-1"), pair[1].decode("Latin-1")] for pair in self.merges],
                     "last_free_index": self.last_free_index,
                    })
                )

    def encode(self, s: str) -> list[int]:
        # 1. pre-tokenize
        special_tokens_pattern = _create_special_tokens_pattern([st.encode("utf-8") for st in self.special_tokens])
        pretokenized_tokens_iterator = _pretokenize_chunk(s.encode("utf-8"), PAT.encode(), special_tokens_pattern)
        # 2. apply merges
        encoded_tokens: list[int] = []
        for token, is_special in pretokenized_tokens_iterator:
            if is_special:
                encoded_tokens.append(self.reverse_vocab.get(b"".join(token), 0))
            else:
                while len(token) > 1:
                    best = min(zip(token[:-1], token[1:]), key=lambda x: self.merges.get(x, float("inf")))
                    if best not in self.merges:
                        break
                    token, _ = _replace_pair_in_word(token, best)
                encoded_tokens.extend((self.reverse_vocab.get(t, 0) for t in token))
        return encoded_tokens

    def encode_iterable(self, s: Iterable[str]) -> Iterator[int]:
        for sub_str in s:
            for token in self.encode(sub_str):
                yield token

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab.get(i, b"") for i in ids]).decode("utf-8", errors="replace")

    def load_vocab(self, vocab_path: str):
        with open(os.path.join(vocab_path, "tokenizer.json"), "r") as f:
            tokens = json.loads(f.read())
        self.last_free_index = int(tokens["last_free_index"])
        self.vocab = {i: v.encode("Latin-1") for v, i in tokens["tokens"]["trained"].items()}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = tokens["tokens"]["special_tokens"]
        self.merges = {(v[0].encode("Latin-1"), v[1].encode("Latin-1")): i for i, v in enumerate(tokens["merges"])}


if __name__ == "__main__":
    tokenizer = PepeBPEFull()
    DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    tokenizer.train_bpe_from_scratch(DATA_PATH, 10000, ["<|endoftext|>"])
    tokenizer.save_vocab(os.path.join(DATA_PATH.split(".")[0], "vocab"))


