import os
import json
import regex as re
from concurrent.futures import ProcessPoolExecutor

from typing import BinaryIO, Iterator
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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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

def _pretokenize_chunk_w_frequencies(text: bytes, pat: bytes, special_tokens_pattern: bytes) -> dict[tuple[bytes], int]:
    frequencies: dict[tuple[bytes], int] = defaultdict(int)
    for token, is_special in _pretokenize_chunk(text, pat, special_tokens_pattern):
        if not is_special:
            frequencies[token] += 1
    return frequencies

def _create_special_tokens_pattern(special_tokens: list[bytes]) -> bytes:
    tokens = [re.escape(token) for token in special_tokens]
    return b"(" + b"|".join(token for token in tokens) + b")"

def _pretokenize_chunk_from_file(file: str, start: int, end: int, pat: str, special_tokens_pattern: bytes) -> dict[
    tuple[bytes], int]:
    with open(file, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
    return _pretokenize_chunk_w_frequencies(text, pat.encode(), special_tokens_pattern)

def _calculate_pair_frequencies(frequencies: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    pairs_frequencies: dict[tuple[bytes], int] = defaultdict(int)
    for word in frequencies.keys():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs_frequencies[pair] += frequencies[word]
    return pairs_frequencies

def _replace_pair_in_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[list[bytes], bool]:
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

def _replace_pair_in_frequencies(frequencies: dict[tuple[bytes], int], pair_to_replace: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    words_to_change: list[tuple[bytes, bytes, int]] = []
    for word in frequencies.keys():
        new_word, word_changed = _replace_pair_in_word(word, pair_to_replace)
        if word_changed:
            words_to_change.append((word, tuple(new_word), frequencies[word]))
    for word, new_word, count in words_to_change:
        del frequencies[word]
        frequencies[new_word] = count
    return frequencies


class PepeBPEFull:

    def __init__(self):
        self.vocab: dict[bytes, int] = defaultdict(int)
        self.special_tokens: dict[bytes, int] = dict()
        self.merges: dict[tuple[bytes, bytes], int] = dict()
        self.last_free_index: int = 0

    def _merge_most_frequent_pair(self, pairs_frequencies: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        if not pairs_frequencies:
            return

        sorted_pairs_frequencies = sorted(
            ((pair, count) for pair, count in pairs_frequencies.items()),
            key=lambda x: (x[1], x[0][0] + x[0][1]),
            reverse=True)

        i = 0
        while i < len(sorted_pairs_frequencies) and sorted_pairs_frequencies[i][0][0] + sorted_pairs_frequencies[i][0][1] in self.vocab:
            i += 1
        if i == len(sorted_pairs_frequencies):
            return
        final_pair = sorted_pairs_frequencies[i][0]

        self.vocab[self.last_free_index] = sorted_pairs_frequencies[i][0][0] + sorted_pairs_frequencies[i][0][1]
        self.last_free_index += 1
        ind = len(self.merges)
        self.merges[sorted_pairs_frequencies[i][0]] = ind
        return final_pair

    def train_bpe_from_scratch(self,
                               input_path: str,
                               vocab_size: int,
                               special_tokens: list[str],
                               num_chunks: int = 128,
                               num_processes: int = 8,
                               **kwargs):
        for i in range(256):
            self.vocab[self.last_free_index] = bytes([i])
            self.last_free_index += 1

        for token in special_tokens:
            self.vocab[self.last_free_index] = token.encode()
            self.special_tokens[token.encode()] = self.last_free_index
            self.last_free_index += 1

        with open(input_path, "rb") as f:
            special_tokens_pattern = _create_special_tokens_pattern([s.encode("utf-8") for s in special_tokens])
            chunks = find_chunk_boundaries(f, num_chunks, special_tokens_pattern)

        frequencies: dict[tuple[bytes], int] = defaultdict(int)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(chunks[:-1], chunks[1:]):
                futures.append(executor.submit(_pretokenize_chunk_from_file, input_path, start, end, PAT, special_tokens_pattern))

            for future in tqdm(futures, desc="Pretokenizing chunks"):
                chunk_vocab = future.result()
                for token, count in chunk_vocab.items():
                    frequencies[token] += count

        frequencies_chunks = [dict(list(frequencies.items())[i::num_processes]) for i in range(num_processes)]
        # Calculate pair frequencies
        with tqdm(total=vocab_size - len(self.vocab), desc="Merging pairs") as pbar:
            while len(self.vocab) < vocab_size:
                
                pairs_frequencies: dict[tuple[bytes, bytes], int] = defaultdict(int)
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    for chunk in frequencies_chunks:
                        futures.append(executor.submit(_calculate_pair_frequencies, chunk))

                    for future in futures:
                        chunk_pairs_frequencies = future.result()
                        for pair, count in chunk_pairs_frequencies.items():
                            pairs_frequencies[pair] += count

                pair = self._merge_most_frequent_pair(pairs_frequencies)
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    for chunk in frequencies_chunks:
                        futures.append(executor.submit(_replace_pair_in_frequencies, chunk, pair))

                frequencies_chunks = []
                for future in futures:
                    frequencies_chunks.append(future.result())
                pbar.update(1)

    def save_vocab(self, vocab_path: str):
        os.makedirs(vocab_path, exist_ok=True)
        with open(os.path.join(vocab_path, "tokenizer.json"), "w") as f:
            f.write(
                json.dumps(
                    {"tokens": {
                        "trained": {token.decode("latin-1"): i for i, token in self.vocab.items()},
                        "special_tokens": {token.decode("latin-1"): i for token, i in self.special_tokens.items()},
                    },
                     "merges": [[pair[0].decode("latin-1"), pair[1].decode("latin-1")] for pair, _ in sorted(self.merges.items(), key=lambda x: x[1])],
                     "last_free_index": self.last_free_index,
                    })
                )

    def encode(self, s: str) -> list[int]:
        # 1. pre-tokenize
        special_tokens_pattern = _create_special_tokens_pattern([s.encode("latin-1") for s in self.special_tokens.keys()])
        pretokenized_tokens_iterator = _pretokenize_chunk(s.encode("utf-8"), PAT.encode(), special_tokens_pattern)
        # 2. apply merges
        encoded_tokens: list[int] = []
        for token, is_special in pretokenized_tokens_iterator:
            if is_special:
                encoded_tokens.append(self.vocab.get(b"".join(token), 0))
            else:
                while len(token) > 1:
                    best = min(zip(token[:-1], token[1:]), key=lambda x: self.merges.get(x, float("inf")))
                    if best not in self.merges:
                        break
                    token, _ = _replace_pair_in_word(token, best)
                encoded_tokens.extend((self.vocab.get(t, 0) for t in token))
        return encoded_tokens


    def load_vocab(self, vocab_path: str):
        with open(os.path.join(vocab_path, "tokenizer.json"), "r") as f:
            tokens = json.loads(f.read())
            self.last_free_index = int(tokens["last_free_index"])
            self.vocab = {v.encode("latin-1"): i for v, i in tokens["tokens"]["trained"].items()}
            self.special_tokens = tokens["tokens"]["special_tokens"]
            self.merges = {(v[0].encode("latin-1"), v[1].encode("latin-1")): i for i, v in enumerate(tokens["merges"])}


if __name__ == "__main__":
    tokenizer = PepeBPEFull()
    DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    tokenizer.train_bpe_from_scratch(DATA_PATH, 10000, ["<|endoftext|>"])
    tokenizer.save_vocab(os.path.join(DATA_PATH.split(".")[0], "vocab"))

