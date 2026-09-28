from cs336_basics.pepe_bpe_full import PepeBPEFull

if __name__ == '__main__':
    VOCAB_PATH = "data/TinyStoriesV2-GPT4-train/vocab"
    bpe = PepeBPEFull()
    bpe.load_vocab(vocab_path=VOCAB_PATH)
    TEXT = "Hello World!<|endoftext|> How are you?"
    print(bpe.encode(TEXT))
