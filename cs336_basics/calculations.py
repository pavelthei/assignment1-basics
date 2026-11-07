GPT_2_XL = {
    "vocab_size": 50257,
    "d_model": 1600,
    "num_heads": 25,
    "d_ff": 4_288,
    "context_len": 1_024,
    "num_layers": 48,
}


def calculate_bytes(config: dict, fp_bytes: int = 4) -> tuple[int, dict[str, int]]:
    params_size = dict()
    embedding_bytes = config["vocab_size"] * config["d_model"]
    params_size["embedding_bytes"] = embedding_bytes * fp_bytes

    attention_bytes = 4 * config["d_model"] ** 2
    ff_bytes = config["d_model"] * config["d_ff"] * 3
    norn_bytes = 2 * config["d_model"]
    transformer_block_bytes = attention_bytes + ff_bytes + norn_bytes
    params_size["transformer_block_bytes"] = transformer_block_bytes * fp_bytes
    transformer_layers_bytes = transformer_block_bytes * config["num_layers"]
    params_size["transformer_layers_bytes"] = transformer_layers_bytes * fp_bytes
    output_bytes = config["d_model"] * config["vocab_size"] + config["d_model"]
    params_size["output_bytes"] = output_bytes * fp_bytes
    return (embedding_bytes + transformer_layers_bytes + output_bytes) * fp_bytes, params_size


def calculate_flops(config: dict, seq_len: int) -> tuple[int, dict[str, int]]:
    flops_size = dict()
    s, d, h = seq_len, config["d_model"], config["num_heads"]
    d_k = d // h
    d_ff, V, L = config["d_ff"], config["vocab_size"], config["num_layers"]

    qkv_proj = 2 * s * d * d * 3
    out_proj = 2 * s * d * d
    qk = 2 * s * s * d_k * h
    av = 2 * s * s * d_k * h
    softmax = s * s * h * 6
    attention_flops = qkv_proj + out_proj + qk + av + softmax
    flops_size["attention_flops"] = attention_flops


    # ---

    ffn = 2 * s * d * d_ff * 3 + 2 * s * d_ff * 3
    flops_size["ffn_flops"] = ffn
    norms = 2 * s * d * 5
    block_flops = attention_flops + ffn + norms
    flops_size["transformer_block_flops"] = block_flops

    final_norm = s * d * 5
    out_flops = 2 * s * d * V

    return block_flops * L + final_norm + out_flops, flops_size





if __name__ == "__main__":
    bytes, detailed_sizes = calculate_bytes(GPT_2_XL)
    print(f"GPT-2 XL model size: {bytes / 1e9:.2f} GB")
    for category, size in detailed_sizes.items():
        print(f"* {category.replace('_', ' ').title()}: {size / 1e9:.2f} GB")

    flops, detailed_flops = calculate_flops(GPT_2_XL, seq_len=GPT_2_XL["context_len"])
    print(f"\nGPT-2 XL FLOPs for a single forward pass with sequence length {GPT_2_XL['context_len']}: {flops / 1e12:.2f} TFLOPs")
    for category, size in detailed_flops.items():
        print(f"* {category.replace('_', ' ').title()}: {size / 1e12:.2f} TFLOPs")

    flops, detailed_flops = calculate_flops(GPT_2_XL, seq_len=1_000_000)
    print(f"\nGPT-2 XL FLOPs for a single forward pass with sequence length 1000000: {flops / 1e12:.2f} TFLOPs")
    for category, size in detailed_flops.items():
        print(f"* {category.replace('_', ' ').title()}: {size / 1e12:.2f} TFLOPs")