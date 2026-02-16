# Llama3.Rust

A high-performance Llama 3 and 3.1 inference engine written in Rust. This is a complete port of `Llama.java`, featuring SIMD-optimized kernels and multi-threaded execution.

## Features

- **GGUF Support**: Loads model weights and metadata from GGUF files.
- **Quantization**: Supports `Q4_0` and `Q8_0` quantized models with SIMD optimization.
- **SIMD Optimized**: Manual vectorization using AVX2 for maximum performance on x86_64.
- **Multi-threaded**: Parallelized matrix-vector multiplication using `rayon`.
- **Tokenizer**: Full BPE tokenizer implementation compatible with Llama 3.
- **Llama 3.1 Support**: Includes RoPE scaling logic for extended context lengths.
- **Interactive Chat**: Support for interactive chat sessions with Llama 3 templates.

## Prerequisites

- Rust 1.80+

## Building

To build the optimized executable:

```bash
cargo build --release
```

## Usage

### Instruct Mode (Single Prompt)

```bash
./target/release/llama3_rust --model <path_to_model.gguf> --prompt "Why is the sky blue?"
```

### Chat Mode (Interactive)

```bash
./target/release/llama3_rust --model <path_to_model.gguf> --chat
```

### Options

- `--model, -m`: Path to the GGUF model file.
- `--prompt, -p`: Input prompt for instruct mode.
- `--chat`: Enable interactive chat mode.
- `--temp`: Temperature for sampling (default: 0.1).
- `--top-p`: Top-p (nucleus) sampling (default: 0.9).
- `--threads, -t`: Number of threads for parallel execution.
- `--seed`: Random seed for sampling.
- `--stream`: Print tokens as they are generated (default: true).
- `--echo`: Echo the prompt tokens (default: false).

## Performance Tips

- Always build with `--release` for the best performance.
- The engine automatically detects AVX2 support at runtime.
