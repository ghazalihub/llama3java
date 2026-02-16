# Llama3.zig

A high-performance Llama 3 and 3.1 inference engine written in Zig. This is a complete port of `Llama.java`, featuring SIMD-optimized kernels and multi-threaded execution.

## Features

- **GGUF Support**: Loads model weights and metadata from GGUF files.
- **Quantization**: Supports `Q4_0` and `Q8_0` quantized models with SIMD optimization.
- **SIMD Optimized**: Manual vectorization using Zig's `@Vector` for maximum performance on supported CPUs.
- **Multi-threaded**: Parallelized matrix-vector multiplication using a thread pool.
- **Tokenizer**: Full BPE tokenizer implementation compatible with Llama 3.
- **Llama 3.1 Support**: Includes RoPE scaling logic for extended context lengths.
- **Interactive Chat**: Support for interactive chat sessions with Llama 3 templates.

## Prerequisites

- Zig 0.13.0

## Building

To build the optimized executable:

```bash
zig build-exe llama.zig -OReleaseFast -Dcpu=native
```

## Usage

### Instruct Mode (Single Prompt)

```bash
./llama --model <path_to_model.gguf> --prompt "Why is the sky blue?"
```

### Chat Mode (Interactive)

```bash
./llama --model <path_to_model.gguf> --chat
```

### Options

- `--model, -m`: Path to the GGUF model file.
- `--prompt, -p`: Input prompt for instruct mode.
- `--chat, -i`: Enable interactive chat mode.
- `--temp`: Temperature for sampling (default: 0.1).
- `--top-p`: Top-p (nucleus) sampling (default: 0.9).
- `--threads, -t`: Number of threads for parallel execution.
- `--seed`: Random seed for sampling.
- `--stream`: Print tokens as they are generated (true/false).
- `--echo`: Echo the prompt tokens to stderr (true/false).

## Performance Tips

- Always compile with `-OReleaseFast -Dcpu=native` for the best performance.
- Use `--threads` to match the number of physical cores on your machine.
