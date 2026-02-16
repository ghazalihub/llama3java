# Llama3.CSharp

A high-performance Llama 3 and 3.1 inference engine written in C#. This is a complete port of `Llama.java`, featuring SIMD-optimized kernels and multi-threaded execution.

## Features

- **GGUF Support**: Loads model weights and metadata from GGUF files.
- **Quantization**: Supports `Q4_0` and `Q8_0` quantized models with SIMD optimization.
- **SIMD Optimized**: Manual vectorization using `System.Runtime.Intrinsics` (AVX/AVX2) for maximum performance.
- **Multi-threaded**: Parallelized matrix-vector multiplication using `Parallel.For`.
- **Tokenizer**: Full BPE tokenizer implementation compatible with Llama 3.
- **Llama 3.1 Support**: Includes RoPE scaling logic for extended context lengths.
- **Interactive Chat**: Support for interactive chat sessions with Llama 3 templates.

## Prerequisites

- .NET 10 SDK or later

## Building

To build the optimized executable:

```bash
dotnet build -c Release
```

## Usage

### Instruct Mode (Single Prompt)

```bash
dotnet run -c Release -- --model <path_to_model.gguf> --prompt "Why is the sky blue?"
```

### Chat Mode (Interactive)

```bash
dotnet run -c Release -- --model <path_to_model.gguf> --chat
```

### Options

- `--model, -m`: Path to the GGUF model file.
- `--prompt, -p`: Input prompt for instruct mode.
- `--chat, -i`: Enable interactive chat mode.
- `--temp`: Temperature for sampling (default: 0.1).
- `--top-p`: Top-p (nucleus) sampling (default: 0.9).
- `--seed`: Random seed for sampling.
- `--stream`: Print tokens as they are generated (default: true).
- `--echo`: Echo the prompt tokens (default: false).
