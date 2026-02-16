using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;

namespace Llama3.CSharp;

// Helper to wrap raw pointer in Memory<T> without copying
public sealed unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
    private readonly T* _pointer;
    private readonly long _length;

    public UnmanagedMemoryManager(T* pointer, long length)
    {
        _pointer = pointer;
        _length = length;
    }

    public override Span<T> GetSpan() => new Span<T>(_pointer, (int)Math.Min(_length, int.MaxValue));

    protected override void Dispose(bool disposing) { }

    public override MemoryHandle Pin(int elementIndex = 0) => new MemoryHandle(_pointer + elementIndex);

    public override void Unpin() { }
}

class Program
{
    static void Main(string[] args)
    {
        string modelPath = null;
        string prompt = null;
        bool chat = false;
        float temp = 0.1f;
        float topp = 0.9f;
        int? seed = null;
        bool stream = true;
        bool echo = false;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-m" || args[i] == "--model") modelPath = args[++i];
            else if (args[i] == "-p" || args[i] == "--prompt") prompt = args[++i];
            else if (args[i] == "-i" || args[i] == "--chat") chat = true;
            else if (args[i] == "--temp") temp = float.Parse(args[++i]);
            else if (args[i] == "--top-p") topp = float.Parse(args[++i]);
            else if (args[i] == "--seed") seed = int.Parse(args[++i]);
            else if (args[i] == "--stream") stream = bool.Parse(args[++i]);
            else if (args[i] == "--echo") echo = bool.Parse(args[++i]);
        }

        if (modelPath == null || (prompt == null && !chat))
        {
            Console.WriteLine("Usage: Llama3.CSharp --model <path> [--prompt <string>] [--chat] [--temp <float>] [--top-p <float>] [--seed <int>] [--stream true|false] [--echo true|false]");
            return;
        }

        Console.WriteLine($"Loading model {modelPath}...");
        var gguf = GGUF.Load(modelPath);
        var config = Config.FromGGUF(gguf);

        using var mmf = MemoryMappedFile.CreateFromFile(modelPath, FileMode.Open);
        using var accessor = mmf.CreateViewAccessor();
        unsafe
        {
            byte* pData = null;
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref pData);
            long fileLength = new FileInfo(modelPath).Length;

            Console.WriteLine("Loading weights...");
            // Use UnmanagedMemoryManager to wrap the pointer in a Memory<byte> without copying
            var manager = new UnmanagedMemoryManager<byte>(pData, fileLength);
            var mmapMemory = manager.Memory;
            var weights = Weights.Load(config, gguf, mmapMemory);

            Console.WriteLine("Initializing tokenizer...");
            var tokensMeta = (object[])gguf.Metadata["tokenizer.ggml.tokens"];
            var tokens = new string[tokensMeta.Length];
            for (int i = 0; i < tokensMeta.Length; i++) tokens[i] = (string)tokensMeta[i];
            var vocab = new Vocabulary(tokens);

            var mergesMeta = (object[])gguf.Metadata["tokenizer.ggml.merges"];
            var mergesRaw = new string[mergesMeta.Length];
            for (int i = 0; i < mergesMeta.Length; i++) mergesRaw[i] = (string)mergesMeta[i];

            var specialTokens = new Dictionary<string, int>();
            int? bosId = vocab.GetIndex("<|begin_of_text|>");
            if (bosId != null) specialTokens["<|begin_of_text|>"] = bosId.Value;
            int? eosId = vocab.GetIndex("<|end_of_text|>");
            if (eosId != null) specialTokens["<|end_of_text|>"] = eosId.Value;
            int? eotId = vocab.GetIndex("<|eot_id|>");
            if (eotId != null) specialTokens["<|eot_id|>"] = eotId.Value;

            var tokenizer = new Tokenizer(vocab, mergesRaw, specialTokens);
            var state = new State(config);
            var sampler = new Sampler(seed);

            if (chat)
            {
                RunChat(config, state, weights, tokenizer, sampler, temp, topp, stream);
            }
            else
            {
                RunInstruct(config, state, weights, tokenizer, sampler, prompt, temp, topp, stream, echo);
            }
        }
    }

    static void RunInstruct(Config config, State state, Weights weights, Tokenizer tokenizer, Sampler sampler, string prompt, float temp, float topp, bool stream, bool echo)
    {
        var promptTokens = tokenizer.Encode(prompt, Array.Empty<string>());
        int pos = 0;
        int next = tokenizer.SpecialTokens.TryGetValue("<|begin_of_text|>", out var bos) ? bos : 128000;

        var sw = Stopwatch.StartNew();
        foreach (var t in promptTokens)
        {
            Transformer.Forward(config, state, weights, next, pos);
            if (echo) Console.Write(tokenizer.Decode(new[] { t }));
            pos++;
            next = t;
        }

        while (pos < config.SeqLen)
        {
            Transformer.Forward(config, state, weights, next, pos);
            next = sampler.Sample(state.Logits, temp, topp);

            if (tokenizer.SpecialTokens.TryGetValue("<|end_of_text|>", out var eos) && next == eos) break;
            if (tokenizer.SpecialTokens.TryGetValue("<|eot_id|>", out var eot) && next == eot) break;

            if (stream) Console.Write(tokenizer.Decode(new[] { next }));
            pos++;
        }
        sw.Stop();
        Console.WriteLine($"\n\n{pos} tokens in {sw.Elapsed.TotalSeconds:F2}s ({pos / sw.Elapsed.TotalSeconds:F2} tok/s)");
    }

    static void RunChat(Config config, State state, Weights weights, Tokenizer tokenizer, Sampler sampler, float temp, float topp, bool stream)
    {
        int pos = 0;
        int next = tokenizer.SpecialTokens.TryGetValue("<|begin_of_text|>", out var bos) ? bos : 128000;

        while (true)
        {
            Console.Write("\n> ");
            string line = Console.ReadLine();
            if (string.IsNullOrEmpty(line) || line == "quit" || line == "exit") break;

            string userPrompt = $"<|start_header_id|>user<|end_header_id|>\n\n{line}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            var promptTokens = tokenizer.Encode(userPrompt, new[] { "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>" });

            foreach (var t in promptTokens)
            {
                Transformer.Forward(config, state, weights, next, pos);
                pos++;
                next = t;
            }

            while (pos < config.SeqLen)
            {
                Transformer.Forward(config, state, weights, next, pos);
                next = sampler.Sample(state.Logits, temp, topp);

                if (tokenizer.SpecialTokens.TryGetValue("<|eot_id|>", out var eot) && next == eot) break;
                if (tokenizer.SpecialTokens.TryGetValue("<|end_of_text|>", out var eos) && next == eos) break;

                if (stream) Console.Write(tokenizer.Decode(new[] { next }));
                pos++;
            }
            Console.WriteLine();
        }
    }
}
