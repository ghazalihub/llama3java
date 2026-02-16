using System;

namespace Llama3.CSharp;

public record Config
{
    public int Dim { get; init; }
    public int HiddenDim { get; init; }
    public int NLayers { get; init; }
    public int NHeads { get; init; }
    public int NKvHeads { get; init; }
    public int VocabSize { get; init; }
    public int SeqLen { get; init; }
    public float RopeTheta { get; init; }
    public float Eps { get; init; }
    public int HeadSize { get; init; }

    public bool RopeScaling { get; init; }
    public float ScaleFactor { get; init; } = 8.0f;
    public float LoFreqFactor { get; init; } = 1.0f;
    public float HiFreqFactor { get; init; } = 3.0f;
    public float OldContextLen { get; init; } = 8192.0f;

    public static Config FromGGUF(GGUF gguf)
    {
        int dim = (int)(uint)gguf.Metadata["llama.embedding_length"];
        int hiddenDim = (int)(uint)gguf.Metadata["llama.feed_forward_length"];
        int nLayers = (int)(uint)gguf.Metadata["llama.block_count"];
        int nHeads = (int)(uint)gguf.Metadata["llama.attention.head_count"];
        int nKvHeads = gguf.Metadata.TryGetValue("llama.attention.head_count_kv", out var kvh) ? (int)(uint)kvh : nHeads;
        int vocabSize = ((object[])gguf.Metadata["tokenizer.ggml.tokens"]).Length;
        int seqLen = (int)(uint)gguf.Metadata["llama.context_length"];
        float ropeTheta = gguf.Metadata.TryGetValue("llama.rope.freq_base", out var rt) ? (float)rt : 10000.0f;
        float eps = gguf.Metadata.TryGetValue("llama.attention.layer_norm_rms_epsilon", out var e) ? (float)e : 1e-5f;

        bool ropeScaling = false;
        float scaleFactor = 8.0f;
        float loFreqFactor = 1.0f;
        float hiFreqFactor = 3.0f;
        float oldContextLen = 8192.0f;

        if (gguf.Metadata.TryGetValue("llama.rope.scaling.type", out var rst))
        {
            if ((string)rst == "linear" || (string)rst == "yarn") ropeScaling = true;
        }
        if (gguf.Metadata.TryGetValue("llama.rope.scaling.factor", out var rsf)) scaleFactor = (float)rsf;
        if (gguf.Metadata.TryGetValue("llama.rope.scaling.low_freq_factor", out var rslf)) loFreqFactor = (float)rslf;
        if (gguf.Metadata.TryGetValue("llama.rope.scaling.high_freq_factor", out var rshf)) hiFreqFactor = (float)rshf;
        if (gguf.Metadata.TryGetValue("llama.rope.scaling.orig_ctx_len", out var rsoc)) oldContextLen = (uint)rsoc;

        return new Config
        {
            Dim = dim,
            HiddenDim = hiddenDim,
            NLayers = nLayers,
            NHeads = nHeads,
            NKvHeads = nKvHeads,
            VocabSize = vocabSize,
            SeqLen = seqLen,
            RopeTheta = ropeTheta,
            Eps = eps,
            HeadSize = dim / nHeads,
            RopeScaling = ropeScaling,
            ScaleFactor = scaleFactor,
            LoFreqFactor = loFreqFactor,
            HiFreqFactor = hiFreqFactor,
            OldContextLen = oldContextLen
        };
    }
}
