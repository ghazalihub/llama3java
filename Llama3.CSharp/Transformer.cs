using System;
using System.Collections.Generic;
using System.IO;

namespace Llama3.CSharp;

public class Weights
{
    public Tensor TokenEmbd { get; set; }
    public Tensor[] RmsAttW { get; set; }
    public Tensor[] Wq { get; set; }
    public Tensor[] Wk { get; set; }
    public Tensor[] Wv { get; set; }
    public Tensor[] Wo { get; set; }
    public Tensor[] RmsFfnW { get; set; }
    public Tensor[] W1 { get; set; }
    public Tensor[] W2 { get; set; }
    public Tensor[] W3 { get; set; }
    public Tensor RmsFinalW { get; set; }
    public Tensor OutputW { get; set; }

    public static Weights Load(Config config, GGUF gguf, Memory<byte> mmapData)
    {
        var w = new Weights
        {
            TokenEmbd = GetTensor(gguf, mmapData, "token_embd.weight"),
            RmsAttW = new Tensor[config.NLayers],
            Wq = new Tensor[config.NLayers],
            Wk = new Tensor[config.NLayers],
            Wv = new Tensor[config.NLayers],
            Wo = new Tensor[config.NLayers],
            RmsFfnW = new Tensor[config.NLayers],
            W1 = new Tensor[config.NLayers],
            W2 = new Tensor[config.NLayers],
            W3 = new Tensor[config.NLayers]
        };

        for (int i = 0; i < config.NLayers; i++)
        {
            w.RmsAttW[i] = GetTensor(gguf, mmapData, $"blk.{i}.attn_norm.weight");
            w.Wq[i] = GetTensor(gguf, mmapData, $"blk.{i}.attn_q.weight");
            w.Wk[i] = GetTensor(gguf, mmapData, $"blk.{i}.attn_k.weight");
            w.Wv[i] = GetTensor(gguf, mmapData, $"blk.{i}.attn_v.weight");
            w.Wo[i] = GetTensor(gguf, mmapData, $"blk.{i}.attn_output.weight");
            w.RmsFfnW[i] = GetTensor(gguf, mmapData, $"blk.{i}.ffn_norm.weight");
            w.W1[i] = GetTensor(gguf, mmapData, $"blk.{i}.ffn_gate.weight");
            w.W2[i] = GetTensor(gguf, mmapData, $"blk.{i}.ffn_down.weight");
            w.W3[i] = GetTensor(gguf, mmapData, $"blk.{i}.ffn_up.weight");
        }

        w.RmsFinalW = GetTensor(gguf, mmapData, "output_norm.weight");
        w.OutputW = gguf.TensorInfos.ContainsKey("output.weight")
            ? GetTensor(gguf, mmapData, "output.weight")
            : w.TokenEmbd;

        return w;
    }

    private static Tensor GetTensor(GGUF gguf, Memory<byte> mmapData, string name)
    {
        if (!gguf.TensorInfos.TryGetValue(name, out var info)) throw new Exception($"Tensor not found: {name}");
        long offset = gguf.TensorDataOffset + info.Offset;
        long size = info.Type.ByteSizeFor(info.NumElements());
        return new Tensor
        {
            Data = mmapData.Slice((int)offset, (int)size),
            Type = info.Type,
            NumElements = info.NumElements()
        };
    }
}

public class State
{
    public float[] X { get; }
    public float[] Xb { get; }
    public float[] Xb2 { get; }
    public float[] Hb { get; }
    public float[] Hb2 { get; }
    public float[] Q { get; }
    public float[] K { get; }
    public float[] V { get; }
    public float[] Att { get; }
    public float[] Logits { get; }
    public float[] KeyCache { get; }
    public float[] ValueCache { get; }

    public State(Config config)
    {
        int kvDim = config.NKvHeads * config.HeadSize;
        X = new float[config.Dim];
        Xb = new float[config.Dim];
        Xb2 = new float[config.Dim];
        Hb = new float[config.HiddenDim];
        Hb2 = new float[config.HiddenDim];
        Q = new float[config.Dim];
        K = new float[config.Dim];
        V = new float[config.Dim];
        Att = new float[config.NHeads * config.SeqLen];
        Logits = new float[config.VocabSize];
        KeyCache = new float[config.NLayers * config.SeqLen * kvDim];
        ValueCache = new float[config.NLayers * config.SeqLen * kvDim];
    }
}

public static class Transformer
{
    public static void Forward(Config config, State state, Weights weights, int token, int pos)
    {
        int dim = config.Dim;
        int hdim = config.HiddenDim;
        int hsz = config.HeadSize;
        int nhead = config.NHeads;
        int nkv = config.NKvHeads;
        int kvDim = nkv * hsz;
        int kvMul = nhead / nkv;

        float[] embd = weights.TokenEmbd.CopyToF32(token * dim, dim);
        Array.Copy(embd, state.X, dim);

        for (int l = 0; l < config.NLayers; l++)
        {
            MathKernels.RmsNorm(state.Xb, state.X, weights.RmsAttW[l].CopyToF32(0, dim), config.Eps);

            MathKernels.MatMul(state.Q, state.Xb, weights.Wq[l], nhead * hsz, dim);
            MathKernels.MatMul(state.K, state.Xb, weights.Wk[l], nkv * hsz, dim);
            MathKernels.MatMul(state.V, state.Xb, weights.Wv[l], nkv * hsz, dim);

            // RoPE
            for (int h = 0; h < nhead; h++)
            {
                for (int i = 0; i < hsz / 2; i++)
                {
                    float freq = 1.0f / (float)Math.Pow(config.RopeTheta, (float)(2 * i) / hsz);
                    if (config.RopeScaling)
                    {
                        float wlen = 2.0f * (float)Math.PI / freq;
                        float low = config.OldContextLen / config.LoFreqFactor;
                        float high = config.OldContextLen / config.HiFreqFactor;
                        if (wlen > low) freq /= config.ScaleFactor;
                        else if (wlen > high)
                        {
                            float smooth = (config.OldContextLen / wlen - config.LoFreqFactor) / (config.HiFreqFactor - config.LoFreqFactor);
                            freq = (1.0f - smooth) * freq / config.ScaleFactor + smooth * freq;
                        }
                    }
                    float val = pos * freq;
                    float fcr = (float)Math.Cos(val);
                    float fci = (float)Math.Sin(val);

                    float q0 = state.Q[h * hsz + i * 2];
                    float q1 = state.Q[h * hsz + i * 2 + 1];
                    state.Q[h * hsz + i * 2] = q0 * fcr - q1 * fci;
                    state.Q[h * hsz + i * 2 + 1] = q0 * fci + q1 * fcr;

                    if (h < nkv)
                    {
                        float k0 = state.K[h * hsz + i * 2];
                        float k1 = state.K[h * hsz + i * 2 + 1];
                        state.K[h * hsz + i * 2] = k0 * fcr - k1 * fci;
                        state.K[h * hsz + i * 2 + 1] = k0 * fci + k1 * fcr;
                    }
                }
            }

            int loff = l * config.SeqLen * kvDim;
            Array.Copy(state.K, 0, state.KeyCache, loff + pos * kvDim, kvDim);
            Array.Copy(state.V, 0, state.ValueCache, loff + pos * kvDim, kvDim);

            // Multi-head Attention
            for (int h = 0; h < nhead; h++)
            {
                int qOffset = h * hsz;
                int attOffset = h * config.SeqLen;
                for (int t = 0; t <= pos; t++)
                {
                    int kOffset = loff + t * kvDim + (h / kvMul) * hsz;
                    float score = 0;
                    for (int i = 0; i < hsz; i++) score += state.Q[qOffset + i] * state.KeyCache[kOffset + i];
                    score /= (float)Math.Sqrt(hsz);
                    state.Att[attOffset + t] = score;
                }

                MathKernels.Softmax(state.Att.AsSpan(attOffset, pos + 1));

                int xbOffset = h * hsz;
                for (int i = 0; i < hsz; i++) state.Xb[xbOffset + i] = 0;
                for (int t = 0; t <= pos; t++)
                {
                    int vOffset = loff + t * kvDim + (h / kvMul) * hsz;
                    float a = state.Att[attOffset + t];
                    for (int i = 0; i < hsz; i++) state.Xb[xbOffset + i] += a * state.ValueCache[vOffset + i];
                }
            }

            MathKernels.MatMul(state.Xb2, state.Xb, weights.Wo[l], dim, dim);
            for (int i = 0; i < dim; i++) state.X[i] += state.Xb2[i];

            MathKernels.RmsNorm(state.Xb, state.X, weights.RmsFfnW[l].CopyToF32(0, dim), config.Eps);
            MathKernels.MatMul(state.Hb, state.Xb, weights.W1[l], hdim, dim);
            MathKernels.MatMul(state.Hb2, state.Xb, weights.W3[l], hdim, dim);

            for (int i = 0; i < hdim; i++)
            {
                float val = state.Hb[i];
                val *= 1.0f / (1.0f + (float)Math.Exp(-val));
                state.Hb[i] = val * state.Hb2[i];
            }

            MathKernels.MatMul(state.Xb, state.Hb, weights.W2[l], dim, hdim);
            for (int i = 0; i < dim; i++) state.X[i] += state.Xb[i];
        }

        MathKernels.RmsNorm(state.X, state.X, weights.RmsFinalW.CopyToF32(0, dim), config.Eps);
        MathKernels.MatMul(state.Logits, state.X, weights.OutputW, config.VocabSize, dim);
    }
}
