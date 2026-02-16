using System;
using System.Collections.Generic;
using System.Linq;

namespace Llama3.CSharp;

public class Sampler
{
    private readonly Random _rng;

    public Sampler(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public int Sample(float[] logits, float temp, float topp)
    {
        if (temp == 0) return Argmax(logits);

        for (int i = 0; i < logits.Length; i++) logits[i] /= temp;
        MathKernels.Softmax(logits);

        if (topp <= 0 || topp >= 1) return CategoricalSample(logits);
        return ToppSample(logits, topp);
    }

    private int Argmax(float[] logits)
    {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxVal)
            {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private int CategoricalSample(float[] probs)
    {
        float r = (float)_rng.NextDouble();
        float cdf = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cdf += probs[i];
            if (r < cdf) return i;
        }
        return probs.Length - 1;
    }

    private int ToppSample(float[] probs, float topp)
    {
        float threshold = (1.0f - topp) / probs.Length;
        var candidates = new List<(float p, int i)>();
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] > threshold) candidates.Add((probs[i], i));
        }

        if (candidates.Count == 0) return Argmax(probs);

        candidates.Sort((a, b) => b.p.CompareTo(a.p));

        float cumulativeProb = 0;
        int lastIdx = candidates.Count - 1;
        for (int i = 0; i < candidates.Count; i++)
        {
            cumulativeProb += candidates[i].p;
            if (cumulativeProb > topp)
            {
                lastIdx = i;
                break;
            }
        }

        float r = (float)_rng.NextDouble() * cumulativeProb;
        float cdf = 0;
        for (int i = 0; i <= lastIdx; i++)
        {
            cdf += candidates[i].p;
            if (r < cdf) return candidates[i].i;
        }
        return candidates[lastIdx].i;
    }
}
