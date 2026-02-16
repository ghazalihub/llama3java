using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Llama3.CSharp;

public class Vocabulary
{
    public string[] Tokens { get; }
    public Dictionary<string, int> TokenToIndex { get; }

    public Vocabulary(string[] tokens)
    {
        Tokens = tokens;
        TokenToIndex = new Dictionary<string, int>(tokens.Length);
        for (int i = 0; i < tokens.Length; i++)
        {
            TokenToIndex[tokens[i]] = i;
        }
    }

    public int? GetIndex(string token) => TokenToIndex.TryGetValue(token, out int index) ? index : null;
}

public class Tokenizer
{
    private readonly Vocabulary _vocab;
    private readonly Dictionary<(int, int), int> _merges;
    private readonly Dictionary<string, int> _specialTokens;
    private readonly uint[] _b2u = new uint[256];
    private readonly Dictionary<uint, byte> _u2b = new();

    public Tokenizer(Vocabulary vocab, string[] mergesRaw, Dictionary<string, int> specialTokens)
    {
        _vocab = vocab;
        _specialTokens = specialTokens;
        _merges = new Dictionary<(int, int), int>(mergesRaw.Length);

        foreach (var line in mergesRaw)
        {
            var parts = line.Split(' ');
            if (parts.Length != 2) continue;
            var id1 = _vocab.GetIndex(parts[0]);
            var id2 = _vocab.GetIndex(parts[1]);
            if (id1 == null || id2 == null) continue;

            var combined = parts[0] + parts[1];
            var cid = _vocab.GetIndex(combined);
            if (cid != null)
            {
                _merges[(id1.Value, id2.Value)] = cid.Value;
            }
        }

        InitByteUnicodeMap();
    }

    private void InitByteUnicodeMap()
    {
        var bs = new List<byte>();
        for (int i = '!'; i <= '~'; i++) bs.Add((byte)i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add((byte)i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add((byte)i);

        var cs = bs.Select(b => (uint)b).ToList();
        uint n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains((byte)b))
            {
                bs.Add((byte)b);
                cs.Add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < 256; i++)
        {
            _b2u[bs[i]] = cs[i];
            _u2b[cs[i]] = bs[i];
        }
    }

    public int[] Encode(string text, string[] allowedSpecial)
    {
        foreach (var s in allowedSpecial)
        {
            if (text == s && _specialTokens.TryGetValue(s, out int id))
            {
                return new[] { id };
            }
        }

        var ids = new List<int>();
        foreach (var b in Encoding.UTF8.GetBytes(text))
        {
            var token = GetTokenFromByte(b);
            var id = _vocab.GetIndex(token);
            if (id != null) ids.Add(id.Value);
        }

        while (ids.Count >= 2)
        {
            int? bestPairIdx = null;
            int bestIdx = int.MaxValue;

            for (int i = 0; i < ids.Count - 1; i++)
            {
                if (_merges.TryGetValue((ids[i], ids[i + 1]), out int mergeIdx))
                {
                    if (mergeIdx < bestIdx)
                    {
                        bestIdx = mergeIdx;
                        bestPairIdx = i;
                    }
                }
            }

            if (bestPairIdx != null)
            {
                int i = bestPairIdx.Value;
                int mid = bestIdx;
                ids[i] = mid;
                ids.RemoveAt(i + 1);
            }
            else
            {
                break;
            }
        }

        return ids.ToArray();
    }

    public string Decode(int[] tokens)
    {
        var res = new List<byte>();
        foreach (var t in tokens)
        {
            var s = _vocab.Tokens[t];
            foreach (var cp in GetCodepoints(s))
            {
                if (_u2b.TryGetValue(cp, out byte b))
                {
                    res.Add(b);
                }
                else
                {
                    res.AddRange(Encoding.UTF8.GetBytes(char.ConvertFromUtf32((int)cp)));
                }
            }
        }
        return Encoding.UTF8.GetString(res.ToArray());
    }

    private string GetTokenFromByte(byte b)
    {
        uint code = _b2u[b];
        return char.ConvertFromUtf32((int)code);
    }

    private static IEnumerable<uint> GetCodepoints(string s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                yield return (uint)char.ConvertToUtf32(s[i], s[i + 1]);
                i++;
            }
            else
            {
                yield return (uint)s[i];
            }
        }
    }

    public Dictionary<string, int> SpecialTokens => _specialTokens;
}
