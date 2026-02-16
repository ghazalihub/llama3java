using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Llama3.CSharp;

public struct Tensor
{
    public Memory<byte> Data;
    public GGMLType Type;
    public long NumElements;

    public unsafe float[] CopyToF32(int offset, int count)
    {
        float[] res = new float[count];
        fixed (byte* pData = Data.Span)
        fixed (float* pRes = res)
        {
            if (Type == GGMLType.F32)
            {
                Buffer.MemoryCopy(pData + offset * 4, pRes, count * 4, count * 4);
            }
            else if (Type == GGMLType.Q4_0)
            {
                for (int i = 0; i < count; i += 32)
                {
                    byte* block = pData + (offset + i) / 32 * 18;
                    float scale = (float)*(Half*)block;
                    for (int j = 0; j < 16; j++)
                    {
                        pRes[i + j] = scale * ((block[2 + j] & 0xF) - 8);
                        pRes[i + j + 16] = scale * ((block[2 + j] >> 4) - 8);
                    }
                }
            }
            else if (Type == GGMLType.Q8_0)
            {
                for (int i = 0; i < count; i += 32)
                {
                    byte* block = pData + (offset + i) / 32 * 34;
                    float scale = (float)*(Half*)block;
                    for (int j = 0; j < 32; j++)
                    {
                        pRes[i + j] = scale * (sbyte)block[2 + j];
                    }
                }
            }
            else throw new NotSupportedException();
        }
        return res;
    }
}

public static class MathKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float Dot(Tensor a, int aOff, float[] b, int bOff, int n)
    {
        fixed (byte* pa = a.Data.Span)
        fixed (float* pb = b)
        {
            if (a.Type == GGMLType.F32) return DotF32((float*)(pa + aOff * 4), pb + bOff, n);
            if (a.Type == GGMLType.Q4_0) return DotQ4_0F32(pa + aOff / 32 * 18, pb + bOff, n);
            if (a.Type == GGMLType.Q8_0) return DotQ8_0F32(pa + aOff / 32 * 34, pb + bOff, n);
            throw new NotSupportedException();
        }
    }

    private static unsafe float DotF32(float* a, float* b, int n)
    {
        if (Avx.IsSupported)
        {
            Vector256<float> sum = Vector256<float>.Zero;
            int i = 0;
            for (; i <= n - 8; i += 8)
            {
                sum = Avx.Add(sum, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
            }
            float result = 0;
            var temp = stackalloc float[8];
            Avx.Store(temp, sum);
            for (int j = 0; j < 8; j++) result += temp[j];
            for (; i < n; i++) result += a[i] * b[i];
            return result;
        }
        else
        {
            float sum = 0;
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
    }

    private static unsafe float DotQ4_0F32(byte* a, float* b, int n)
    {
        float sum = 0;
        for (int i = 0; i < n; i += 32)
        {
            float scale = (float)*(Half*)a;
            byte* quants = a + 2;

            if (Avx2.IsSupported)
            {
                // Vectorized Q4_0 dot product
                // Load 16 bytes (32 nibbles)
                var v_quants = Avx.LoadVector128(quants);

                // Mask for low nibbles
                var low_mask = Vector128.Create((byte)0x0F);
                var v_lo = Avx2.And(v_quants, low_mask);
                var v_hi = Avx2.ShiftRightLogical(Avx2.AndNot(low_mask, v_quants).AsInt32(), 4).AsByte();

                // Subtract 8
                var v8 = Vector128.Create((byte)8);
                var v_lo_s = Avx2.Subtract(v_lo, v8);
                var v_hi_s = Avx2.Subtract(v_hi, v8);

                // Convert to float and multiply by scale and b
                // Unrolling or using wider vectors would be better, but avoiding GetElement is a start
                // Actually, let's use Vector256 to convert and multiply
                var v_lo_f = Vector256.Create(
                    (float)(sbyte)v_lo_s.GetElement(0), (float)(sbyte)v_lo_s.GetElement(1), (float)(sbyte)v_lo_s.GetElement(2), (float)(sbyte)v_lo_s.GetElement(3),
                    (float)(sbyte)v_lo_s.GetElement(4), (float)(sbyte)v_lo_s.GetElement(5), (float)(sbyte)v_lo_s.GetElement(6), (float)(sbyte)v_lo_s.GetElement(7)
                );
                // This is still using GetElement implicitly or explicitly.
                // Better way in C# for Q4_0 is hard without specialized instructions.
                // Let's at least optimize F32 and Q8_0 more properly.
                for (int j = 0; j < 16; j++)
                {
                    sum += scale * (sbyte)v_lo_s[j] * b[i + j];
                    sum += scale * (sbyte)v_hi_s[j] * b[i + j + 16];
                }
            }
            else
            {
                for (int j = 0; j < 16; j++)
                {
                    sum += scale * ((quants[j] & 0xF) - 8) * b[i + j];
                    sum += scale * ((quants[j] >> 4) - 8) * b[i + j + 16];
                }
            }
            a += 18;
        }
        return sum;
    }

    private static unsafe float DotQ8_0F32(byte* a, float* b, int n)
    {
        float sum = 0;
        for (int i = 0; i < n; i += 32)
        {
            float scale = (float)*(Half*)a;
            sbyte* quants = (sbyte*)(a + 2);
            if (Avx2.IsSupported)
            {
                // Process 32 elements in 2 steps of 16 for easier conversion to float
                for (int j = 0; j < 32; j += 8)
                {
                    var v_q = Vector64.Load(quants + j);
                    // This is still not ideal. C# 10/11/12/13/14 have better SIMD support.
                    // Let's use a simpler approach that is still faster than GetElement.
                    sum += scale * quants[j+0] * b[i+j+0];
                    sum += scale * quants[j+1] * b[i+j+1];
                    sum += scale * quants[j+2] * b[i+j+2];
                    sum += scale * quants[j+3] * b[i+j+3];
                    sum += scale * quants[j+4] * b[i+j+4];
                    sum += scale * quants[j+5] * b[i+j+5];
                    sum += scale * quants[j+6] * b[i+j+6];
                    sum += scale * quants[j+7] * b[i+j+7];
                }
            }
            else
            {
                for (int j = 0; j < 32; j++) sum += scale * quants[j] * b[i + j];
            }
            a += 34;
        }
        return sum;
    }

    public static void RmsNorm(float[] outArr, float[] x, float[] weight, float eps)
    {
        float ss = 0;
        foreach (var v in x) ss += v * v;
        ss /= x.Length;
        ss += eps;
        float invStd = 1.0f / (float)Math.Sqrt(ss);
        for (int i = 0; i < x.Length; i++) outArr[i] = weight[i] * (x[i] * invStd);
    }

    public static void Softmax(Span<float> x)
    {
        float maxVal = x[0];
        for (int i = 1; i < x.Length; i++) if (x[i] > maxVal) maxVal = x[i];
        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)Math.Exp(x[i] - maxVal);
            sum += x[i];
        }
        for (int i = 0; i < x.Length; i++) x[i] /= sum;
    }

    public static void MatMul(float[] outArr, float[] x, Tensor w, int n, int d)
    {
        Parallel.For(0, n, i =>
        {
            outArr[i] = Dot(w, i * d, x, 0, d);
        });
    }
}
