using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Llama3.CSharp;

public enum GGMLType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

public static class GGMLTypeExtensions
{
    public static int TypeSize(this GGMLType type)
    {
        return type switch
        {
            GGMLType.F32 => 4,
            GGMLType.F16 => 2,
            GGMLType.Q4_0 => 18,
            GGMLType.Q8_0 => 34,
            _ => throw new NotSupportedException($"Unsupported GGML type: {type}")
        };
    }

    public static int BlockSize(this GGMLType type)
    {
        return type switch
        {
            GGMLType.F32 or GGMLType.F16 => 1,
            GGMLType.Q4_0 or GGMLType.Q8_0 => 32,
            _ => throw new NotSupportedException($"Unsupported GGML type: {type}")
        };
    }

    public static long ByteSizeFor(this GGMLType type, long numElements)
    {
        int bsize = type.BlockSize();
        int tsize = type.TypeSize();
        if (numElements % bsize != 0) throw new ArgumentException("numElements must be multiple of block size");
        return (numElements / bsize) * tsize;
    }
}

public enum MetadataValueType : uint
{
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
}

public record GGUFTensorInfo(string Name, long[] Dimensions, GGMLType Type, long Offset)
{
    public long NumElements()
    {
        long res = 1;
        foreach (var d in Dimensions) res *= d;
        return res;
    }
}

public class GGUF
{
    public const uint Magic = 0x46554747;
    public uint Version { get; private set; }
    public long TensorCount { get; private set; }
    public long MetadataCount { get; private set; }
    public Dictionary<string, object> Metadata { get; } = new();
    public Dictionary<string, GGUFTensorInfo> TensorInfos { get; } = new();
    public long TensorDataOffset { get; private set; }

    public static GGUF Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);

        uint magic = reader.ReadUInt32();
        if (magic != Magic) throw new Exception("Invalid GGUF magic");

        uint version = reader.ReadUInt32();
        if (version < 2) throw new Exception($"Unsupported GGUF version: {version}");

        long tensorCount = reader.ReadInt64();
        long metadataCount = reader.ReadInt64();

        var gguf = new GGUF
        {
            Version = version,
            TensorCount = tensorCount,
            MetadataCount = metadataCount
        };

        for (int i = 0; i < metadataCount; i++)
        {
            string key = ReadString(reader);
            MetadataValueType type = (MetadataValueType)reader.ReadUInt32();
            object value = ReadMetadataValue(reader, type);
            gguf.Metadata[key] = value;
        }

        for (int i = 0; i < tensorCount; i++)
        {
            string name = ReadString(reader);
            uint n_dims = reader.ReadUInt32();
            long[] dims = new long[n_dims];
            for (int j = 0; j < n_dims; j++) dims[j] = reader.ReadInt64();
            GGMLType type = (GGMLType)reader.ReadUInt32();
            long offset = reader.ReadInt64();
            gguf.TensorInfos[name] = new GGUFTensorInfo(name, dims, type, offset);
        }

        long currentPos = fs.Position;
        uint alignment = gguf.Metadata.TryGetValue("general.alignment", out var alg) ? (uint)alg : 32;
        long padding = (alignment - (currentPos % alignment)) % alignment;
        gguf.TensorDataOffset = currentPos + padding;

        return gguf;
    }

    private static string ReadString(BinaryReader reader)
    {
        long len = reader.ReadInt64();
        byte[] bytes = reader.ReadBytes((int)len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object ReadMetadataValue(BinaryReader reader, MetadataValueType type)
    {
        return type switch
        {
            MetadataValueType.UINT8 => reader.ReadByte(),
            MetadataValueType.INT8 => reader.ReadSByte(),
            MetadataValueType.UINT16 => reader.ReadUInt16(),
            MetadataValueType.INT16 => reader.ReadInt16(),
            MetadataValueType.UINT32 => reader.ReadUInt32(),
            MetadataValueType.INT32 => reader.ReadInt32(),
            MetadataValueType.FLOAT32 => reader.ReadSingle(),
            MetadataValueType.BOOL => reader.ReadByte() != 0,
            MetadataValueType.STRING => ReadString(reader),
            MetadataValueType.ARRAY => ReadArray(reader),
            MetadataValueType.UINT64 => reader.ReadUInt64(),
            MetadataValueType.INT64 => reader.ReadInt64(),
            MetadataValueType.FLOAT64 => reader.ReadDouble(),
            _ => throw new NotSupportedException($"Unsupported metadata type: {type}")
        };
    }

    private static object ReadArray(BinaryReader reader)
    {
        MetadataValueType itemType = (MetadataValueType)reader.ReadUInt32();
        long len = reader.ReadInt64();
        var array = new object[len];
        for (int i = 0; i < len; i++)
        {
            array[i] = ReadMetadataValue(reader, itemType);
        }
        return array;
    }
}
