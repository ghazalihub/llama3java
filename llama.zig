const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const io = std.io;

// --- GGUF & Types ---

pub const GGUF_MAGIC = 0x46554747;

pub const GGMLType = enum(u32) {
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

    pub fn typeSize(self: GGMLType) usize {
        return switch (self) {
            .F32 => 4,
            .F16 => 2,
            .Q4_0 => 18,
            .Q8_0 => 34,
            else => @panic("Unsupported GGML type size"),
        };
    }

    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .F32, .F16 => 1,
            .Q4_0, .Q8_0 => 32,
            else => @panic("Unsupported GGML block size"),
        };
    }

    pub fn byteSizeFor(self: GGMLType, num_elements: usize) usize {
        const bsize = self.blockSize();
        const tsize = self.typeSize();
        std.debug.assert(num_elements % bsize == 0);
        return (num_elements / bsize) * tsize;
    }
};

pub const MetadataValueType = enum(u32) {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

pub const MetadataValue = union(MetadataValueType) {
    UINT8: u8, INT8: i8, UINT16: u16, INT16: i16, UINT32: u32, INT32: i32,
    FLOAT32: f32, BOOL: bool, STRING: []const u8, ARRAY: MetadataArray,
    UINT64: u64, INT64: i64, FLOAT64: f64,
};

pub const MetadataArray = struct {
    type: MetadataValueType,
    len: usize,
    data: []MetadataValue,
};

pub const GGUFTensorInfo = struct {
    name: []const u8,
    dimensions: []u64,
    ggml_type: GGMLType,
    offset: u64,

    pub fn numElements(self: GGUFTensorInfo) u64 {
        var res: u64 = 1;
        for (self.dimensions) |d| res *= d;
        return res;
    }
};

pub const GGUF = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    metadata: std.StringHashMap(MetadataValue),
    tensor_infos: std.StringHashMap(GGUFTensorInfo),
    tensor_data_offset: u64,
    allocator: mem.Allocator,

    pub fn load(path: []const u8, allocator: mem.Allocator) !GGUF {
        const file = try fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();

        const magic = try reader.readInt(u32, .little);
        if (magic != GGUF_MAGIC) return error.InvalidMagic;

        const version = try reader.readInt(u32, .little);
        if (version < 2) return error.UnsupportedVersion;

        const tensor_count = try reader.readInt(u64, .little);
        const metadata_count = try reader.readInt(u64, .little);

        var gguf = GGUF{
            .magic = magic,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_count = metadata_count,
            .metadata = std.StringHashMap(MetadataValue).init(allocator),
            .tensor_infos = std.StringHashMap(GGUFTensorInfo).init(allocator),
            .tensor_data_offset = 0,
            .allocator = allocator,
        };

        for (0..metadata_count) |_| {
            const key = try readString(reader, allocator);
            const val_type = @as(MetadataValueType, @enumFromInt(try reader.readInt(u32, .little)));
            const value = try readMetadataValue(reader, val_type, allocator);
            try gguf.metadata.put(key, value);
        }

        for (0..tensor_count) |_| {
            const name = try readString(reader, allocator);
            const n_dims = try reader.readInt(u32, .little);
            const dims = try allocator.alloc(u64, n_dims);
            for (0..n_dims) |j| dims[j] = try reader.readInt(u64, .little);
            const ggml_type = @as(GGMLType, @enumFromInt(try reader.readInt(u32, .little)));
            const offset = try reader.readInt(u64, .little);
            try gguf.tensor_infos.put(name, .{ .name = name, .dimensions = dims, .ggml_type = ggml_type, .offset = offset });
        }

        const current_pos = try file.getPos();
        const alignment = if (gguf.metadata.get("general.alignment")) |v| v.UINT32 else 32;
        const padding = (alignment - (current_pos % alignment)) % alignment;
        gguf.tensor_data_offset = current_pos + padding;

        return gguf;
    }

    pub fn deinit(self: *GGUF) void {
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            deinitMetadataValue(entry.value_ptr.*, self.allocator);
        }
        self.metadata.deinit();
        var t_it = self.tensor_infos.iterator();
        while (t_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.dimensions);
        }
        self.tensor_infos.deinit();
    }

    fn deinitMetadataValue(val: MetadataValue, allocator: mem.Allocator) void {
        switch (val) {
            .STRING => |s| allocator.free(s),
            .ARRAY => |a| {
                for (a.data) |item| deinitMetadataValue(item, allocator);
                allocator.free(a.data);
            },
            else => {},
        }
    }

    fn readString(reader: anytype, allocator: mem.Allocator) ![]const u8 {
        const len = try reader.readInt(u64, .little);
        const buf = try allocator.alloc(u8, len);
        try reader.readNoEof(buf);
        return buf;
    }

    fn readMetadataValue(reader: anytype, val_type: MetadataValueType, allocator: mem.Allocator) !MetadataValue {
        return switch (val_type) {
            .UINT8 => .{ .UINT8 = try reader.readByte() },
            .INT8 => .{ .INT8 = try reader.readByteSigned() },
            .UINT16 => .{ .UINT16 = try reader.readInt(u16, .little) },
            .INT16 => .{ .INT16 = try reader.readInt(i16, .little) },
            .UINT32 => .{ .UINT32 = try reader.readInt(u32, .little) },
            .INT32 => .{ .INT32 = try reader.readInt(i32, .little) },
            .FLOAT32 => .{ .FLOAT32 = @bitCast(try reader.readInt(u32, .little)) },
            .BOOL => .{ .BOOL = (try reader.readByte()) != 0 },
            .STRING => .{ .STRING = try readString(reader, allocator) },
            .ARRAY => {
                const item_type = @as(MetadataValueType, @enumFromInt(try reader.readInt(u32, .little)));
                const len = try reader.readInt(u64, .little);
                const data = try allocator.alloc(MetadataValue, len);
                for (0..len) |i| data[i] = try readMetadataValue(reader, item_type, allocator);
                return .{ .ARRAY = .{ .type = item_type, .len = len, .data = data } };
            },
            .UINT64 => .{ .UINT64 = try reader.readInt(u64, .little) },
            .INT64 => .{ .INT64 = try reader.readInt(i64, .little) },
            .FLOAT64 => .{ .FLOAT64 = @bitCast(try reader.readInt(u64, .little)) },
        };
    }
};

// --- Math & SIMD ---

pub const Tensor = struct {
    data: []u8,
    ggml_type: GGMLType,
    num_elements: usize,
};

fn dotF32(af: []const f32, bf: []const f32, n: usize) f32 {
    var sum_v: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 32 <= n) : (i += 32) {
        sum_v += @as(@Vector(8, f32), af[i..][0..8].*) * @as(@Vector(8, f32), bf[i..][0..8].*);
        sum_v += @as(@Vector(8, f32), af[i + 8 ..][0..8].*) * @as(@Vector(8, f32), bf[i + 8 ..][0..8].*);
        sum_v += @as(@Vector(8, f32), af[i + 16 ..][0..8].*) * @as(@Vector(8, f32), bf[i + 16 ..][0..8].*);
        sum_v += @as(@Vector(8, f32), af[i + 24 ..][0..8].*) * @as(@Vector(8, f32), bf[i + 24 ..][0..8].*);
    }
    var sum = @reduce(.Add, sum_v);
    while (i < n) : (i += 1) sum += af[i] * bf[i];
    return sum;
}

fn dotQ4_0F32(a: []const u8, bf: []const f32, n: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 32) {
        const block = a[i / 32 * 18 ..];
        const scale = @as(f32, @floatCast(mem.bytesToValue(f16, block[0..2])));
        const quants: @Vector(16, u8) = block[2..18].*;

        const v_lo_u8 = quants & @as(@Vector(16, u8), @splat(0x0F));
        const v_hi_u8 = quants >> @as(@Vector(16, u8), @splat(4));

        const v_lo_i8: @Vector(16, i8) = @bitCast(v_lo_u8);
        const v_hi_i8: @Vector(16, i8) = @bitCast(v_hi_u8);

        const v_lo_f32: @Vector(16, f32) = @floatFromInt(v_lo_i8 - @as(@Vector(16, i8), @splat(8)));
        const v_hi_f32: @Vector(16, f32) = @floatFromInt(v_hi_i8 - @as(@Vector(16, i8), @splat(8)));

        sum += scale * (@reduce(.Add, v_lo_f32 * bf[i..][0..16].*) + @reduce(.Add, v_hi_f32 * bf[i + 16 ..][0..16].*));
    }
    return sum;
}

fn dotQ8_0F32(a: []const u8, bf: []const f32, n: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + 64 <= n) : (i += 64) {
        const b1 = a[i / 32 * 34 ..];
        const s1 = @as(f32, @floatCast(mem.bytesToValue(f16, b1[0..2])));
        const vq1: @Vector(32, i8) = @bitCast(b1[2..34].*);
        sum += s1 * @reduce(.Add, @as(@Vector(32, f32), @floatFromInt(vq1)) * bf[i..][0..32].*);
        const b2 = a[(i + 32) / 32 * 34 ..];
        const s2 = @as(f32, @floatCast(mem.bytesToValue(f16, b2[0..2])));
        const vq2: @Vector(32, i8) = @bitCast(b2[2..34].*);
        sum += s2 * @reduce(.Add, @as(@Vector(32, f32), @floatFromInt(vq2)) * bf[i + 32 ..][0..32].*);
    }
    while (i < n) : (i += 32) {
        const b = a[i / 32 * 34 ..];
        const s = @as(f32, @floatCast(mem.bytesToValue(f16, b[0..2])));
        const vq: @Vector(32, i8) = @bitCast(b[2..34].*);
        sum += s * @reduce(.Add, @as(@Vector(32, f32), @floatFromInt(vq)) * bf[i..][0..32].*);
    }
    return sum;
}

pub fn copyToF32(t: Tensor, offset: usize, out: []f32) void {
    const n = out.len;
    if (t.ggml_type == .F32) {
        const tf = mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(t.data[offset * 4 ..])));
        mem.copyForwards(f32, out, tf[0..n]);
    } else if (t.ggml_type == .Q4_0) {
        var i: usize = 0;
        while (i < n) : (i += 32) {
            const block = t.data[(offset + i) / 32 * 18 ..];
            const scale = @as(f32, @floatCast(mem.bytesToValue(f16, block[0..2])));
            for (0..16) |j| {
                out[i + j] = scale * @as(f32, @floatFromInt(@as(i8, @intCast(block[2 + j] & 0xF)) - 8));
                out[i + j + 16] = scale * @as(f32, @floatFromInt(@as(i8, @intCast(block[2 + j] >> 4)) - 8));
            }
        }
    } else if (t.ggml_type == .Q8_0) {
        var i: usize = 0;
        while (i < n) : (i += 32) {
            const block = t.data[(offset + i) / 32 * 34 ..];
            const scale = @as(f32, @floatCast(mem.bytesToValue(f16, block[0..2])));
            for (0..32) |j| {
                out[i + j] = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j]))));
            }
        }
    } else @panic("Unsupported quantization for embedding lookup");
}

pub fn dot(a: Tensor, a_off: usize, b: Tensor, b_off: usize, n: usize) f32 {
    if (a.ggml_type == .F32 and b.ggml_type == .F32) {
        const af = mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(a.data[a_off * 4 ..])));
        const bf = mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b.data[b_off * 4 ..])));
        return dotF32(af, bf, n);
    } else if (a.ggml_type == .Q4_0 and b.ggml_type == .F32) {
        const bf = mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b.data[b_off * 4 ..])));
        return dotQ4_0F32(a.data[a_off / 32 * 18 ..], bf, n);
    } else if (a.ggml_type == .Q8_0 and b.ggml_type == .F32) {
        const bf = mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b.data[b_off * 4 ..])));
        return dotQ8_0F32(a.data[a_off / 32 * 34 ..], bf, n);
    }
    @panic("Unsupported dot product types");
}

pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    ss /= @floatFromInt(x.len);
    ss += eps;
    const inv_std = 1.0 / @sqrt(ss);
    for (0..x.len) |i| out[i] = weight[i] * (x[i] * inv_std);
}

pub fn softmax(x: []f32) void {
    var max_val = x[0];
    for (x) |v| if (v > max_val) { max_val = v; };
    var sum: f32 = 0;
    for (0..x.len) |i| {
        x[i] = @exp(x[i] - max_val);
        sum += x[i];
    }
    for (0..x.len) |i| x[i] /= sum;
}

const MatMulContext = struct {
    out: []f32, x: Tensor, w: Tensor, n: usize, d: usize, start: usize, end: usize,
};

fn matmulChunk(ctx: MatMulContext) void {
    for (ctx.start..ctx.end) |i| ctx.out[i] = dot(ctx.w, i * ctx.d, ctx.x, 0, ctx.d);
}

fn matmulChunkTask(ctx: MatMulContext, wg: *std.Thread.WaitGroup) void {
    defer wg.finish();
    matmulChunk(ctx);
}

pub fn matmul(out: []f32, x: Tensor, w: Tensor, n: usize, d: usize, pool: ?*std.Thread.Pool) void {
    if (pool) |p| {
        const num_threads = p.threads.len + 1;
        const chunk_size = (n + num_threads - 1) / num_threads;
        var wg = std.Thread.WaitGroup{};
        var start: usize = 0;
        while (start < n) {
            const end = @min(start + chunk_size, n);
            wg.start();
            p.spawn(matmulChunkTask, .{ .{ .out = out, .x = x, .w = w, .n = n, .d = d, .start = start, .end = end }, &wg }) catch {
                matmulChunk(.{ .out = out, .x = x, .w = w, .n = n, .d = d, .start = start, .end = end });
                wg.finish();
            };
            start = end;
        }
        wg.wait();
    } else {
        matmulChunk(.{ .out = out, .x = x, .w = w, .n = n, .d = d, .start = 0, .end = n });
    }
}

// --- Tokenizer ---

const ByteUnicodeMap = struct {
    b2u: [256]u32, u2b: std.AutoHashMap(u32, u8),
    pub fn init(allocator: mem.Allocator) !ByteUnicodeMap {
        var b2u: [256]u32 = undefined;
        var u2b = std.AutoHashMap(u32, u8).init(allocator);
        var bs = std.ArrayList(u8).init(allocator);
        defer bs.deinit();
        for ('!'..'~' + 1) |i| try bs.append(@intCast(i));
        for (0xA1..0xAC + 1) |i| try bs.append(@intCast(i));
        for (0xAE..0xFF + 1) |i| try bs.append(@intCast(i));
        var cs = std.ArrayList(u32).init(allocator);
        defer cs.deinit();
        for (bs.items) |b| try cs.append(b);
        var n: u32 = 0;
        for (0..256) |b_int| {
            const b: u8 = @intCast(b_int);
            var found = false;
            for (bs.items) |eb| if (eb == b) { found = true; break; };
            if (!found) { try bs.append(b); try cs.append(256 + n); n += 1; }
        }
        for (0..256) |i| { b2u[bs.items[i]] = cs.items[i]; try u2b.put(cs.items[i], bs.items[i]); }
        return ByteUnicodeMap{ .b2u = b2u, .u2b = u2b };
    }
    pub fn deinit(self: *ByteUnicodeMap) void { self.u2b.deinit(); }
};

fn allocTokenFromByte(b: u8, b2u: *const ByteUnicodeMap, allocator: mem.Allocator) ![]const u8 {
    const code = b2u.b2u[b];
    if (code < 128) return std.fmt.allocPrint(allocator, "{c}", .{@as(u8, @intCast(code))});
    var buf: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(@intCast(code), &buf);
    return allocator.dupe(u8, buf[0..len]);
}

pub const Vocabulary = struct {
    tokens: [][]const u8, token_to_index: std.StringHashMap(u32), allocator: mem.Allocator,
    pub fn init(tokens: [][]const u8, allocator: mem.Allocator) !Vocabulary {
        var t2i = std.StringHashMap(u32).init(allocator);
        for (tokens, 0..) |t, i| try t2i.put(t, @intCast(i));
        return Vocabulary{ .tokens = tokens, .token_to_index = t2i, .allocator = allocator };
    }
    pub fn deinit(self: *Vocabulary) void { self.token_to_index.deinit(); }
    pub fn getIndex(self: *const Vocabulary, token: []const u8) ?u32 { return self.token_to_index.get(token); }
};

pub const Tokenizer = struct {
    vocab: Vocabulary, merges: std.AutoHashMap([2]u32, u32), special_tokens: std.StringHashMap(u32), allocator: mem.Allocator,
    pub fn init(vocab: Vocabulary, merges_raw: []const []const u8, special_tokens: std.StringHashMap(u32), allocator: mem.Allocator) !Tokenizer {
        var merges = std.AutoHashMap([2]u32, u32).init(allocator);
        for (merges_raw) |line| {
            var it = mem.splitScalar(u8, line, ' ');
            const s1 = it.next() orelse continue;
            const s2 = it.next() orelse continue;
            const id1 = vocab.getIndex(s1) orelse continue;
            const id2 = vocab.getIndex(s2) orelse continue;
            const combined = try std.fmt.allocPrint(allocator, "{s}{s}", .{ s1, s2 });
            defer allocator.free(combined);
            if (vocab.getIndex(combined)) |cid| try merges.put(.{ id1, id2 }, cid);
        }
        return Tokenizer{ .vocab = vocab, .merges = merges, .special_tokens = special_tokens, .allocator = allocator };
    }
    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit();
        var it = self.special_tokens.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.special_tokens.deinit();
    }
    pub fn encode(self: *const Tokenizer, text: []const u8, allowed_special: []const []const u8) ![]u32 {
        for (allowed_special) |s| if (mem.eql(u8, text, s)) if (self.special_tokens.get(s)) |id| {
            const res = try self.allocator.alloc(u32, 1); res[0] = id; return res;
        };
        var b2u = try ByteUnicodeMap.init(self.allocator); defer b2u.deinit();
        var ids = std.ArrayList(u32).init(self.allocator); defer ids.deinit();
        for (text) |b| {
            const t = try allocTokenFromByte(b, &b2u, self.allocator); defer self.allocator.free(t);
            if (self.vocab.getIndex(t)) |id| try ids.append(id);
        }
        while (ids.items.len >= 2) {
            var best_pair: ?[2]u32 = null; var best_idx: u32 = std.math.maxInt(u32);
            for (0..ids.items.len - 1) |i| {
                const pair = [2]u32{ ids.items[i], ids.items[i + 1] };
                if (self.merges.get(pair)) |mid| if (mid < best_idx) { best_idx = mid; best_pair = pair; };
            }
            if (best_pair) |p| {
                var new_ids = std.ArrayList(u32).init(self.allocator);
                var i: usize = 0;
                while (i < ids.items.len) {
                    if (i < ids.items.len - 1 and ids.items[i] == p[0] and ids.items[i + 1] == p[1]) {
                        try new_ids.append(best_idx); i += 2;
                    } else { try new_ids.append(ids.items[i]); i += 1; }
                }
                ids.deinit(); ids = new_ids;
            } else break;
        }
        return ids.toOwnedSlice();
    }
    pub fn decode(self: *const Tokenizer, tokens: []const u32) ![]const u8 {
        var b2u = try ByteUnicodeMap.init(self.allocator);
        defer b2u.deinit();
        var res = std.ArrayList(u8).init(self.allocator);
        for (tokens) |t| {
            const s = self.vocab.tokens[t];
            var view = try std.unicode.Utf8View.init(s);
            var it = view.iterator();
            while (it.nextCodepoint()) |cp| {
                if (b2u.u2b.get(cp)) |b| {
                    try res.append(b);
                } else {
                    // If not in map, it's a regular UTF-8 character (like for special tokens)
                    var buf: [4]u8 = undefined;
                    const len = try std.unicode.utf8Encode(cp, &buf);
                    try res.appendSlice(buf[0..len]);
                }
            }
        }
        return res.toOwnedSlice();
    }
};

// --- Transformer ---

pub const Config = struct {
    dim: usize, hidden_dim: usize, n_layers: usize, n_heads: usize, n_kv_heads: usize,
    vocab_size: usize, seq_len: usize, rope_theta: f32, eps: f32, head_size: usize,
    rope_scaling: bool, scale_factor: f32, lo_freq_factor: f32, hi_freq_factor: f32, old_context_len: f32,

    pub fn fromGGUF(gguf: GGUF) !Config {
        const dim = gguf.metadata.get("llama.embedding_length").?.UINT32;
        const hdim = gguf.metadata.get("llama.feed_forward_length").?.UINT32;
        const nlay = gguf.metadata.get("llama.block_count").?.UINT32;
        const nhead = gguf.metadata.get("llama.attention.head_count").?.UINT32;
        const nkvhead = if (gguf.metadata.get("llama.attention.head_count_kv")) |v| v.UINT32 else nhead;
        const v_meta = gguf.metadata.get("tokenizer.ggml.tokens").?.ARRAY;
        const slen = gguf.metadata.get("llama.context_length").?.UINT32;
        const rtheta = if (gguf.metadata.get("llama.rope.freq_base")) |v| v.FLOAT32 else 10000.0;
        const eps = if (gguf.metadata.get("llama.attention.layer_norm_rms_epsilon")) |v| v.FLOAT32 else 1e-5;

        var rope_scaling = false;
        var scale_factor: f32 = 8.0;
        var lo_freq_factor: f32 = 1.0;
        var hi_freq_factor: f32 = 3.0;
        var old_ctx_len: f32 = 8192.0;

        if (gguf.metadata.get("llama.rope.scaling.type")) |v| {
            if (mem.eql(u8, v.STRING, "linear") or mem.eql(u8, v.STRING, "yarn")) {
                rope_scaling = true;
            }
        }
        if (gguf.metadata.get("llama.rope.scaling.factor")) |v| scale_factor = v.FLOAT32;
        if (gguf.metadata.get("llama.rope.scaling.low_freq_factor")) |v| lo_freq_factor = v.FLOAT32;
        if (gguf.metadata.get("llama.rope.scaling.high_freq_factor")) |v| hi_freq_factor = v.FLOAT32;
        if (gguf.metadata.get("llama.rope.scaling.orig_ctx_len")) |v| old_ctx_len = @floatFromInt(v.UINT32);

        return Config{
            .dim = @intCast(dim), .hidden_dim = @intCast(hdim), .n_layers = @intCast(nlay), .n_heads = @intCast(nhead), .n_kv_heads = @intCast(nkvhead),
            .vocab_size = v_meta.len, .seq_len = @intCast(slen), .rope_theta = rtheta, .eps = eps, .head_size = @intCast(dim / nhead),
            .rope_scaling = rope_scaling, .scale_factor = scale_factor, .lo_freq_factor = lo_freq_factor, .hi_freq_factor = hi_freq_factor, .old_context_len = old_ctx_len,
        };
    }
};

pub const Weights = struct {
    token_embd: Tensor, rms_att_w: []Tensor, wq: []Tensor, wk: []Tensor, wv: []Tensor, wo: []Tensor,
    rms_ffn_w: []Tensor, w1: []Tensor, w2: []Tensor, w3: []Tensor, rms_final_w: Tensor, output_w: Tensor,
    pub fn load(config: Config, gguf: GGUF, mmap_data: []u8, allocator: mem.Allocator) !Weights {
        var w: Weights = undefined;
        w.token_embd = try getTensor(gguf, mmap_data, "token_embd.weight");
        w.rms_att_w = try allocator.alloc(Tensor, config.n_layers);
        w.wq = try allocator.alloc(Tensor, config.n_layers);
        w.wk = try allocator.alloc(Tensor, config.n_layers);
        w.wv = try allocator.alloc(Tensor, config.n_layers);
        w.wo = try allocator.alloc(Tensor, config.n_layers);
        w.rms_ffn_w = try allocator.alloc(Tensor, config.n_layers);
        w.w1 = try allocator.alloc(Tensor, config.n_layers);
        w.w2 = try allocator.alloc(Tensor, config.n_layers);
        w.w3 = try allocator.alloc(Tensor, config.n_layers);
        for (0..config.n_layers) |i| {
            var buf: [64]u8 = undefined;
            w.rms_att_w[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{i}));
            w.wq[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{i}));
            w.wk[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{i}));
            w.wv[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{i}));
            w.wo[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{i}));
            w.rms_ffn_w[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{i}));
            w.w1[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{i}));
            w.w2[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{i}));
            w.w3[i] = try getTensor(gguf, mmap_data, try std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{i}));
        }
        w.rms_final_w = try getTensor(gguf, mmap_data, "output_norm.weight");
        w.output_w = getTensor(gguf, mmap_data, "output.weight") catch w.token_embd;
        return w;
    }
    fn getTensor(gguf: GGUF, mmap_data: []u8, name: []const u8) !Tensor {
        const info = gguf.tensor_infos.get(name) orelse return error.TensorNotFound;
        const off = gguf.tensor_data_offset + info.offset;
        const size = info.ggml_type.byteSizeFor(@intCast(info.numElements()));
        return Tensor{ .data = mmap_data[off .. off + size], .ggml_type = info.ggml_type, .num_elements = @intCast(info.numElements()) };
    }
};

pub const State = struct {
    x: []f32, xb: []f32, xb2: []f32, hb: []f32, hb2: []f32, q: []f32, k: []f32, v: []f32, att: []f32,
    logits: []f32, key_cache: []f32, value_cache: []f32,
    pub fn init(config: Config, allocator: mem.Allocator) !State {
        const kv_dim = config.n_kv_heads * config.head_size;
        return State{
            .x = try allocator.alloc(f32, config.dim), .xb = try allocator.alloc(f32, config.dim), .xb2 = try allocator.alloc(f32, config.dim),
            .hb = try allocator.alloc(f32, config.hidden_dim), .hb2 = try allocator.alloc(f32, config.hidden_dim),
            .q = try allocator.alloc(f32, config.dim), .k = try allocator.alloc(f32, config.dim), .v = try allocator.alloc(f32, config.dim),
            .att = try allocator.alloc(f32, config.n_heads * config.seq_len), .logits = try allocator.alloc(f32, config.vocab_size),
            .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
            .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
        };
    }
    pub fn deinit(self: *State, allocator: mem.Allocator) void {
        allocator.free(self.x); allocator.free(self.xb); allocator.free(self.xb2); allocator.free(self.hb); allocator.free(self.hb2);
        allocator.free(self.q); allocator.free(self.k); allocator.free(self.v); allocator.free(self.att); allocator.free(self.logits);
        allocator.free(self.key_cache); allocator.free(self.value_cache);
    }
};

pub fn forward(config: Config, state: *State, weights: Weights, token: u32, pos: usize, pool: ?*std.Thread.Pool) void {
    const dim = config.dim; const hdim = config.hidden_dim; const hsz = config.head_size;
    const nhead = config.n_heads; const nkv = config.n_kv_heads; const kv_dim = nkv * hsz; const kv_mul = nhead / nkv;
    copyToF32(weights.token_embd, token * dim, state.x);
    for (0..config.n_layers) |l| {
        rmsnorm(state.xb, state.x, mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(weights.rms_att_w[l].data))), config.eps);
        const x_tensor = Tensor{ .data = mem.sliceAsBytes(state.xb), .ggml_type = .F32, .num_elements = dim };
        matmul(state.q, x_tensor, weights.wq[l], nhead * hsz, dim, pool);
        matmul(state.k, x_tensor, weights.wk[l], nkv * hsz, dim, pool);
        matmul(state.v, x_tensor, weights.wv[l], nkv * hsz, dim, pool);
        for (0..nhead) |h| {
            for (0..hsz / 2) |i| {
                var freq = 1.0 / std.math.pow(f32, config.rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(hsz)));
                if (config.rope_scaling) {
                    const wlen = 2.0 * std.math.pi / freq;
                    const low = config.old_context_len / config.lo_freq_factor;
                    const high = config.old_context_len / config.hi_freq_factor;
                    if (wlen > low) freq /= config.scale_factor else if (wlen > high) {
                        const smooth = (config.old_context_len / wlen - config.lo_freq_factor) / (config.hi_freq_factor - config.lo_freq_factor);
                        freq = (1.0 - smooth) * freq / config.scale_factor + smooth * freq;
                    }
                }
                const val = @as(f32, @floatFromInt(pos)) * freq; const fcr = @cos(val); const fci = @sin(val);
                const q0 = state.q[h * hsz + i * 2]; const q1 = state.q[h * hsz + i * 2 + 1];
                state.q[h * hsz + i * 2] = q0 * fcr - q1 * fci; state.q[h * hsz + i * 2 + 1] = q0 * fci + q1 * fcr;
                if (h < nkv) {
                    const k0 = state.k[h * hsz + i * 2]; const k1 = state.k[h * hsz + i * 2 + 1];
                    state.k[h * hsz + i * 2] = k0 * fcr - k1 * fci; state.k[h * hsz + i * 2 + 1] = k0 * fci + k1 * fcr;
                }
            }
        }
        const loff = l * config.seq_len * kv_dim;
        mem.copyForwards(f32, state.key_cache[loff + pos * kv_dim ..][0..kv_dim], state.k[0..kv_dim]);
        mem.copyForwards(f32, state.value_cache[loff + pos * kv_dim ..][0..kv_dim], state.v[0..kv_dim]);
        for (0..nhead) |h| {
            const head_q = state.q[h * hsz .. (h + 1) * hsz];
            const att_h = state.att[h * config.seq_len .. (h + 1) * config.seq_len];
            for (0..pos + 1) |t| {
                const head_k = state.key_cache[loff + t * kv_dim + (h / kv_mul) * hsz ..][0..hsz];
                att_h[t] = dotF32(head_q, head_k, hsz) / @sqrt(@as(f32, @floatFromInt(hsz)));
            }
            softmax(att_h[0 .. pos + 1]);
            const head_xb = state.xb[h * hsz .. (h + 1) * hsz]; @memset(head_xb, 0);
            for (0..pos + 1) |t| {
                const head_v = state.value_cache[loff + t * kv_dim + (h / kv_mul) * hsz ..][0..hsz];
                const a = att_h[t];
                var i: usize = 0; while (i + 8 <= hsz) : (i += 8) {
                    const v_xb: @Vector(8, f32) = head_xb[i..][0..8].*;
                    const v_v: @Vector(8, f32) = head_v[i..][0..8].*;
                    head_xb[i..][0..8].* = v_xb + @as(@Vector(8, f32), @splat(a)) * v_v;
                }
                while (i < hsz) : (i += 1) head_xb[i] += a * head_v[i];
            }
        }
        const xb_tensor = Tensor{ .data = mem.sliceAsBytes(state.xb), .ggml_type = .F32, .num_elements = dim };
        matmul(state.xb2, xb_tensor, weights.wo[l], dim, dim, pool);
        for (0..dim) |i| state.x[i] += state.xb2[i];
        rmsnorm(state.xb, state.x, mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(weights.rms_ffn_w[l].data))), config.eps);
        matmul(state.hb, xb_tensor, weights.w1[l], hdim, dim, pool);
        matmul(state.hb2, xb_tensor, weights.w3[l], hdim, dim, pool);
        for (0..hdim) |i| { var val = state.hb[i]; val *= 1.0 / (1.0 + @exp(-val)); state.hb[i] = val * state.hb2[i]; }
        const hb_tensor = Tensor{ .data = mem.sliceAsBytes(state.hb), .ggml_type = .F32, .num_elements = hdim };
        matmul(state.xb, hb_tensor, weights.w2[l], dim, hdim, pool);
        for (0..dim) |i| state.x[i] += state.xb[i];
    }
    rmsnorm(state.x, state.x, mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(weights.rms_final_w.data))), config.eps);
    const x_final = Tensor{ .data = mem.sliceAsBytes(state.x), .ggml_type = .F32, .num_elements = dim };
    matmul(state.logits, x_final, weights.output_w, config.vocab_size, dim, pool);
}

// --- Sampling & Generation ---

pub const ProbIndex = struct {
    prob: f32, index: u32,
    fn compare(_: void, a: ProbIndex, b: ProbIndex) bool { return a.prob > b.prob; }
};

pub fn sample(logits: []f32, temp: f32, topp: f32, rng: *std.Random.DefaultPrng) u32 {
    if (temp == 0) {
        var midx: u32 = 0; var mval = logits[0];
        for (logits, 0..) |v, i| if (v > mval) { mval = v; midx = @intCast(i); };
        return midx;
    }
    for (logits) |*v| v.* /= temp;
    softmax(logits);
    if (topp <= 0 or topp >= 1) {
        const r = rng.random().float(f32); var cdf: f32 = 0;
        for (logits, 0..) |p, i| { cdf += p; if (r < cdf) return @intCast(i); }
        return @intCast(logits.len - 1);
    }

    var pi = std.ArrayList(ProbIndex).init(std.heap.page_allocator); defer pi.deinit();
    // Only add tokens with probability > 0 (or a small threshold) to speed up sorting
    const threshold = (1.0 - topp) / @as(f32, @floatFromInt(logits.len));
    for (logits, 0..) |p, i| {
        if (p > threshold) pi.append(.{ .prob = p, .index = @intCast(i) }) catch unreachable;
    }

    if (pi.items.len == 0) return argmax(logits);

    std.sort.pdq(ProbIndex, pi.items, {}, ProbIndex.compare);
    var cprob: f32 = 0; var lidx: usize = pi.items.len - 1;
    for (pi.items, 0..) |p, i| { cprob += p.prob; if (cprob > topp) { lidx = i; break; } }
    const r = rng.random().float(f32) * cprob; var cdf: f32 = 0;
    for (pi.items[0 .. lidx + 1]) |p| { cdf += p.prob; if (r < cdf) return p.index; }
    return pi.items[lidx].index;
}

fn argmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val = logits[0];
    for (logits, 0..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

pub fn generate(config: Config, state: *State, weights: Weights, tokenizer: Tokenizer, prompt: []const u32, max_tok: usize, temp: f32, topp: f32, rng: *std.Random.DefaultPrng, allocator: mem.Allocator, pool: ?*std.Thread.Pool, stream: bool, echo: bool) !void {
    var next: u32 = 0; var pos: usize = 0; const start = std.time.milliTimestamp();
    for (prompt) |t| {
        forward(config, state, weights, t, pos, pool);
        if (echo) {
            const s = try tokenizer.decode(&.{t}); defer allocator.free(s);
            std.debug.print("{s}", .{s});
        }
        pos += 1; next = t;
    }
    while (pos < max_tok) {
        forward(config, state, weights, next, pos, pool);
        next = sample(state.logits, temp, topp, rng);
        if (tokenizer.special_tokens.get("<|end_of_text|>")) |eos| if (next == eos) break;
        if (tokenizer.special_tokens.get("<|eot_id|>")) |eot| if (next == eot) break;
        if (stream) {
            const s = try tokenizer.decode(&.{next}); defer allocator.free(s);
            std.debug.print("{s}", .{s});
        }
        pos += 1; if (pos >= config.seq_len) break;
    }
    if (!stream and !echo) {
        // TODO: decode and print full response if not streaming
    }
    const end = std.time.milliTimestamp(); const elap = @as(f32, @floatFromInt(end - start)) / 1000.0;
    std.debug.print("\n\n{d} tokens in {d:.2}s ({d:.2} tok/s)\n", .{ pos, elap, @as(f32, @floatFromInt(pos)) / elap });
}

// --- Main ---

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator); defer std.process.argsFree(allocator, args);

    var mpath: ?[]const u8 = null; var ptext: ?[]const u8 = null; var temp: f32 = 0.1; var topp: f32 = 0.9; var threads: ?usize = null;
    var seed: u64 = @intCast(std.time.timestamp()); var stream: bool = true; var echo: bool = false; var chat: bool = false;
    var i: usize = 1; while (i < args.len) : (i += 1) {
        if (mem.eql(u8, args[i], "-m") or mem.eql(u8, args[i], "--model")) { i += 1; mpath = args[i]; }
        else if (mem.eql(u8, args[i], "-p") or mem.eql(u8, args[i], "--prompt")) { i += 1; ptext = args[i]; }
        else if (mem.eql(u8, args[i], "--temp")) { i += 1; temp = try std.fmt.parseFloat(f32, args[i]); }
        else if (mem.eql(u8, args[i], "--top-p")) { i += 1; topp = try std.fmt.parseFloat(f32, args[i]); }
        else if (mem.eql(u8, args[i], "-t") or mem.eql(u8, args[i], "--threads")) { i += 1; threads = try std.fmt.parseInt(usize, args[i], 10); }
        else if (mem.eql(u8, args[i], "--seed")) { i += 1; seed = try std.fmt.parseInt(u64, args[i], 10); }
        else if (mem.eql(u8, args[i], "--stream")) { i += 1; stream = mem.eql(u8, args[i], "true"); }
        else if (mem.eql(u8, args[i], "--echo")) { i += 1; echo = mem.eql(u8, args[i], "true"); }
        else if (mem.eql(u8, args[i], "-i") or mem.eql(u8, args[i], "--chat")) { chat = true; }
    }
    if (mpath == null or (ptext == null and !chat)) { std.debug.print("Usage: llama --model <path> [--prompt <str>] [--chat] [--temp <f32>] [--top-p <f32>] [--threads <int>] [--seed <int>] [--stream true|false] [--echo true|false]\n", .{}); return; }

    var pool: ?std.Thread.Pool = null;
    if (threads) |t| {
        pool = .{ .allocator = allocator, .threads = undefined };
        try pool.?.init(.{ .allocator = allocator, .n_jobs = @intCast(t) });
    }
    defer if (pool) |*p| p.deinit();

    var gguf = try GGUF.load(mpath.?, allocator); defer gguf.deinit();
    const config = try Config.fromGGUF(gguf);
    const file = try fs.cwd().openFile(mpath.?, .{}); defer file.close();
    const fsize = try file.getEndPos();
    const mdata = try std.posix.mmap(null, fsize, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0); defer std.posix.munmap(mdata);
    const weights = try Weights.load(config, gguf, mdata, allocator);
    defer {
        allocator.free(weights.rms_att_w); allocator.free(weights.wq); allocator.free(weights.wk); allocator.free(weights.wv); allocator.free(weights.wo);
        allocator.free(weights.rms_ffn_w); allocator.free(weights.w1); allocator.free(weights.w2); allocator.free(weights.w3);
    }
    const vt_meta = gguf.metadata.get("tokenizer.ggml.tokens").?.ARRAY;
    var tokens = try allocator.alloc([]const u8, vt_meta.len); defer allocator.free(tokens);
    for (vt_meta.data, 0..) |v, idx| tokens[idx] = v.STRING;
    var vocab = try Vocabulary.init(tokens, allocator); defer vocab.deinit();
    const m_meta = gguf.metadata.get("tokenizer.ggml.merges").?.ARRAY;
    var mraw = try allocator.alloc([]const u8, m_meta.len); defer allocator.free(mraw);
    for (m_meta.data, 0..) |v, idx| mraw[idx] = v.STRING;
    var st = std.StringHashMap(u32).init(allocator);
    if (vocab.getIndex("<|begin_of_text|>")) |id| try st.put(try allocator.dupe(u8, "<|begin_of_text|>"), id);
    if (vocab.getIndex("<|end_of_text|>")) |id| try st.put(try allocator.dupe(u8, "<|end_of_text|>"), id);
    if (vocab.getIndex("<|eot_id|>")) |id| try st.put(try allocator.dupe(u8, "<|eot_id|>"), id);
    var tokenizer = try Tokenizer.init(vocab, mraw, st, allocator); defer tokenizer.deinit();
    var state = try State.init(config, allocator); defer state.deinit(allocator);
    var rng = std.Random.DefaultPrng.init(seed);
    if (chat) {
        try runChat(config, &state, weights, tokenizer, temp, topp, &rng, allocator, if (pool) |*p| p else null, stream, echo);
    } else {
        std.debug.print("Encoding prompt...\n", .{});
        const ptokens = try tokenizer.encode(ptext.?, &.{}); defer allocator.free(ptokens);
        try generate(config, &state, weights, tokenizer, ptokens, config.seq_len, temp, topp, &rng, allocator, if (pool) |*p| p else null, stream, echo);
    }
}

pub fn runChat(config: Config, state: *State, weights: Weights, tokenizer: Tokenizer, temp: f32, topp: f32, rng: *std.Random.DefaultPrng, allocator: mem.Allocator, pool: ?*std.Thread.Pool, stream: bool, echo: bool) !void {
    _ = echo;
    var pos: usize = 0;
    var next: u32 = tokenizer.special_tokens.get("<|begin_of_text|>") orelse 128000;
    const stdin = std.io.getStdIn().reader();
    var buf: [4096]u8 = undefined;
    while (true) {
        std.debug.print("\n> ", .{});
        const line = try stdin.readUntilDelimiterOrEof(&buf, '\n') orelse break;
        if (mem.eql(u8, line, "quit") or mem.eql(u8, line, "exit")) break;
        const user_prompt = try std.fmt.allocPrint(allocator, "<|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", .{line});
        defer allocator.free(user_prompt);
        const ptokens = try tokenizer.encode(user_prompt, &.{ "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>" });
        defer allocator.free(ptokens);
        for (ptokens) |t| { forward(config, state, weights, next, pos, pool); pos += 1; next = t; }
        while (pos < config.seq_len) {
            forward(config, state, weights, next, pos, pool);
            next = sample(state.logits, temp, topp, rng);
            if (tokenizer.special_tokens.get("<|eot_id|>")) |eot| if (next == eot) break;
            if (tokenizer.special_tokens.get("<|end_of_text|>")) |eos| if (next == eos) break;
            if (stream) { const s = try tokenizer.decode(&.{next}); defer allocator.free(s); std.debug.print("{s}", .{s}); }
            pos += 1;
        }
        std.debug.print("\n", .{});
    }
}

test "GGUF parsing" {
    const allocator = std.testing.allocator; const test_path = "test_model.gguf";
    const file = try fs.cwd().createFile(test_path, .{}); defer fs.cwd().deleteFile(test_path) catch {};
    var writer = file.writer();
    try writer.writeInt(u32, GGUF_MAGIC, .little); try writer.writeInt(u32, 3, .little);
    try writer.writeInt(u64, 1, .little); try writer.writeInt(u64, 1, .little);
    const key = "general.alignment"; try writer.writeInt(u64, key.len, .little); try writer.writeAll(key);
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.UINT32), .little); try writer.writeInt(u32, 32, .little);
    const t_name = "token_embd.weight"; try writer.writeInt(u64, t_name.len, .little); try writer.writeAll(t_name);
    try writer.writeInt(u32, 1, .little); try writer.writeInt(u64, 128, .little);
    try writer.writeInt(u32, @intFromEnum(GGMLType.F32), .little); try writer.writeInt(u64, 0, .little);
    file.close();
    var gguf = try GGUF.load(test_path, allocator); defer gguf.deinit();
    try std.testing.expectEqual(@as(u32, GGUF_MAGIC), gguf.magic);
}
