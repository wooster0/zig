//! Addressable memory where each byte has an address.

const std = @import("std");
const builtin = @import("builtin");
const debug = std.debug;
const panic = debug.panic;
const assert = debug.assert;
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const log = std.log.scoped(.codegen);
const Module = @import("../../Module.zig");
const Decl = Module.Decl;
const link = @import("../../link.zig");
const Type = @import("../../type.zig").Type;
const Value = @import("../../value.zig").Value;
const Air = @import("../../Air.zig");
const Liveness = @import("../../Liveness.zig");
const codegen = @import("../../codegen.zig");
const GenerateSymbolError = codegen.GenerateSymbolError;
const Result = codegen.Result;
const DebugInfoOutput = codegen.DebugInfoOutput;
const bits = @import("bits.zig");
const abi = @import("abi.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");
const Func = @import("Func.zig");

const AddrMem = @This();

/// Memory addresses in the first page of addressable memory available for storage.
/// This is expensive, sparsely available memory and very valuable because
/// writing and reading from it is faster than for absolute memory.
/// We treat each address separately, rather than using an offset, for ideal zero page
/// usage on any system.
/// For example, this handles the case of a system where byte #0 is usable,
/// byte #1 is unusable (reserved by the system),
/// and all other bytes after that of the zero page are usable.
/// If we used offsets, this means our offset would start
/// at byte #2, giving us one less byte than with this method of zero page allocation.
/// The zero page is also the only way to dereference runtime pointers.
zp_free: std.BoundedArray(u8, 256),
/// The offset at which free absolute memory starts.
/// Must not start in zero page. Must be contiguously free memory.
/// This can be called a software stack because it grows downwards
/// and emulates a stack similar to the hardware stack but is typically much bigger than 256 bytes.
abs_offset: u16,
// TODO: record all alloc() calls and where they came from (src_loc) and then after we know the total program size,
//       if any recorded alloc() has abs_offset <= program_size,
//       report an error for that alloc that crashed the stack into the program
//
//       alternatively, associate each alloc with an AIR instruction and track it that way?
///// The absolute memory address at which memory ends.
//abs_end: u16,

/// An address space.
pub const AddrSpace = enum { any, zp };
/// An address in a specific address space.
pub const Addr = union(enum) { zp: u8, abs: u16 };

pub fn init(target: std.Target, bin_file: *link.File) AddrMem {
    if (bin_file.cast(link.File.Prg)) |prg| {
        if (prg.zp_free != null and prg.abs_offset != null) {
            // TODO: encapsulate into MemoryState or something and or require the linker to have a
            //       `6502_mem_state: *anyopaque` field or similar for you.
            //       or we introduce some other general concept to all linkers for storing state across function codegens
            return .{
                .zp_free = prg.zp_free.?,
                .abs_offset = prg.abs_offset.?,
            };
        }
    } else unreachable;

    const zp_free = abi.getZeroPageAddresses(target);
    const abs_offset = abi.getAbsoluteMemoryOffset(target);
    return AddrMem{
        .zp_free = zp_free,
        .abs_offset = abs_offset,
    };
}

/// Returns whether all addressable memory is deallocated.
/// This can be used to detect memory leaks.
/// Pass the same `std.Target` as was passed to `init()`.
// TODO: run this after all function codegens finished to detect leaks,
//       possibly in flushModule (pass a fn ptr as part of the encapsulated mem state?).
pub fn isDeallocated(addr_mem: AddrMem, target: std.Target) bool {
    // Compare with the initial state.
    const init_zp_free = abi.getZeroPageAddresses(target);
    const init_abs_offset = abi.getAbsoluteMemoryOffset(target);
    for (addr_mem.zp_free.constSlice()) |addr, i|
        if (addr != init_zp_free.constSlice()[i]) return false;
    if (addr_mem.abs_offset != init_abs_offset) return false;
    return true;
}

/// Returns the total amount of bytes allocatable.
pub fn getMax(addr_mem: AddrMem, target: std.Target) u16 {
    _ = addr_mem;
    return @intCast(u8, abi.getZeroPageAddresses(target).len) +
        (abi.getAbsoluteMemoryOffset(target)
    // TODO: - addr_mem.abs_end;
    );
}

/// Preserves the current memory state for the next function's codegen.
pub fn preserve(addr_mem: *AddrMem) void {
    // Here we're tackling a problem unique to the 6502 of all backends:
    // while the other codegen backends use the hardware stack and their registers for storing everything
    // and that works fine for them because their stack is much bigger,
    // we use our hardware stack of 256 bytes only for function calling and preserving registers.
    // We use the software stack in form of zero page + absolute memory for everything else.
    // That memory is global rather than stack-local so we have to preserve the memory state
    // across function codegens.
    // The next function can simply continue with this memory state.
    const func = @fieldParentPtr(Func, "addr_mem", addr_mem);
    if (func.bin_file.cast(link.File.Prg)) |prg| {
        prg.abs_offset = addr_mem.abs_offset;
        prg.zp_free = addr_mem.zp_free;
    } else unreachable;
}

/// Allocates addressable memory from the given address space and returns the first memory address of the newly-allocated bytes.
/// Returns null if OOM or if the address space is zero page and no contiguous bytes were found.
pub fn alloc(addr_mem: *AddrMem, byte_count: u16, addr_space: AddrSpace) ?Addr {
    defer addr_mem.preserve(); // TODO
    log.debug("allocating {}B from .{s}", .{ byte_count, @tagName(addr_space) });

    switch (addr_space) {
        .any => {
            const use_zero_page =
                // Avoid allocating too much and possibly depleting the zero page.
                (byte_count >= 2 and byte_count <= 4 and
                // If this would deplete the zero page and does not allocate only one byte,
                // we deem it better to use absolute memory.
                addr_mem.zp_free.len >= byte_count + 1) or
                // We like single-byte allocations because we think it helps
                // the zero page be used for as many different things as possible,
                // meaning the zero page's efficiency can spread to more code,
                // and we allow this to deplete the zero page.
                (byte_count == 1 and addr_mem.zp_free.len >= 1);

            if (use_zero_page) {
                if (addr_mem.allocZeroPageMemory(byte_count)) |addr|
                    return .{ .zp = addr };
            }

            if (addr_mem.allocAbsoluteMemory(byte_count)) |addr|
                return .{ .abs = addr };
        },
        .zp => {
            if (addr_mem.allocZeroPageMemory(byte_count)) |addr|
                return .{ .zp = addr };
        },
    }

    // OOM
    return null;
}

/// Allocates contiguous zero page memory of the given size and returns a start address to the first byte.
/// Returns null if OOM or if no contiguous bytes were found.
fn allocZeroPageMemory(addr_mem: *AddrMem, byte_count: u16) ?u8 {
    assert(byte_count != 0);

    if (addr_mem.zp_free.len == 0)
        return null;

    const addrs = addr_mem.zp_free.constSlice();

    if (debug.runtime_safety) {
        if (!std.sort.isSorted(u8, addrs, {}, std.sort.asc(u8)))
            @panic("zero page addresses must be sorted low to high");
        for (addrs) |addr1, addr1_i| {
            for (addrs) |addr2, addr2_i| {
                if (addr1_i != addr2_i and addr1 == addr2)
                    @panic("zero page addresses must not have duplicates");
            }
        }
    }

    log.debug("allocZeroPageMemory: zp_free BEFORE: {any}", .{addrs});
    defer log.debug("allocZeroPageMemory: zp_free AFTER: {any}", .{addr_mem.zp_free.constSlice()});

    // Find a matching number of free bytes, each with an address only 1 apart from the one before it (contiguous).
    var contig_addrs_i: u8 = 0;
    var contig_addrs_len: u8 = 1;
    {
        var i: u8 = 0;
        while (true) {
            if (contig_addrs_len == byte_count) break;
            const addr1 = addrs[i];
            i += 1;
            if (i == addrs.len) {
                if (contig_addrs_len == byte_count) {
                    break;
                } else {
                    return null;
                }
            }
            const addr2 = addrs[i];
            const contig = addr1 == addr2 - 1;
            if (contig) {
                contig_addrs_len += 1;
                if (contig_addrs_len == byte_count) break;
            } else {
                contig_addrs_i = i;
                contig_addrs_len = 1;
            }
        }
    }

    // TODO: what's generally faster?
    //       1. multiple `swapRemove`s and then one `std.sort.sort` at the end,
    //       or 2. multiple `orderedRemove` and no `std.sort.sort`?
    // Remove in reverse to avoid OOB.
    var i = contig_addrs_len - 1;
    while (i > 0) : (i -= 1)
        _ = addr_mem.zp_free.swapRemove(contig_addrs_i + i);
    const start_addr = addr_mem.zp_free.swapRemove(contig_addrs_i + i);
    std.sort.sort(u8, addr_mem.zp_free.slice(), {}, std.sort.asc(u8));
    return start_addr;
}

/// Allocates contiguous absolute memory of the given size and returns a start address to the first byte.
/// Returns null if OOM.
fn allocAbsoluteMemory(addr_mem: *AddrMem, byte_count: u16) ?u16 {
    assert(byte_count != 0);
    // TODO:
    //if (addr == abs.abs_end) {
    //    return null;
    //}
    addr_mem.abs_offset -= byte_count; // Grow the stack downwards.
    return addr_mem.abs_offset + 1;
}

pub fn free(addr_mem: *AddrMem, addr: Addr, byte_count: u16) void {
    log.debug("free: zp_free BEFORE: {any}", .{addr_mem.zp_free.constSlice()});
    defer log.debug("free: zp_free AFTER: {any}", .{addr_mem.zp_free.constSlice()});

    defer addr_mem.preserve(); // TODO
    log.debug("freeing {} bytes from {}", .{ byte_count, addr });
    switch (addr) {
        .zp => |zp_addr| {
            // TODO: insert at the right position right away? without sorting after
            var i: u8 = 0;
            while (i < byte_count) : (i += 1)
                addr_mem.zp_free.appendAssumeCapacity(zp_addr + i);
            std.sort.sort(u8, addr_mem.zp_free.slice(), {}, std.sort.asc(u8));
        },
        .abs => |abs_addr| {
            // TODO: free memory!
            //       do it this way: at the start of a branch record the offset and when that branch dies, reset
            _ = abs_addr;
        },
    }
}

test allocZeroPageMemory {
    var addr_mem = AddrMem{
        .zp_free = std.BoundedArray(u8, 256){},
        .abs_offset = undefined,
    };

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 1, 2, 4, 5, 0x80, 0x90 });
    try testing.expectEqual(@as(u8, 0), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 4), addr_mem.allocZeroPageMemory(2).?);
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(2));
    try testing.expectEqual(@as(u8, 0x80), addr_mem.allocZeroPageMemory(1).?);
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(10));
    try testing.expectEqual(@as(u8, 0x90), addr_mem.allocZeroPageMemory(1).?);
    try testing.expectEqual(@as(u8, 0), @intCast(u8, addr_mem.zp_free.len));

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xfe, 0xff });
    try testing.expectEqual(@as(u8, 0xfe), addr_mem.allocZeroPageMemory(1).?);
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(0xff));
    try testing.expectEqual(@as(u8, 0xff), addr_mem.allocZeroPageMemory(1).?);
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(1));
    try testing.expectEqual(@as(u8, 0), @intCast(u8, addr_mem.zp_free.len));

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 2, 4, 6, 8, 10 });
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(2));
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(3));
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(4));
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(5));
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(6));
    try testing.expectEqual(@as(u8, 6), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFD, 0xFF });
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(2));
    try testing.expectEqual(@as(u8, 2), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 2, 4 });
    try testing.expectEqual(@as(?u8, null), addr_mem.allocZeroPageMemory(3));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFD, 0xFE, 0xFF });
    try testing.expectEqual(@as(u8, 0xFD), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 0), @intCast(u8, addr_mem.zp_free.len));

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFC, 0xFD, 0xFE });
    try testing.expectEqual(@as(u8, 0xFC), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 0), @intCast(u8, addr_mem.zp_free.len));

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFB, 0xFC, 0xFD, 0xFE });
    try testing.expectEqual(@as(u8, 0xFB), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 1), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0x02, 0x2A, 0x52, 0xFB, 0xFC, 0xFD, 0xFE });
    try testing.expectEqual(@as(u8, 0xFB), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 4), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 2, 4, 8, 16, 17, 18, 32, 64, 128 });
    try testing.expectEqual(@as(u8, 16), addr_mem.allocZeroPageMemory(3).?);
    try testing.expectEqual(@as(u8, 6), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;

    addr_mem.zp_free.appendSliceAssumeCapacity(&[_]u8{ 2, 42, 82, 253, 254 });
    try testing.expectEqual(@as(u8, 5), @intCast(u8, addr_mem.zp_free.len));
    addr_mem.zp_free.len = 0;
}

test allocAbsoluteMemory {
    // TODO: more tests
    var addr_mem = AddrMem{
        .zp_free = undefined,
        .abs_offset = undefined,
        //.abs_end = 0xfeff,
    };

    addr_mem.abs_offset = 0xffff;
    try testing.expectEqual(@as(u16, 0xffff), addr_mem.allocAbsoluteMemory(1).?);
    try testing.expectEqual(@as(u16, 0xfffe), addr_mem.allocAbsoluteMemory(1).?);
    try testing.expectEqual(@as(u16, 0xfff0), addr_mem.allocAbsoluteMemory(0xe).?);
    try testing.expectEqual(@as(u16, 0xffee), addr_mem.allocAbsoluteMemory(2).?);
    try testing.expectEqual(@as(u16, 0xff6e), addr_mem.allocAbsoluteMemory(128).?);
    //try testing.expectEqual(@as(?MV,null), addr_mem.allocAbsoluteMemory(0xffff).?);
    try testing.expectEqual(@as(u16, 0xff6d), addr_mem.abs_offset);
}

// TODO: are these tests run as part of `zig build test`?
comptime {
    _ = AddrMem;
}
