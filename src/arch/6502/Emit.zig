//! This is what emits our MIR to the format we want, such as executable machine code.

// TODO: would be nice if we could emit a textual representation using -femit-asm once that is a possibility for in-house backends

const std = @import("std");
const math = std.math;
const mem = std.mem;
const log = std.log;
const debug = std.debug;
const assert = debug.assert;
const Mir = @import("Mir.zig");
const link = @import("../../link.zig");
const Module = @import("../../Module.zig");
const ErrorMsg = Module.ErrorMsg;
const DebugInfoOutput = @import("../../codegen.zig").DebugInfoOutput;

const Emit = @This();

/// Our machine instructions including payloads to be lowered.
mir: Mir,
bin_file: *link.File,
/// This is the final output of our codegen.
code: *std.ArrayList(u8),

pub fn emitMir(emit: *Emit) !void {
    const mir_tags = emit.mir.instructions.items(.tag);

    try emit.code.ensureTotalCapacity(mir_tags.len);

    for (mir_tags) |tag, inst| {
        try emit.code.append(@enumToInt(tag));

        // emit the operand, if any
        const data = emit.mir.instructions.items(.data)[inst];
        const addr_mode = tag.getAddressingMode();
        if (mem.eql(u8, addr_mode, "impl")) {
            // no payload
        } else if (mem.eql(u8, addr_mode, "imm")) {
            try emit.emitByte(data.imm);
        } else if (mem.eql(u8, addr_mode, "zp")) {
            try emit.emitByte(data.zp);
        } else if (mem.eql(u8, addr_mode, "abs")) {
            try emit.emitAddress(data.abs);
        } else unreachable;
    }
}

fn emitByte(emit: *Emit, byte: u8) !void {
    try emit.code.append(byte);
}

fn emitWord(emit: *Emit, word: u16) !void {
    try emit.code.writer().writeIntLittle(u16, word);
}

fn emitAddress(emit: *Emit, abs_addr: Mir.Inst.AbsoluteAddress) !void {
    switch (abs_addr) {
        .imm => |imm| try emit.emitWord(imm),
        .unresolved => |unresolved| {
            // resolve and emit
            const addr = if (emit.bin_file.cast(link.File.Prg)) |prg| addr: {
                var it = prg.blocks.iterator();
                while (it.next()) |entry| {
                    std.log.debug("block(Emit.zig): {} -> {}", .{ entry.key_ptr.*, entry.value_ptr.* });
                }
                const load_address = prg.getLoadAddress();
                var program_size = emit.getProgramSize()
                // note that our program is loaded at the load address so
                // subtract the size of the address itself
                - @sizeOf(@TypeOf(load_address));
                var program_index: u16 = 0;
                const decl_index_and_blocks = try prg.getAllBlocks(emit.bin_file.allocator);
                defer emit.bin_file.allocator.free(decl_index_and_blocks);
                break :addr for (decl_index_and_blocks) |decl_index_and_block| {
                    const block = decl_index_and_block.block;
                    assert(!block.entry_point);
                    std.log.debug("load_address: {}, program_size: {}, program_index: {}, offset: {}", .{ load_address, program_size, program_index, unresolved.offset });
                    if (block.index == unresolved.blk_i) {
                        const imm = load_address + ((program_size + program_index) + unresolved.offset);
                        log.debug("block with index {x} will end up at address {x} in the final binary", .{ block.index, imm });
                        break imm;
                    }
                    if (block.code) |code|
                        program_index += @intCast(u16, code.len);
                } else unreachable;
            } else {
                unreachable;
            };
            try emit.emitWord(addr);
        },
        .current => |current| {
            const addr = if (emit.bin_file.cast(link.File.Prg)) |prg| addr: {
                const load_address = prg.getLoadAddress();
                var program_size = emit.getProgramSize()
                // note that our program is loaded at the load address so
                // subtract the size of the address itself
                - @sizeOf(@TypeOf(load_address));
                var program_index: u16 = 0;
                const decl_index_and_blocks = try prg.getAllBlocks(emit.bin_file.allocator);
                defer emit.bin_file.allocator.free(decl_index_and_blocks);
                break :addr for (decl_index_and_blocks) |decl_index_and_block| {
                    if (decl_index_and_block.decl_index == current.decl_index) {
                        break :addr load_address + program_size + program_index
                        // - 3 for JMP + absolute address
                            -
                            3
                        // correct off-by-one
                        + 1;
                    }
                    if (decl_index_and_block.block.code) |code|
                        program_index += @intCast(u16, code.len);
                } else unreachable;
            } else unreachable;
            try emit.emitWord(addr);
        },
    }
}

/// Returns the size of the program up to this function including any headers sans any symbols.
fn getProgramSize(emit: Emit) u16 {
    const header_size = if (emit.bin_file.cast(link.File.Prg)) |prg| header_size: {
        break :header_size @intCast(u16, prg.header.len);
    } else unreachable;
    var byte_size: u16 = header_size;
    var i: u16 = 0;
    while (i < emit.mir.instructions.len) : (i += 1) {
        const inst = Mir.Inst{
            .tag = emit.mir.instructions.items(.tag)[i],
            .data = emit.mir.instructions.items(.data)[i],
        };
        byte_size += inst.getByteSize();
    }
    return byte_size;
}
