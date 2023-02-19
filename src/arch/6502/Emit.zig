//! Emits MIR to the format we want, such as executable machine code.

// TODO: would be nice if we could emit a textual representation using -femit-asm once that is a possibility for in-house backends.
//       also, add ';' comments that explain the purpose of an instruction. for that add metadata to each instruction explaining why it was added
//       (maybe only in debug mode or make it configurable)

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const mem = std.mem;
const log = std.log.scoped(.emit);
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
/// The output of this emit.
code: *std.ArrayList(u8),
/// The index of the declaration that code is being generated for.
decl_index: Module.Decl.Index,

pub fn emitMir(emit: *Emit) !void {
    const mir_tags = emit.mir.instructions.items(.tag);

    try emit.code.ensureTotalCapacity(
        // This accounts for the opcode bytes.
        mir_tags.len
        // This accounts for that most opcodes take at least one operand byte.
        * 2,
    );
    log.debug("emit.code: len/capacity: {d}/{d}", .{ emit.code.items.len, emit.code.capacity });
    defer log.debug("emit.code: len/capacity: {d}/{d}", .{ emit.code.items.len, emit.code.capacity });

    for (mir_tags) |tag, inst| {
        // Emit the opcode.
        try emit.emitByte(@enumToInt(tag));

        // Emit the operand, if any.
        const data = emit.mir.instructions.items(.data)[inst];
        const addr_mode = tag.getAddrMode();
        switch (addr_mode) {
            .impl => {
                // No payload.
            },
            .imm => try emit.emitImmediate(data.imm),
            .zp, .x_zp, .y_zp, .x_ind_zp, .ind_y_zp => try emit.emitZeroPageAddress(data.zp),
            .abs, .x_abs, .y_abs, .ind_abs => try emit.emitAbsoluteAddress(data.abs),
            .rel => try emit.emitRelativeOffset(data.rel),
        }
    }
}

fn emitByte(emit: *Emit, byte: u8) !void {
    try emit.code.append(byte);
}
fn emitWord(emit: *Emit, word: u16) !void {
    try emit.code.writer().writeIntLittle(u16, word);
}
fn getCodeOffset(emit: Emit) u16 {
    return @intCast(u16, emit.code.items.len);
}

fn emitImmediate(emit: *Emit, imm: Mir.Inst.Imm) !void {
    switch (imm) {
        .val => |val| try emit.emitByte(val),
        .unres_addr_half => |unres_addr_half| {
            // We are currently emitting a single function's code and we can not
            // resolve this absolute address half in this function before we have the code of all
            // other functions, so we will let the linker fix this up later and emit
            // a placeholder for now.
            const code_offset = emit.getCodeOffset();
            try emit.emitByte(undefined);
            if (emit.bin_file.cast(link.File.Prg)) |prg| {
                try prg.unres_addrs.append(emit.bin_file.allocator, .{
                    .decl_index = emit.decl_index,
                    .code_offset = code_offset,
                    .block_index = unres_addr_half.block_index,
                    .addend = 0,
                    .half = unres_addr_half.half,
                });
            } else unreachable;
        },
    }
}

fn emitZeroPageAddress(emit: *Emit, zp_addr: u8) !void {
    try emit.emitByte(zp_addr);
}

fn emitAbsoluteAddress(emit: *Emit, abs_addr: Mir.Inst.Abs) !void {
    switch (abs_addr) {
        .fixed => |fixed| {
            // Although it is perfectly fine to use operands <= $FF for absolute addressing modes,
            // if the address is <= $FF it means that we could have used a zero page addressing mode
            // which we would want to use instead because it saves one byte, so this assertion ensures we do that optimization.
            assert(fixed > 0xFF);
            try emit.emitWord(fixed);
        },
        .unres => |unres| {
            // We are currently emitting a single function's code and we can not
            // resolve this absolute address in this function before we have the code of all
            // other functions, so we will let the linker fix this up later and emit
            // a placeholder for now.
            // TODO(meeting): I have investigated the above statement's accuracy extensively in the past
            //                but maybe someone else knows better - is it possible to resolve absolute memory addresses on the spot
            //                in Func.zig? Can we come up with a design to allow that? one worry is that we don't know in which
            //                order functions are generated. if confirmed that it is not possible, remove this TODO.
            const code_offset = emit.getCodeOffset();
            try emit.emitWord(undefined);
            if (emit.bin_file.cast(link.File.Prg)) |prg| {
                try prg.unres_addrs.append(emit.bin_file.allocator, .{
                    .decl_index = emit.decl_index,
                    .code_offset = code_offset,
                    .block_index = unres.block_index,
                    .addend = abs_addr.unres.addend,
                    .half = null,
                });
            } else unreachable;
        },
        .decl => |decl| {
            // We are currently emitting a single function's code and we can not
            // resolve this absolute address in this function before we have the code of all
            // other functions, so we will let the linker fix this up later and emit
            // a placeholder for now.
            const code_offset = emit.getCodeOffset();
            try emit.emitWord(undefined);
            if (emit.bin_file.cast(link.File.Prg)) |prg| {
                try prg.unres_addrs.append(emit.bin_file.allocator, .{
                    .decl_index = decl.index,
                    .code_offset = code_offset,
                    .block_index = null,
                    .addend = decl.addend,
                    .half = null,
                });
            } else unreachable;
        },
    }
}

fn emitRelativeOffset(emit: *Emit, rel: i8) !void {
    try emit.emitByte(@bitCast(u8, rel));
}
