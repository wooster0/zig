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
        .fixed => |fixed_addr| try emit.emitWord(fixed_addr),
        .unres => |unres| {
            // We are currently emitting a single function's code and we can not
            // resolve this absolute address in this function before we have the code of all
            // other functions, so we will let the linker fix this up later and emit
            // a placeholder for now.
            const code_offset = emit.getCodeOffset();
            switch (abs_addr) {
                .fixed => unreachable,
                .unres => try emit.emitWord(undefined),
            }
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
        //.current => |current| {
        //    _ = current;
        //    @panic("TODO");
        //    //const addr = if (emit.bin_file.cast(link.File.Prg)) |prg| addr: {
        //    //    const load_address = prg.getLoadAddress();
        //    //    var program_size = emit.getProgramSize()
        //    //    // note that our program is loaded at the load address so
        //    //    // subtract the size of the address itself
        //    //    - @sizeOf(@TypeOf(load_address));
        //    //    var program_index: u16 = 0;
        //    //    const decl_index_and_blocks = try prg.getAllBlocks(emit.bin_file.allocator);
        //    //    defer emit.bin_file.allocator.free(decl_index_and_blocks);
        //    //    break :addr for (decl_index_and_blocks) |decl_index_and_block| {
        //    //        if (decl_index_and_block.decl_index == current.decl_index) {
        //    //            break :addr load_address + program_size + program_index
        //    //            // - 3 for JMP + absolute address
        //    //                -
        //    //                3
        //    //            // correct off-by-one
        //    //            + 1;
        //    //        }
        //    //        if (decl_index_and_block.block.code) |code|
        //    //            program_index += @intCast(u16, code.len);
        //    //    } else unreachable;
        //    //} else unreachable;
        //    //try emit.emitWord(addr);
        //},
    }
}

fn emitRelativeOffset(emit: *Emit, rel: i8) !void {
    try emit.emitByte(@bitCast(u8, rel));
}

///// Returns the size of the program up to this function including any headers sans any symbols.
//fn getProgramSize(emit: Emit) u16 {
//    const header_size = if (emit.bin_file.cast(link.File.Prg)) |prg| header_size: {
//        break :header_size @intCast(u16, prg.header.len);
//    } else unreachable;
//    var byte_size: u16 = header_size;
//    var i: u16 = 0;
//    while (i < emit.mir.instructions.len) : (i += 1) {
//        const inst = Mir.Inst{
//            .tag = emit.mir.instructions.items(.tag)[i],
//            .data = emit.mir.instructions.items(.data)[i],
//        };
//        byte_size += inst.getByteSize();
//    }
//    return byte_size;
//}
