//! Code generation of a function for the MOS Technology 6502 8-bit microprocessor,
//! powering your favorite retro console.
//!
//! Various material and references:
//!
//! * http://6502.org
//! * https://en.wikipedia.org/wiki/MOS_Technology_6502
//! * https://skilldrick.github.io/easy6502
//! * https://www.pagetable.com/c64ref/6502/?tab=2
//! * http://6502.org/tutorials/6502opcodes.html
//! * https://www.masswerk.at/6502/6502_instruction_set.html
//! * https://www.nesdev.org/wiki/6502_assembly_optimisations
//! * https://llx.com/Neil/a2/opcodes.html
//! * https://dustlayer.com/news
//! * https://github.com/KaiSandstrom/C64-Minesweeper
//!
//! We make use of two stacks:
//! * The hardware stack which is limited and we only use for function calls and temporarily preserving registers during transfer operations.
//! * The software stack which is used for allocating memory because it is much bigger than the hardware stack.

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
const Register = bits.Register;
const abi = @import("abi.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");

const Func = @This();

//
// input
//

/// Represents the file for the final output of the whole codegen.
/// This is the only field that references data preserved across multiple function codegens.
bin_file: *link.File,
/// The location in source of this function.
src_loc: Module.SrcLoc,
/// Properties of this function.
props: *Module.Fn,
/// Analyzed Intermediate Representation; the input for the codegen of this function.
air: Air,
/// This lets us know whether an operand dies at which location.
liveness: Liveness,

//
// call info
//

/// The locations of the arguments to this function.
args: []MV = undefined,
/// The index of the current argument that is being resolved in `airArg`.
arg_index: u16 = 0,
/// The result location of the return value of the current function.
ret_val: MV = undefined,

/// An error message for if codegen fails.
/// This is set if error.CodegenFail happens.
err_msg: *Module.ErrorMsg = undefined,

/// Contains a list of current branches.
/// When we return from a branch, the branch will be popped from this list,
/// which means branches can only contain references from within its own branch,
/// or a branch higher (lower index) in the tree.
/// The first branch contains comptime-known constants;
/// all other branches contain runtime-known values.
branches: std.ArrayListUnmanaged(Branch) = .{},

/// The MIR output of this codegen.
mir_instructions: std.MultiArrayList(Mir.Inst) = .{},
//mir_extra: std.ArrayListUnmanaged(u32) = .{},

//
// safety
//

/// After we finished lowering something we need to call `airFinish`.
/// This is equal to the amount of times we called `airFinish`.
/// Used to find forgotten calls to `airFinish`.
air_bookkeeping: if (debug.runtime_safety) usize else void = if (debug.runtime_safety) 0 else {},
/// Holds the index of the current AIR instruction generated.
/// Used to prevent clobbering registers unintentionally.
air_current_inst: if (debug.runtime_safety) Air.Inst.Index else void = if (debug.runtime_safety) undefined else {},

// TODO: the original Furby used a 6502 chip with 128 bytes of RAM and no Y register.
//       allow compatibility with such variants of the 6502 using a constraint config like this
//       based on which we generate conformant code.
//       This also applies to the NES. to be compatible we basically just don't have to use decimal mode AFAIUI
//       (it's the 2A03: https://en.wikipedia.org/wiki/MOS_Technology_6502#Variations_and_derivatives)
//constraints: struct {},

// TODO regarding naming of arch and OS for 6502-related targets:
// * rename the `c64` OS to c64_basic? cbm_basic?
// * what if we wanted to add official support for https://c64os.com/? Wouldn't it be 6502-c64_os?
//   figure this out after testing the other Commodore machines with this backend
//   and see how compatible they are.
// * what about the NES? would it be 6502-nes or 6502-freestanding or 2a03-freestanding?

//
// memory
//
// it is encouraged to tell helpers where you think they should put their result in memory
// TODO: let the user provide information about the register and flags' initial state so we don't have to assume the initial state is unknown?
//       what if we're embedded into something else? isn't there a 6502 CPU implemented in Minecraft? that way we can avoid setting these flags.

/// Addressable memory state global across all functions.
/// Helpers: allocMem
memory: Memory,

// registers
//
// if you wish to mark a register as used but have no AIR instruction to offer,
// you should probably allocate to addressable memory.
//
// helpers: freeReg, takeReg

// TODO: add prefix _owner _user _holder or something to this field.
//       that would also discourage using them directly.
/// The AIR instruction that uses the accumulator register, if any.
reg_a: ?Air.Inst.Index = null,
/// The AIR instruction that uses the X index register, if any.
reg_x: ?Air.Inst.Index = null,
/// The AIR instruction that uses the Y index register, if any.
reg_y: ?Air.Inst.Index = null,

// status register
// TODO: how could we properly track these throughout functions?

/// Whether binary-coded decimal mode is on.
decimal: Flag(.sed_impl, .cld_impl, "decimal") = .{},
/// Whether we have a carry bit.
carry: Flag(.sec_impl, .clc_impl, "carry") = .{},

pub fn generate(
    bin_file: *link.File,
    src_loc: Module.SrcLoc,
    props: *Module.Fn,
    air: Air,
    liveness: Liveness,
    code: *std.ArrayList(u8),
    debug_output: DebugInfoOutput,
) GenerateSymbolError!Result {
    _ = debug_output;

    const memory = Memory.init(bin_file.options.target, bin_file);
    var func = Func{
        .bin_file = bin_file,
        .src_loc = src_loc,
        .props = props,
        .air = air,
        .liveness = liveness,
        .memory = memory,
    };
    defer func.deinit();

    func.gen() catch |err| switch (err) {
        error.CodegenFail => return Result{ .fail = func.err_msg },
        error.OutOfMemory => |other| return other,
    };

    var mir = Mir{
        .instructions = func.mir_instructions.slice(),
    };

    var emit = Emit{
        .mir = mir,
        .bin_file = bin_file,
        .code = code,
        .decl_index = func.props.owner_decl,
    };
    try emit.emitMir();

    return Result.ok;
}

fn deinit(func: *Func) void {
    const allocator = func.getAllocator();
    for (func.branches.items) |*branch|
        branch.deinit(allocator);
    func.branches.deinit(allocator);
    func.mir_instructions.deinit(allocator);
}

/// Returns the allocator we will be using throughout.
fn getAllocator(func: Func) mem.Allocator {
    return func.bin_file.allocator;
}

/// Returns the target we are compiling for.
fn getTarget(func: Func) std.Target {
    return func.bin_file.options.target;
}

/// Returns the module representing this entire compilation.
fn getMod(func: Func) *Module {
    return func.bin_file.options.module.?;
}

/// Returns the index of this function's declaration.
fn getDeclIndex(func: Func) Decl.Index {
    return func.props.owner_decl;
}

/// Returns the type of this function.
fn getType(func: Func) Type {
    return func.getMod().declPtr(func.getDeclIndex()).ty;
}

/// This represents the location of a value in memory.
const MemoryValue = union(enum) {
    /// The value has no bits we could represent at runtime.
    none: void,
    /// The value is immediately available.
    /// This becomes Mir.Inst.Data.imm.
    // TODO: combine zp_imm with this?
    imm: u8,
    /// The value is in a register.
    reg: Register,
    /// The value is in the first page of memory.
    /// TODO: merge this and `abs` into a single `addr` and check if < 255 in Emit.zig.
    ///       after that rename abs_unresolved too.
    /// This becomes Mir.Inst.Data.zp.
    zp: u8,
    /// The value is in one of the 256 pages of memory (64 KiB),
    /// excluding the first one (the zero page).
    /// This becomes Mir.Inst.Data.abs.imm.
    abs: u16,
    /// Same as `abs` except that this is a block index and
    /// the absolute memory address is yet to be resolved by the linker.
    /// Here is an example of why this exists:
    /// In the PRG file format we want the main function's code to come first.
    /// This means if we encounter an unnamed constant in the main function,
    /// and we put it after the main's code, we do not at that point know
    /// how big main's code is going to be in the end because we are still in the process
    /// of codegening it, meaning we do not know where the unnamed constant will be, so we use this.
    /// This becomes Mir.Inst.Data.abs.unresolved.
    // TODO: rename to block?
    abs_unresolved: struct {
        blk_i: u16,
        offset: u16 = 0,
    },
    // TODO: merge the following two into a single `addr_imm` and check if < 255 in Emit.zig.
    //       otherwise, document why exactly the distinction is necessary
    /// The value is an absolute memory address and is immediately available.
    abs_imm: u16,
    /// The value is a zero page address and is immediately available.
    zp_imm: u8,

    fn eql(lhs: MV, rhs: MV) bool {
        return switch (lhs) {
            .none => rhs == .none,
            .imm => |imm| rhs == .imm and rhs.imm == imm,
            .reg => |reg| rhs == .reg and rhs.reg == reg,
            .zp => |addr| rhs == .zp and rhs.zp == addr,
            .abs => |addr| rhs == .abs and rhs.abs == addr,
            .abs_unresolved => |unresolved| rhs == .abs_unresolved and rhs.abs_unresolved.offset == unresolved.offset and rhs.abs_unresolved.blk_i == unresolved.blk_i,
            .abs_imm => unreachable, // TODO
            .zp_imm => unreachable, // TODO
        };
    }

    /// Returns the given value in memory subscripted by the index in form of an offset.
    fn index(val: MV, offset: u16) MV {
        switch (val) {
            .none => unreachable,
            .imm => |_| {
                assert(offset == 0);
                return val;
            },
            .reg => {
                assert(offset == 0);
                return val;
            },
            .zp => |addr| {
                // this is safe and will not cross the page boundary
                // as long as the zero page allocation happened correctly
                return .{ .zp = addr + @intCast(u8, offset) };
            },
            .abs => |addr| return .{ .abs = addr + offset },
            .abs_unresolved => |unresolved| {
                // we do not want both offsets to be non-zero because it may not be clear in that case what to do
                assert(unresolved.offset == 0 or offset == 0);
                const effective_offset = if (unresolved.offset == 0)
                    offset
                else if (offset == 0)
                    unresolved.offset
                else
                    unreachable;
                return .{ .abs_unresolved = .{ .blk_i = unresolved.blk_i, .offset = effective_offset } };
            },
            .abs_imm => |imm| {
                const bytes = @bitCast([2]u8, imm);
                return .{ .imm = bytes[offset] };
            },
            .zp_imm => {
                assert(offset == 0);
                return val;
            },
        }
    }
};
// this is common enough to deserve an alias
const MV = MemoryValue;

fn Flag(comptime set_inst: Mir.Inst.Tag, comptime clear_inst: Mir.Inst.Tag, comptime field_name: []const u8) type {
    return struct {
        const Self = @This();

        state: State = .unknown,

        const State = enum { set, clear, unknown };

        fn set(flag: *Self) !void {
            switch (flag.state) {
                .set => {},
                .clear, .unknown => {
                    const func = @fieldParentPtr(Func, field_name, flag);
                    try func.addInstImpl(set_inst);
                    flag.state = .set;
                },
            }
        }
        fn clear(flag: *Self) !void {
            switch (flag.state) {
                .clear => {},
                .set, .unknown => {
                    const func = @fieldParentPtr(Func, field_name, flag);
                    try func.addInstImpl(clear_inst);
                    flag.state = .clear;
                },
            }
        }
    };
}

/// Spills the register if not free and associates it with the new instruction.
/// If null is given, it means the register is temporarily needed and
/// will not have an owner thereafter.
fn freeReg(func: *Func, reg: Register, maybe_inst: ?Air.Inst.Index) !MV {
    log.debug("freeReg: making %{?} the owner of {}", .{ maybe_inst, reg });

    const spillReg = struct {
        fn spillReg(func2: *Func, reg2: Register, inst: Air.Inst.Index) !void {
            // TODO: instead of doing this here, do this everywhere else?
            //       can this be removed once we check for liveness in every AIR instruction?
            if (func2.liveness.isUnused(inst))
                return;
            const val = func2.getResolvedInst(inst);
            if (val == .reg)
                assert(reg2 == val.reg);
            const new_home = try func2.allocMem(Type.u8);
            log.debug("spilling register {} into {}", .{ reg2, new_home });
            try func2.trans(val, new_home, Type.u8, Type.u8);
            func2.currentBranch().inst_vals.putAssumeCapacity(inst, new_home);
        }
    }.spillReg;

    switch (reg) {
        .a => {
            if (func.reg_a) |old_inst| {
                if (maybe_inst) |inst|
                    assert(inst != old_inst); // TODO: can it happen that they're the same? in that case don't spill
                // TODO: use PHA and PLA to spill using the hardware stack?
                //       for that we very likely have to introduce a new MemoryValue that says
                //       the value has to be PLA'd first.
                //       also, can we TXA and TYA below and do it there too? measure total cycles of each solution
                try spillReg(func, reg, old_inst);
            }
            func.reg_a = maybe_inst;
        },
        .x => {
            if (func.reg_x) |old_inst| {
                if (maybe_inst) |inst|
                    assert(inst != old_inst); // TODO: can it happen that they're the same?
                try spillReg(func, reg, old_inst);
            }
            func.reg_x = maybe_inst;
        },
        .y => {
            if (func.reg_y) |old_inst| {
                if (maybe_inst) |inst|
                    assert(inst != old_inst); // TODO: can it happen that they're the same?
                try spillReg(func, reg, old_inst);
            }
            func.reg_y = maybe_inst;
        },
    }
    return .{ .reg = reg };
}

/// Assumes the given register is now owned by the new instruction and associates it with the new instruction.
/// Accepts null for compatibility.
fn takeReg(func: *Func, reg: Register, maybe_inst: ?Air.Inst.Index) MV {
    if (maybe_inst) |inst| {
        switch (reg) {
            .a => func.reg_a = inst,
            .x => func.reg_x = inst,
            .y => func.reg_y = inst,
        }
    }
    return .{ .reg = reg };
}

/// This represents the system's addressable memory where each byte has an address.
// TODO: move this to Memory.zig or better even AddressableMemory.zig.
//       and maybe rename bits.zig to regs.zig
const Memory = struct {
    /// Memory addresses in the first page of memory available for storage.
    /// This is expensive, sparsely available memory and very valuable because
    /// writing and reading from it is faster than for absolute memory.
    /// We treat each address separately, rather than using an offset, for ideal zero page
    /// usage on any system.
    /// For example, this handles the case of a system where byte #0 is usable,
    /// byte #1 is unusable (reserved by the system),
    /// and all other bytes after that of the zero page are usable.
    /// If we used offsets, this means our offset would start
    /// at byte #2, giving us one less byte than with this method of zero page allocation.
    zp_free: std.BoundedArray(u8, 256),
    /// The offset at which free absolute memory starts.
    /// Must not start in zero page. Must be contiguously free memory.
    /// This can be called a software stack because it grows downwards
    /// and emulates a stack similar to the hardware stack but is typically much bigger than 256 bytes.
    abs_offset: u16,
    // TODO: record all alloc() calls and where they came from (src_loc) and then after we know the total program size,
    //       if the last record's abs_offset is <= program_size,
    //       report an error for the specific alloc that crashed the stack into the program
    //
    //       alternatively, associate each alloc with an AIR instruction and track it that way?
    ///// The absolute memory address at which memory ends.
    //abs_end: u16,
    /// Our reserve of two contiguous zero page addresses, for temporary use.
    /// We need at most this in order for codegen not to fail due to zero page shortage.
    /// It is fine for this to be null if the reserve never ends up being used.
    // TODO: use this reserve for other things if we never end up needing this reserve.
    //       in the past we had a allocZeroPageDerefAddress function which allocated on-demand.
    //       problem was that it couldn't guarantee 2 contiguous addresses even if we had them once.
    zp_res: ?[2]u8,

    fn init(target: std.Target, bin_file: *link.File) Memory {
        if (bin_file.cast(link.File.Prg)) |prg| {
            if (prg.zp_free != null and prg.abs_offset != null and prg.zp_res != null) {
                // TODO: encapsulate into MemoryState or something and or require the linker to have a
                //       `6502_mem_state: *anyopaque` field or similar for you.
                //       or we introduce some other general concept to all linkers for storing state across function codegens
                return .{
                    .zp_free = prg.zp_free.?,
                    .abs_offset = prg.abs_offset.?,
                    .zp_res = prg.zp_res.?,
                };
            }
        } else unreachable;

        const zp_free = abi.getZeroPageAddresses(target);
        const abs_offset = abi.getAbsoluteMemoryOffset(target);
        var memory = Memory{
            .zp_free = zp_free,
            .abs_offset = abs_offset,
            .zp_res = undefined,
        };
        memory.zp_res = if (memory.allocZeroPageMemory(2)) |contig_zp_start|
            [2]u8{ contig_zp_start, contig_zp_start + 1 }
        else
            null;
        return memory;
    }

    ///// Allocates two bytes specifically from the zero page for the purpose of storing a pointer address and dereferencing that pointer.
    //// TODO: unused; remove
    //fn allocZeroPageDerefAddress(memory: *Memory) !MV {
    //    if (memory.allocZeroPageMemory(2)) |addr|
    //        return addr;
    //    const func = @fieldParentPtr(Func, "memory", memory);
    //    return func.fail("unable to dereference pointer due to zero page shortage", .{});
    //}

    /// Preserves the current memory state for the next function's codegen.
    fn preserve(memory: *Memory) void {
        // here we're tackling a problem unique to the 6502 of all backends:
        // while the other codegen backends use the hardware stack and their registers for storing everything
        // and that works fine for them because their stack is much bigger,
        // we use our hardware stack of 256 bytes only for function calling and preserving registers.
        // we use the software stack in form of zero page + absolute memory for everything else.
        // that memory is global rather than stack-local so we have to preserve the memory state
        // across function codegens.
        // any next function can simply continue with this memory state.
        const func = @fieldParentPtr(Func, "memory", memory);
        if (func.bin_file.cast(link.File.Prg)) |prg| {
            prg.abs_offset = memory.abs_offset;
            prg.zp_free = memory.zp_free;
            prg.zp_res = memory.zp_res;
        } else unreachable;
    }

    /// Returns a memory address pointing to the start of the newly-allocated amount of bytes.
    /// Returns either MV.zp or MV.abs.
    fn alloc(memory: *Memory, byte_count: u16) !MV {
        log.debug("allocating {} bytes", .{byte_count});

        const use_zero_page =
            // avoid allocating too much and possibly depleting the zero page
            (byte_count >= 2 and byte_count <= 4 and
            // if this would deplete the zero page and does not allocate only one byte,
            // we deem it better to use absolute memory
            memory.zp_free.len >= byte_count + 1) or
            // we like single-byte allocations because we think it helps
            // the zero page be used for as many different things as possible,
            // meaning the zero page's efficiency can spread to more code,
            // and we allow this to deplete the zero page
            (byte_count == 1 and memory.zp_free.len >= 1);

        if (use_zero_page) {
            if (memory.allocZeroPageMemory(byte_count)) |addr|
                return .{ .zp = addr };
        }

        if (memory.allocAbsoluteMemory(byte_count)) |addr|
            return .{ .abs = addr };

        // OOM
        // TODO: test this error
        const func = @fieldParentPtr(Func, "memory", memory);
        var depleted_total: u16 = 0;
        const zp_free = abi.getZeroPageAddresses(func.getTarget());
        depleted_total += @intCast(u8, zp_free.len);
        const abs_offset = abi.getAbsoluteMemoryOffset(func.getTarget());
        depleted_total += abs_offset; // TODO: `- abs_end`
        return func.fail("program depleted all {} bytes of memory", .{depleted_total});
    }

    /// Allocates contiguous zero page memory of the given size and returns a start address to the first byte.
    /// Returns null if OOM or if no contiguous bytes were found.
    fn allocZeroPageMemory(memory: *Memory, byte_count: u16) ?u8 {
        assert(byte_count != 0);

        std.debug.print("zp_free BEFORE: {any}\n", .{memory.zp_free.constSlice()});
        defer std.debug.print("zp_free AFTER: {any}\n", .{memory.zp_free.constSlice()});

        if (memory.zp_free.len == 0)
            return null;

        const addrs = memory.zp_free.constSlice();
        if (debug.runtime_safety and !std.sort.isSorted(u8, addrs, {}, std.sort.asc(u8)))
            @panic("free zero page addresses must be sorted low to high");

        // find a matching number of free bytes, each with an address only 1 apart from the one before it (contiguous)
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
                assert(addr1 != addr2); // zero page addresses must not contain duplicates
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

        std.debug.print("i: {} len: {}\n", .{ contig_addrs_i, contig_addrs_len });
        // TODO: what's generally faster?
        //       1. multiple `swapRemove`s and then one `std.sort.sort` at the end
        //       or 2. multiple `orderedRemove` and no `std.sort.sort`?
        // remove in reverse to avoid OOB
        var i = contig_addrs_len - 1;
        while (i > 0) : (i -= 1)
            _ = memory.zp_free.swapRemove(contig_addrs_i + i);
        const start_addr = memory.zp_free.swapRemove(contig_addrs_i + i);
        std.sort.sort(u8, memory.zp_free.slice(), {}, std.sort.asc(u8));
        return start_addr;
    }

    /// Allocates contiguous absolute memory of the given size and returns a start address to the first byte.
    /// Returns null if OOM.
    fn allocAbsoluteMemory(memory: *Memory, byte_count: u16) ?u16 {
        assert(byte_count != 0);
        // TODO:
        //if (addr == abs.abs_end) {
        //    return null;
        //}
        memory.abs_offset -= byte_count; // grow the stack downwards
        return memory.abs_offset + 1;
    }

    fn free(memory: *Memory, val: MV, ty: Type) void {
        const func = @fieldParentPtr(Func, "memory", memory);
        const byte_size = func.getByteSize(ty).?;
        log.debug("freeing {} bytes from {}", .{ byte_size, val });
        std.debug.print("zp_free BEFORE: {any}\n", .{memory.zp_free.constSlice()});
        defer std.debug.print("zp_free AFTER: {any}\n", .{memory.zp_free.constSlice()});
        switch (val) {
            .zp => |addr| {
                // TODO: insert at the right position right away? without sorting after
                var i: u8 = 0;
                while (i < byte_size) : (i += 1)
                    memory.zp_free.appendAssumeCapacity(addr + i);
                std.sort.sort(u8, memory.zp_free.slice(), {}, std.sort.asc(u8));
            },
            .abs => |addr| {
                // TODO: free memory!
                //       do it this way: at the start of a branch record the offset and when that branch dies, reset
                _ = addr;
            },
            else => unreachable,
        }
    }

    test allocZeroPageMemory {
        var zp_free = std.BoundedArray(u8, 256){};
        zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 1, 2, 4, 5, 0x80, 0x90 });
        var memory = Memory{
            .zp_free = zp_free,
            .abs_offset = undefined,
            .zp_res = undefined,
        };
        try testing.expectEqual(@as(u8, 0), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 4), memory.allocZeroPageMemory(2).?);
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(2));
        try testing.expectEqual(@as(u8, 0x80), memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(10));
        try testing.expectEqual(@as(u8, 0x90), memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xfe, 0xff });
        try testing.expectEqual(@as(u8, 0xfe), memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(0xff));
        try testing.expectEqual(@as(u8, 0xff), memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(1));
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 2, 4, 6, 8, 10 });
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(2));
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(3));
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(4));
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(5));
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(6));
        try testing.expectEqual(@as(u8, 6), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFD, 0xFF });
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(2));
        try testing.expectEqual(@as(u8, 2), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0, 2, 4 });
        try testing.expectEqual(@as(?u8, null), memory.allocZeroPageMemory(3));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFD, 0xFE, 0xFF });
        try testing.expectEqual(@as(u8, 0xFD), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFC, 0xFD, 0xFE });
        try testing.expectEqual(@as(u8, 0xFC), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xFB, 0xFC, 0xFD, 0xFE });
        try testing.expectEqual(@as(u8, 0xFB), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 1), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0x02, 0x2A, 0x52, 0xFB, 0xFC, 0xFD, 0xFE });
        try testing.expectEqual(@as(u8, 0xFB), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 4), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 2, 4, 8, 16, 17, 18, 32, 64, 128 });
        try testing.expectEqual(@as(u8, 16), memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(@as(u8, 6), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 2, 42, 82, 253, 254 });
        try testing.expectEqual(@as(u8, 5), @intCast(u8, memory.zp_free.len));
        memory.zp_free.len = 0;
    }

    test allocAbsoluteMemory {
        var memory = Memory{
            .zp_free = undefined,
            .abs_offset = 0xffff,
            //.abs_end = 0xfeff,
            .zp_res = undefined,
        };
        try testing.expectEqual(@as(u16, 0xffff), memory.allocAbsoluteMemory(1).?);
        try testing.expectEqual(@as(u16, 0xfffe), memory.allocAbsoluteMemory(1).?);
        try testing.expectEqual(@as(u16, 0xfff0), memory.allocAbsoluteMemory(0xe).?);
        try testing.expectEqual(@as(u16, 0xffee), memory.allocAbsoluteMemory(2).?);
        try testing.expectEqual(@as(u16, 0xff6e), memory.allocAbsoluteMemory(128).?);
        //try testing.expectEqual(@as(?MV,null), memory.allocAbsoluteMemory(0xffff).?);
        try testing.expectEqual(@as(u16, 0xff6d), memory.abs_offset);
    }
};

// TODO: are these tests run as part of `zig build test`?
comptime {
    _ = Memory;
}

const Branch = struct {
    inst_vals: std.AutoArrayHashMapUnmanaged(Air.Inst.Index, MV) = .{},

    fn deinit(branch: *Branch, allocator: mem.Allocator) void {
        branch.inst_vals.deinit(allocator);
    }
};

fn currentBranch(func: *Func) *Branch {
    return &func.branches.items[func.branches.items.len - 1];
}

/// Returns the byte size of a type or null if the byte size is too big.
/// Assert non-null in contexts where a type is involved where the compiler made sure the bit width is <= 65535.
/// For specific bit sizes to functions that only take type parameters, use `Type.Tag.int_unsigned.create(allocator, size)`.
fn getByteSize(func: Func, ty: Type) ?u16 {
    return math.cast(u16, ty.abiSize(func.getTarget()));
}

const CallMVs = struct {
    args: []MV,
    return_value: MV,

    fn deinit(values: *CallMVs, allocator: mem.Allocator) void {
        allocator.free(values.args);
    }
};
/// This tells us where the arguments and the return value should be based on a function's prototype.
/// This function must always give the same output for the same input.
///
/// The reason we need calling conventions is that a function code is included only once
/// in the binary and it can thus only take in the arguments in one specific way
/// and so all call sites need to do it the same way for it to work out.
///
/// So, this is called both when we call a function and once that function is generated.
/// In both cases for the same function we will receive the same call values.
fn resolveCallingConventionValues(func: *Func, fn_ty: Type) !CallMVs {
    // ref: https://llvm-mos.org/wiki/C_calling_convention
    const cc = fn_ty.fnCallingConvention();
    const allocator = func.getAllocator();
    const param_types = try allocator.alloc(Type, fn_ty.fnParamLen());
    defer allocator.free(param_types);
    fn_ty.fnParamTypes(param_types);
    var values = CallMVs{
        .args = try allocator.alloc(MV, param_types.len),
        .return_value = undefined,
    };
    errdefer allocator.free(values.args);

    switch (cc) {
        .Naked => {
            // naked functions are not callable in normal code
            assert(values.args.len == 0);
            values.return_value = .{ .none = {} };
            return values;
        },
        .Unspecified, .C => {
            for (values.args) |*arg, i| {
                const param_ty = param_types[i];
                const param_size = func.getByteSize(param_ty).?;
                // TODO: preserve zero page addresses for this stuff once we have more of them
                // TODO: if size is > 3 bytes, allocate and pass pointer as A:X or something
                if (param_size != 1)
                    return func.fail("TODO: support parameters bigger than 1 byte", .{});
                arg.* = switch (i) {
                    0 => .{ .reg = .a },
                    1 => .{ .reg = .x },
                    2 => .{ .reg = .y },
                    else => return func.fail("TODO: support more than 3 parameters", .{}),
                };
            }
        },
        else => return func.fail("unsupported calling convention {}", .{cc}),
    }

    const ret_ty = fn_ty.fnReturnType();
    if (ret_ty.zigTypeTag() == .NoReturn or !ret_ty.hasRuntimeBits()) {
        values.return_value = .{ .none = {} };
    } else {
        const ret_ty_size = func.getByteSize(ret_ty).?;
        if (ret_ty_size != 1)
            return func.fail("TODO: support return values bigger than 1 byte", .{});
        values.return_value = .{ .reg = .a };
    }

    return values;
}

fn gen(func: *Func) !void {
    const allocator = func.getAllocator();
    var call_values = try func.resolveCallingConventionValues(func.getType());
    defer call_values.deinit(allocator);
    func.args = call_values.args;
    func.ret_val = call_values.return_value;

    try func.branches.append(allocator, .{});
    defer {
        var outer_branch = func.branches.pop();
        outer_branch.deinit(allocator);
    }

    try func.genBody(func.air.getMainBody());

    func.memory.preserve();

    log.debug("MIR generated for this function:", .{});
    var i: u16 = 0;
    while (i < func.mir_instructions.len) : (i += 1) {
        log.debug("  * .{{ .tag = {}, .data = {} }}", .{ func.mir_instructions.items(.tag)[i], func.mir_instructions.items(.data)[i] });
    }
}

fn genBody(func: *Func, body: []const Air.Inst.Index) !void {
    for (body) |inst| {
        const old_bookkeping_value = func.air_bookkeeping;

        if (debug.runtime_safety)
            func.air_current_inst = inst;

        try func.genInst(inst);

        if (debug.runtime_safety) {
            if (func.air_bookkeeping <= old_bookkeping_value) {
                debug.panic("missing `finishAir` call for AIR instruction `{}`", .{
                    func.air.instructions.items(.tag)[inst],
                });
            }
        }
    }
}

fn genInst(func: *Func, inst: Air.Inst.Index) error{ CodegenFail, OutOfMemory }!void {
    try func.ensureProcessDeathCapacity(Liveness.bpi);
    const tag = func.air.instructions.items(.tag)[inst];
    log.debug("lowering {} (%{})...", .{ tag, inst });
    return switch (tag) {
        .arg => func.airArg(inst),
        .add => func.airIntBinOp(inst, .add),
        .add_optimized => func.fail("TODO: handle {}", .{tag}),
        .addwrap => func.airIntBinOp(inst, .addwrap),
        .addwrap_optimized,
        .add_sat,
        => func.fail("TODO: handle {}", .{tag}),
        .sub => func.airIntBinOp(inst, .sub),
        .sub_optimized => func.fail("TODO: handle {}", .{tag}),
        .subwrap => func.airIntBinOp(inst, .subwrap),
        .subwrap_optimized,
        .sub_sat,
        .mul,
        .mul_optimized,
        .mulwrap,
        .mulwrap_optimized,
        .mul_sat,
        .div_float,
        .div_float_optimized,
        .div_trunc,
        .div_trunc_optimized,
        .div_floor,
        .div_floor_optimized,
        .div_exact,
        .div_exact_optimized,
        .rem,
        .rem_optimized,
        .mod,
        .mod_optimized,
        => func.fail("TODO: handle {}", .{tag}),
        .ptr_add => func.airPtrBinOp(inst, .ptr_add),
        .ptr_sub => func.airPtrBinOp(inst, .ptr_sub),
        // TODO: wait for https://github.com/ziglang/zig/issues/14039 to be implemented first
        //       before lowering the following two.
        .max,
        .min,
        .add_with_overflow,
        .sub_with_overflow,
        .mul_with_overflow,
        .shl_with_overflow,
        => func.fail("TODO: handle {}", .{tag}),
        .alloc => func.airAlloc(inst),
        .ret_ptr => func.airRetPtr(inst),
        // TODO: this tag should be called .asm
        .assembly => func.airAsm(inst),
        .bit_and,
        .bit_or,
        .shr,
        .shr_exact,
        .shl,
        .shl_exact,
        .shl_sat,
        .xor,
        => func.fail("TODO: handle {}", .{tag}),
        .not => func.fail("TODO: handle {}", .{tag}), //func.airNot(inst),
        .bitcast => func.airBitCast(inst),
        .block => func.fail("TODO: handle {}", .{tag}), //func.airBlock(inst),
        .loop => func.fail("TODO: handle {}", .{tag}), //func.airLoop(inst),
        .br => func.fail("TODO: handle {}", .{tag}),
        .breakpoint => func.airBreakpoint(inst),
        .ret_addr,
        .frame_addr,
        => func.fail("TODO: handle {}", .{tag}),
        .call => func.airCall(inst, .auto),
        .call_always_tail => func.airCall(inst, .always_tail),
        .call_never_tail => func.airCall(inst, .never_tail),
        .call_never_inline => func.airCall(inst, .never_inline),
        // TODO: can't we simply use compiler_rt for a lot of these?
        .clz,
        .ctz,
        .popcount, // TODO: for this one lib/compiler_rt/popcount.zig for example
        .byte_swap,
        .bit_reverse,

        .sqrt,
        .sin,
        .cos,
        .tan,
        .exp,
        .exp2,
        .log,
        .log2,
        .log10,
        .fabs,
        .floor,
        .ceil,
        .round,
        .trunc_float,
        .neg,
        .neg_optimized,

        .cmp_lt,
        .cmp_lt_optimized,
        .cmp_lte,
        .cmp_lte_optimized,
        .cmp_eq,
        .cmp_eq_optimized,
        .cmp_gte,
        .cmp_gte_optimized,
        .cmp_gt,
        .cmp_gt_optimized,
        .cmp_neq,
        .cmp_neq_optimized,
        .cmp_vector,
        .cmp_vector_optimized,
        => func.fail("TODO: handle {}", .{tag}),

        .cond_br => func.fail("TODO: handle {}", .{tag}), //func.airCondBr(inst),
        .switch_br,
        .@"try",
        .try_ptr,
        => func.fail("TODO: handle {}", .{tag}),
        .constant,
        .const_ty,
        => unreachable, // excluded from function bodies
        .dbg_stmt,
        .dbg_block_begin,
        .dbg_block_end,
        .dbg_inline_begin,
        .dbg_inline_end,
        .dbg_var_ptr,
        .dbg_var_val,
        // TODO
        => func.finishAir(inst, .none, &.{}),
        .is_null,
        .is_non_null,
        .is_null_ptr,
        .is_non_null_ptr,
        .is_err,
        .is_non_err,
        .is_err_ptr,
        .is_non_err_ptr,
        .bool_and,
        .bool_or,
        => func.fail("TODO: handle {}", .{tag}),
        .load => func.airLoad(inst),
        .ptrtoint => func.airPtrToInt(inst),
        .bool_to_int => func.airBoolToInt(inst),
        .ret => func.airRet(inst),
        .ret_load => func.airRetLoad(inst),
        .store => func.airStore(inst),
        .unreach => func.airUnreach(inst),
        .fptrunc,
        .fpext,
        => func.fail("TODO: handle {}", .{tag}),
        .intcast => func.airIntCast(inst),
        .trunc => func.airTrunc(inst),
        .optional_payload,
        .optional_payload_ptr,
        .optional_payload_ptr_set,
        .wrap_optional,
        .unwrap_errunion_payload,
        .unwrap_errunion_err,
        .unwrap_errunion_payload_ptr,
        .unwrap_errunion_err_ptr,
        .errunion_payload_ptr_set,
        .wrap_errunion_payload,
        .wrap_errunion_err,
        => func.fail("TODO: handle {}", .{tag}),
        .struct_field_ptr => func.airStructFieldPtr(inst),
        .struct_field_ptr_index_0 => func.airStructFieldPtrIndex(inst, 0),
        .struct_field_ptr_index_1 => func.airStructFieldPtrIndex(inst, 1),
        .struct_field_ptr_index_2 => func.airStructFieldPtrIndex(inst, 2),
        .struct_field_ptr_index_3 => func.airStructFieldPtrIndex(inst, 3),
        .struct_field_val => func.airStructFieldVal(inst),
        .set_union_tag,
        .get_union_tag,
        .slice,
        .slice_len,
        .slice_ptr,
        .ptr_slice_len_ptr,
        .ptr_slice_ptr_ptr,
        => func.fail("TODO: handle {}", .{tag}),
        .array_elem_val => func.airArrayElemVal(inst),
        .slice_elem_val,
        .slice_elem_ptr,
        => func.fail("TODO: handle {}", .{tag}),
        //.ptr_elem_val => func.airPtrElemVal(inst),
        //.ptr_elem_ptr => func.airPtrElemPtr(inst),
        .ptr_elem_val,
        .ptr_elem_ptr,
        => func.fail("TODO: handle {}", .{tag}),
        .array_to_slice,
        .float_to_int,
        .float_to_int_optimized,
        .int_to_float,

        .reduce,
        .reduce_optimized,
        .splat,
        .shuffle,
        .select,

        // TODO: wait for https://github.com/ziglang/zig/issues/14040 to be implemented first
        //       before lowering the following two.
        .memset,
        .memcpy,

        .cmpxchg_weak,
        .cmpxchg_strong,
        .fence,
        .atomic_load,
        .atomic_store_unordered,
        .atomic_store_monotonic,
        .atomic_store_release,
        .atomic_store_seq_cst,
        .atomic_rmw,

        .is_named_enum_value,

        .tag_name,

        .error_name,

        .error_set_has_value,

        .aggregate_init,

        .union_init,

        .prefetch,

        .mul_add,

        .field_parent_ptr,
        => func.fail("TODO: handle {}", .{tag}),

        .wasm_memory_size,
        .wasm_memory_grow,
        => unreachable, // we are not Wasm

        .cmp_lt_errors_len,

        .err_return_trace,

        .set_err_return_trace,

        .addrspace_cast,

        .save_err_return_trace_index,

        .vector_store_elem,

        // TODO: should these be supported at all?
        .c_va_arg,
        .c_va_copy,
        .c_va_end,
        .c_va_start,
        => func.fail("TODO: handle {}", .{tag}),
    };
}

fn airArg(func: *Func, inst: Air.Inst.Index) !void {
    const air_arg = func.air.instructions.items(.data)[inst].arg;
    const arg_ty = func.air.typeOf(air_arg.ty);
    const arg_src_index = air_arg.src_index;

    const mv = func.args[func.arg_index];
    func.arg_index += 1;

    switch (mv) {
        .reg => |reg| switch (reg) {
            .a => func.reg_a = inst,
            .x => func.reg_x = inst,
            .y => func.reg_y = inst,
        },
        else => {},
    }

    log.debug("lowering arg in {} of type {} with src index {}", .{ mv, arg_ty.tag(), arg_src_index });

    func.finishAir(inst, mv, &.{});
}

/// Emits code to perform the given integer binary operation on two operands of the same type, resulting in the same type.
fn airIntBinOp(func: *Func, inst: Air.Inst.Index, tag: Air.Inst.Tag) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;
    log.debug("bin_op: {}", .{bin_op});
    const lhs = try func.resolveInst(bin_op.lhs);
    const rhs = try func.resolveInst(bin_op.rhs);
    const lhs_ty = func.air.typeOf(bin_op.lhs);
    const rhs_ty = func.air.typeOf(bin_op.rhs);
    const res = try func.binOp(tag, lhs, rhs, lhs_ty, rhs_ty, null, inst);
    func.finishAir(inst, res, &.{ bin_op.lhs, bin_op.rhs });
}

/// Emits code to perform the given integer binary operation on two operands LHS (pointer) and RHS (operand) of possible different types,
/// resulting in the type of LHS (the pointer).
fn airPtrBinOp(func: *Func, inst: Air.Inst.Index, tag: Air.Inst.Tag) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const ty = func.air.getRefType(ty_pl.ty);
    const bin_op = func.air.extraData(Air.Bin, ty_pl.payload).data;
    const lhs = try func.resolveInst(bin_op.lhs);
    const rhs = try func.resolveInst(bin_op.rhs);
    const lhs_ty = func.air.typeOf(bin_op.lhs);
    const rhs_ty = func.air.typeOf(bin_op.rhs);
    assert(ty.eql(lhs_ty, func.getMod()));
    const res = try func.binOp(tag, lhs, rhs, lhs_ty, rhs_ty, null, inst);
    return func.finishAir(inst, res, &.{ bin_op.lhs, bin_op.rhs });
}

/// Emits code to perform the given binary operation on two operands of possibly differing types, resulting in a value of either of the given types.
/// A binary operation is a rule for combining two operands to produce another value.
fn binOp(
    func: *Func,
    tag: Air.Inst.Tag,
    lhs: MV,
    rhs: MV,
    lhs_ty: Type,
    rhs_ty: Type,
    // desired result location
    maybe_dst: ?MV,
    // the result owner
    maybe_inst: ?Air.Inst.Index,
) !MV {
    switch (tag) {
        .add,
        .addwrap,
        .add_sat,
        .sub,
        .subwrap,
        .sub_sat,
        => {
            assert(lhs_ty.eql(rhs_ty, func.getMod()));
            const ty = lhs_ty;
            switch (ty.zigTypeTag()) {
                .Int => {
                    return try func.intBinOp(lhs, rhs, ty, tag, maybe_dst, maybe_inst);
                },
                .Float => return func.fail("TODO: support floats", .{}),
                .Vector => return func.fail("TODO: support vectors", .{}),
                else => unreachable,
            }
        },
        .ptr_add,
        .ptr_sub,
        => {
            switch (lhs_ty.zigTypeTag()) {
                .Pointer => {
                    const ptr_ty = lhs_ty;
                    const child_ty = switch (ptr_ty.ptrSize()) {
                        .One => ptr_ty.childType().childType(), // ptr to array, so get array element type
                        else => ptr_ty.childType(),
                    };
                    const child_size = func.getByteSize(child_ty).?;
                    if (child_size == 1) {
                        const op_tag: Air.Inst.Tag = switch (tag) {
                            .ptr_add => .add,
                            .ptr_sub => .sub,
                            else => unreachable,
                        };
                        // we will simply treat both LHS and and RHS as usize.
                        // this is fine because wrapping past the end of the address space is undefined behavior.
                        return try func.binOp(op_tag, lhs, rhs, Type.usize, Type.usize, maybe_dst, maybe_inst);
                    } else {
                        return func.fail("TODO: implement indexing lists with elements with size > 1 by multiplying", .{});
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

/// Emits code to perform the given binary operation on two operands of the same size, resulting in the same size.
/// Supports the native 8-bit integer size as well as any other size.
/// Commutativeness may be taken into account.
fn intBinOp(
    func: *Func,
    lhs: MV,
    rhs: MV,
    ty: Type,
    op: Air.Inst.Tag,
    // desired result location
    maybe_dst: ?MV,
    // the result owner
    maybe_inst: ?Air.Inst.Index,
) !MV {
    assert(ty.zigTypeTag() == .Int);

    const byte_size = func.getByteSize(ty).?;
    assert(byte_size >= 1);

    const res = switch (op) {
        // this assembly shows the addition of two words:
        // ```
        // lhs: ; $1150
        //     .byte $50, $11 ; little-endian
        // rhs: ; $2260
        //     .byte $60, $22 ; little-endian
        // res:
        //     .res 2 ; reserve bytes for output
        //
        // add:
        //     clc ; carry flag might be set
        //     cld ; decimal flag might be set
        //     lda lhs + 0
        //     adc rhs + 0
        //     ; A = $B0
        //     sta res + 0
        //
        //     lda lhs + 1
        //     adc rhs + 1 ; if we have a carry after this we have an overflow
        //     ; A = $33
        //     sta res + 1
        //
        //     ; at this point `res` now contains the result $1150 + $2260 = $33B0
        //     rts
        // ```
        .add, .addwrap, .add_sat => res: {
            try func.decimal.clear();
            try func.carry.clear();
            defer func.carry.state = .unknown;

            var dst = maybe_dst;
            var reg_a: ?MV = null;

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // we take advantage of the fact that addition is commutative,
                // meaning we can add the operands in any order,
                // so we will check whether one of them is already in the accumulator.
                var values = std.BoundedArray(MV, 2){};
                values.appendAssumeCapacity(lhs.index(i));
                values.appendAssumeCapacity(rhs.index(i));
                for (values.constSlice()) |val, val_i| {
                    // is either LHS or RHS already in the accumulator?
                    if (val == .reg and val.reg == .a) {
                        if (dst == null and byte_size == 1) {
                            // we can do this if LHS or RHS is the same as the result location
                            reg_a = func.takeReg(.a, maybe_inst);
                            try func.addInstAny("adc", values.pop());
                            break :res reg_a.?;
                        }
                        _ = values.swapRemove(val_i);
                        break;
                    }
                } else {
                    // neither LHS nor RHS were already in the accumulator,
                    // so get either of them in the accumulator.
                    if (reg_a == null)
                        reg_a = try func.freeReg(.a, maybe_inst);
                    try func.trans(values.pop(), reg_a.?, Type.u8, Type.u8);
                }
                try func.addInstAny("adc", values.pop());
                assert(values.len == 0);
                if (dst == null)
                    dst = try func.allocMem(ty);
                try func.trans(reg_a.?, dst.?.index(i), Type.u8, Type.u8);
            }
            break :res dst.?;
        },
        .sub, .subwrap, .sub_sat => res: {
            try func.decimal.clear();
            // for subtraction we have to do the opposite of what we do for addition:
            // we set carry, which for subtraction means we *clear borrow*.
            try func.carry.set();
            defer func.carry.state = .unknown;

            if (lhs == .reg and lhs.reg == .a) {
                if (maybe_dst == null and byte_size == 1) {
                    try func.addInstAny("sbc", rhs);
                    const reg_a = func.takeReg(.a, maybe_inst);
                    break :res reg_a;
                }
            }

            const dst = maybe_dst orelse try func.allocMem(ty);
            const reg_a = try func.freeReg(.a, maybe_inst);

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            // TODO: do these loop at runtime at a certain threshold which depends on build mode
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // get the LHS byte into the accumulator and subtract the RHS byte from it
                try func.trans(lhs.index(i), reg_a, Type.u8, Type.u8);
                try func.addInstAny("sbc", rhs.index(i));
                try func.trans(reg_a, dst.index(i), Type.u8, Type.u8);
            }

            break :res dst;
        },
        else => unreachable,
    };
    switch (op) {
        .add_sat, .sub_sat => {
            if (true) panic("TODO: actually implement saturation", .{});
            assert(func.getByteSize(ty).? <= 127);
            try func.addInstRel(.bcc_rel, undefined);
            const bcc = func.currentInst();
            const size_before = func.getSize();
            switch (op) {
                //.add_sat => func.trans(ty.maxInt(func.getAllocator(), func.getTarget()), res, ty, ty),
                //.sub_sat => func.trans(0, res, ty, ty),
                else => unreachable,
            }
            const size_after = func.getSize();
            const size = size_after - size_before;
            bcc.data = size;
        },
        else => {},
    }
    log.debug("intBinOp res: {}", .{res});
    return res;
}

fn airAlloc(func: *Func, inst: Air.Inst.Index) !void {
    // We have two options here:
    //
    // 1. Allocate technically correct stack-local memory by using only the hardware stack and registers,
    //    which only gives us a bit more than 256 bytes per function.
    //    To do this, we would use PHA and PLA to push data and store/load bytes using specific offsets into the second page of memory,
    //    using TSX and TXS to transfer from and to the stack pointer.
    //    The system itself would "clean up" the memory for us.
    //    This is how all other codegen backends do this and it works for them because of how big their stack is.
    //    Choosing this option means we do not have access to the zero page, either, which is crucial for
    //    efficient code generation and necessary for dereferencing pointers (TODO: confirm this).
    //    If we chose this option, we would probably only give access to the zero page and absolute memory
    //    through `addrspace`s.
    // 2. Allocate from the zero page, absolute memory, and registers,
    //    which gives us access to all memory at the cost that it is harder to manage
    //    and we need to keep memory state intact across all function codegens and we have to clean it all up.
    //    This creates a unique problem no other codegen backend has.
    //    TODO: talk about how heap memory allocation plays into this.
    //          how does one write a heap memory allocator for a 6502 target?
    //          1. use the OS? does the C64 or any OS have some CBM kernal routine to allocate heap memory?
    //          2. provide a way for the user to reserve a specific number of bytes that it can use in its heap.
    //             for example, how do I tell the backend I want 0x0800 bytes of the 64 KiB available ones for my heap?
    //             would we need a new builtin for this? ref: @wasmMemorySize and @wasmMemoryGrow
    //
    // We choose the second option.
    const ptr_ty = func.air.instructions.items(.data)[inst].ty;
    const res = try func.allocMem(ptr_ty.childType());
    log.debug("airAlloc: allocated ptr: {}", .{res});
    func.finishAir(inst, res, &.{});
}

fn airRetPtr(func: *Func, inst: Air.Inst.Index) !void {
    // this is equivalent to airAlloc if we do not choose to pass by reference
    const ptr_ty = func.air.instructions.items(.data)[inst].ty;
    const res = try func.allocMem(ptr_ty.childType());
    log.debug("airRetPtr: allocated ptr: {}", .{res});
    func.finishAir(inst, res, &.{});
}

/// Allocates addressable memory capable of storing a value of the given type.
fn allocMem(func: *Func, ty: Type) !MV {
    const byte_size = func.getByteSize(ty) orelse return func.fail("type `{}` too big to fit in address space", .{ty.fmt(func.getMod())});
    const val = try func.memory.alloc(byte_size);
    log.debug("allocated to {}", .{val});
    return val;
}

/// Emits code to write a value to a pointer.
fn airStore(func: *Func, inst: Air.Inst.Index) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;

    const ptr = try func.resolveInst(bin_op.lhs);
    const ptr_ty = func.air.typeOf(bin_op.lhs);
    const val = try func.resolveInst(bin_op.rhs);
    const val_ty = func.air.typeOf(bin_op.rhs);

    log.debug("storing {} ({}) at {} ({})", .{ val, val_ty.tag(), ptr, ptr_ty.tag() });

    //try func.store(val, ptr, val_ty, ptr_ty);
    try func.trans(val, ptr, val_ty, ptr_ty.childType());

    func.finishAir(inst, .none, &.{ bin_op.lhs, bin_op.rhs });
}

/// Emits code to read a value from a pointer.
fn airLoad(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const val_ty = func.air.getRefType(ty_op.ty);
    const ptr = try func.resolveInst(ty_op.operand);
    const ptr_ty = func.air.typeOf(ty_op.operand);
    assert(val_ty.eql(ptr_ty.childType(), func.getMod()));
    const dst = try func.allocMem(ptr_ty.childType());
    try func.trans(ptr, dst, ptr_ty.childType(), val_ty);
    func.finishAir(inst, dst, &.{ty_op.operand});
}

/// Transfers from source to destination.
fn trans(func: *Func, src: MV, dst: MV, src_ty: Type, dst_ty: Type) !void {
    const src_byte_size = func.getByteSize(src_ty).?;
    const dst_byte_size = func.getByteSize(dst_ty).?;
    log.debug("transfering {} ({} bytes) of type {} to {} ({} bytes) of type {}", .{ src, src_byte_size, src_ty.tag(), dst, dst_byte_size, dst_ty.tag() });
    assert(src_byte_size <= dst_byte_size);
    if (src_byte_size == 0)
        return;
    switch (src) {
        .none => unreachable,
        .imm => |src_imm| {
            assert(src_byte_size == 1);
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(dst_byte_size == 1);
                    switch (dst_reg) {
                        .a => try func.addInstImm(.lda_imm, src_imm),
                        .x => try func.addInstImm(.ldx_imm, src_imm),
                        .y => try func.addInstImm(.ldy_imm, src_imm),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    // TODO: make this A reg temporary saving thing into a function
                    //       and check if A is already free
                    assert(dst_byte_size == 1); // TODO: could be more
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    try func.addInstImm(.lda_imm, src_imm);
                    try func.addInstMem("sta", dst);
                    try func.addInstImpl(.pla_impl);
                },
                .abs_imm => unreachable, // TODO
                .zp_imm => unreachable, // TODO
            }
        },
        .reg => |src_reg| {
            assert(src_byte_size == 1);
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(dst_byte_size == 1);
                    switch (src_reg) {
                        .a => {
                            switch (dst_reg) {
                                .a => {},
                                .x => try func.addInstImpl(.tax_impl),
                                .y => try func.addInstImpl(.tay_impl),
                            }
                        },
                        else => panic("TODO: handle {}", .{src_reg}),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    assert(dst_byte_size == 1); // TODO: could be more
                    switch (src_reg) {
                        .a => try func.addInstMem("sta", dst),
                        .x => try func.addInstMem("stx", dst),
                        else => panic("TODO: handle {}", .{src_reg}),
                    }
                },
                .abs_imm => unreachable, // TODO
                .zp_imm => unreachable, // TODO
            }
        },
        .zp, .abs, .abs_unresolved => {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |reg| {
                    assert(dst_byte_size == 1);
                    switch (reg) {
                        .a => try func.addInstMem("lda", src),
                        .x => try func.addInstMem("ldx", src),
                        .y => try func.addInstMem("ldy", src),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    // TODO: make this A reg temporary saving thing into a function
                    //       and check if A is already free
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    var i: u16 = 0;
                    while (i < src_byte_size) : (i += 1) {
                        try func.addInstMem("lda", src.index(i));
                        try func.addInstMem("sta", dst.index(i));
                    }
                    try func.addInstImpl(.pla_impl);
                },
                .abs_imm => unreachable, // TODO
                .zp_imm => unreachable, // TODO
            }
        },
        .abs_imm => unreachable, // TODO
        .zp_imm => unreachable, // TODO
    }
}

fn airIntCast(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const dst_ty = func.air.getRefType(ty_op.ty);
    const operand = try func.resolveInst(ty_op.operand);
    const operand_ty = func.air.typeOf(ty_op.operand);

    // TODO: take signedness into account?
    const target = func.getTarget();
    const operand_info = operand_ty.intInfo(target);
    const dst_info = dst_ty.intInfo(target);
    const res: MV = res: {
        if (operand_info.bits == dst_info.bits) {
            break :res operand;
        }

        const operand_size = func.getByteSize(operand_ty).?;
        const dst_byte_size = func.getByteSize(dst_ty).?;

        if (operand_size == 1 and dst_byte_size == 2) {
            // TODO: func.memory.realloc if possible so that we can reuse the existing allocation and just add a zero byte?
            const res = try func.allocMem(Type.usize);
            try func.trans(operand, res.index(0), Type.u8, Type.u8);
            try func.trans(.{ .imm = 0 }, res.index(1), Type.u8, Type.u8);
            break :res res;
        } else {
            @panic("TODO");
        }
    };

    return func.finishAir(inst, res, &.{ty_op.operand});
}

fn airTrunc(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const dst_ty = func.air.getRefType(ty_op.ty);
    const operand = try func.resolveInst(ty_op.operand);
    const operand_ty = func.air.typeOf(ty_op.operand);

    log.debug("truncating type {} to {}", .{ operand_ty.tag(), dst_ty.tag() });

    // TODO: handle bit-wise truncation in bytes by clearing bits

    // nothing to do; at the type system level the operand is now the wanted type
    // and so next time we take bits from this operand,
    // we will only take that many bits that the type says it has.
    func.finishAir(inst, operand, &.{ty_op.operand});
}

fn airStructFieldPtr(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const extra = func.air.extraData(Air.StructField, ty_pl.payload).data;
    const result = try func.structFieldPtr(inst, extra.struct_operand, extra.field_index);
    return func.finishAir(inst, result, &.{extra.struct_operand});
}

fn airStructFieldPtrIndex(func: *Func, inst: Air.Inst.Index, index: u32) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const result = try func.structFieldPtr(inst, ty_op.operand, index);
    return func.finishAir(inst, result, &.{ty_op.operand});
}

fn structFieldPtr(func: *Func, inst: Air.Inst.Index, operand: Air.Inst.Ref, index: u32) !MV {
    _ = inst;

    const mv = try func.resolveInst(operand);
    const ptr_ty = func.air.typeOf(operand);
    const struct_ty = ptr_ty.childType();
    const struct_field_offset = @intCast(u16, struct_ty.structFieldOffset(index, func.getTarget()));

    log.debug("structFieldPtr mv: {}, index: {}, struct_field_offset: {}", .{ mv, index, struct_field_offset });

    const dst = switch (mv) {
        .abs => |addr| .{ .abs = addr + struct_field_offset },
        else => panic("TODO: handle {}", .{mv}),
    };

    return dst;
}

fn airStructFieldVal(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const extra = func.air.extraData(Air.StructField, ty_pl.payload).data;
    const operand = extra.struct_operand;
    const index = extra.field_index;

    const mv = try func.resolveInst(operand);
    const struct_ty = func.air.typeOf(operand);
    const struct_field_offset = @intCast(u16, struct_ty.structFieldOffset(index, func.getTarget()));

    log.debug("struct_field_offset for ty {}: {}", .{ struct_ty.tag(), struct_field_offset });

    const result = switch (mv) {
        .zp, .abs, .abs_unresolved => mv.index(struct_field_offset),
        else => panic("TODO: handle {}", .{mv}),
    };

    return func.finishAir(inst, result, &.{extra.struct_operand});
}

fn airArrayElemVal(func: *Func, inst: Air.Inst.Index) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;
    const array = try func.resolveInst(bin_op.lhs);
    const array_ty = func.air.typeOf(bin_op.lhs);

    const child_ty = array_ty.childType();
    const child_byte_size = func.getByteSize(child_ty).?;

    const index = try func.resolveInst(bin_op.rhs);
    const index_ty = func.air.typeOf(bin_op.rhs);
    _ = index_ty;

    log.debug("array: {}, index: {}", .{ array, index });

    const res: MV = res: {
        switch (array) {
            .none => unreachable,
            .imm => panic("TODO: handle {}", .{array}),
            .reg => panic("TODO: handle {}", .{array}),
            .zp => panic("TODO: handle {}", .{array}),
            .abs, .abs_unresolved => {
                // this assembly shows indexing of an array in absolute memory using a byte:
                // ```
                // ; our array (assume this will be in absolute memory)
                // hello: .byte 'H', 'E', 'L', 'L', 'O'
                //
                // main:
                //     ldx #1 ; the index
                //     lda hello, x
                //     ; now A contains 'E'
                //     rts
                // ```
                break :res switch (index) {
                    .none => unreachable,
                    .imm => |imm| {
                        try func.addInstMem("lda", array.index(imm * child_byte_size));
                        break :res .{ .reg = .a };
                    },
                    .reg => |reg| switch (reg) {
                        .a => {
                            assert(child_byte_size == 1); // TODO (runtime mul)
                            try func.addInstImpl(.tax_impl);
                            try func.addInstMem("lda_x", array);
                            break :res .{ .reg = .a };
                        },
                        .x => unreachable, // TODO
                        .y => unreachable, // TODO
                    },
                    .zp => |addr| {
                        assert(child_byte_size == 1); // TODO (runtime mul)
                        try func.addInstZp(.ldx_zp, .{ .zp = addr });
                        try func.addInstMem("lda_x", array);
                        break :res .{ .reg = .a };
                    },
                    .abs, .abs_unresolved => {
                        // this assembly shows array indexing using a word:
                        // ```
                        // ; our array
                        // hello: .byte 'H', 'E', 'L', 'L', 'O'
                        // ; our index (1)
                        // index: .byte 1, 0
                        //
                        // main:
                        //     clc ; carry flag might be set
                        //     cld ; decimal flag might be set
                        //     ; get the two-byte address somewhere into the zero page
                        //     ; lo byte
                        //     lda index + 0
                        //     adc #<hello ; add `hello` lo byte
                        //     sta $00
                        //     ; hi byte
                        //     lda index + 1
                        //     adc #>hello ; add `hello` hi byte
                        //     sta $01
                        //
                        //     ; now dereference the pointer and get the byte
                        //     ldx #0       ; alternatively: ldy #0
                        //     lda ($00, x) ;                lda ($00), y
                        //     ; now A contains 'E'
                        //     rts
                        // ```
                        assert(child_byte_size == 1); // TODO (runtime mul)
                        const res = .{ .zp = (func.memory.zp_res orelse return func.fail("unable to dereference pointer due to zero page shortage", .{}))[0] };
                        const addr = .{ .abs_imm = array.abs }; // TODO
                        const res2 = try func.intBinOp(addr, index, Type.usize, .add, res, null);
                        assert(res.zp == res2.zp);
                        _ = try func.freeReg(.x, null);
                        try func.addInstImm(.ldx_imm, 0); // TODO: this is why it'd be worth tracking register contents and all other memory
                        try func.addInstZp(.lda_x_ind_zp, res);
                        break :res .{ .reg = .a };
                    },
                    .abs_imm => unreachable, // TODO
                    .zp_imm => unreachable, // TODO
                };
            },
            .abs_imm => unreachable, // TODO
            .zp_imm => unreachable, // TODO
        }
    };

    log.debug("leaving airArrayElemVal with {} res", .{res});

    return func.finishAir(inst, res, &.{ bin_op.lhs, bin_op.rhs });
}

/// Emits code to read an element value from a pointer using an index.
fn airPtrElemVal(func: *Func, inst: Air.Inst.Index) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;
    const ptr = bin_op.lhs;
    const index = bin_op.rhs;
    const ptr_ty = func.air.typeOf(ptr);
    const res = try func.ptrElemVal(ptr, index, ptr_ty, inst);
    return func.finishAir(inst, res, &.{ ptr, index });
}

/// Emits code to create a new pointer from a pointer and an index.
fn airPtrElemPtr(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const extra = func.air.extraData(Air.Bin, ty_pl.payload).data;
    const ptr = extra.lhs;
    const index = extra.rhs;
    const ptr_ty = func.air.typeOf(ptr);
    const new_ptr = try func.ptrElemPtr(ptr, index, ptr_ty, inst);
    return func.finishAir(inst, new_ptr, &.{ ptr, index });
}

/// Indexes into a pointer and returns that pointer.
fn ptrElemPtr(
    func: *Func,
    ptr: Air.Inst.Ref,
    index: Air.Inst.Ref,
    ptr_ty: Type,
    inst: Air.Inst.Index,
) !MV {
    const ptr_mv = try func.resolveInst(ptr);
    const index_mv = try func.resolveInst(index);
    const new_ptr = try func.binOp(.ptr_add, ptr_mv, index_mv, ptr_ty, Type.usize, null, inst);
    return new_ptr;
}

fn ptrElemVal(
    func: *Func,
    ptr: Air.Inst.Ref,
    index: Air.Inst.Ref,
    ptr_ty: Type,
    inst: Air.Inst.Index,
) !MV {
    const ptr_mv = try func.resolveInst(ptr);
    const index_mv = try func.resolveInst(index);
    const addr = try func.binOp(.ptr_add, ptr_mv, index_mv, ptr_ty, Type.usize, null, inst);
    const child_ty = ptr_ty.childType();
    const dst = try func.allocMem(child_ty);
    try func.trans(addr, dst, child_ty, child_ty);
    return dst;
}

fn airBitCast(func: *Func, inst: Air.Inst.Index) !void {
    // this is only a change at the type system level
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const dst_ty = func.air.getRefType(ty_op.ty);
    _ = dst_ty;
    const res = try func.resolveInst(ty_op.operand);
    log.debug("airBitCast: res: {} operand: {}", .{ res, ty_op.operand });
    func.finishAir(inst, res, &.{ty_op.operand});
}

fn airPtrToInt(func: *Func, inst: Air.Inst.Index) !void {
    // nothing to do because pointers are represented as integers, anyway.
    // this is only a change at the type system level.
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const op = try func.resolveInst(un_op);
    const res = op;
    func.finishAir(inst, res, &.{un_op});
}

fn airBoolToInt(func: *Func, inst: Air.Inst.Index) !void {
    // this is only a change at the type system level
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const op = try func.resolveInst(un_op);
    const res = op;
    return func.finishAir(inst, res, &.{un_op});
}

/// Emits code to return a value from the current function.
fn airRet(func: *Func, inst: Air.Inst.Index) !void {
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const val = try func.resolveInst(un_op);
    const ret_ty = func.getType().fnReturnType();
    //try func.store(val, func.ret_val, val_ty, ret_ty);
    try func.trans(val, func.ret_val, ret_ty, ret_ty);
    try func.addInstImpl(.rts_impl);
    func.finishAir(inst, .none, &.{un_op});
}

fn airRetLoad(func: *Func, inst: Air.Inst.Index) !void {
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const ptr = try func.resolveInst(un_op);
    const ptr_ty = func.air.typeOf(un_op);
    const ret_ty = func.getType().fnReturnType();
    const dst = try func.allocMem(ptr_ty.childType());
    try func.trans(ptr, dst, ptr_ty, ptr_ty);
    const val = dst;
    try func.trans(val, func.ret_val, ret_ty, ret_ty);
    try func.addInstImpl(.rts_impl);
    return func.finishAir(inst, .none, &.{un_op});
}

fn airLoop(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const res_ty = func.air.getRefType(ty_pl.ty);
    assert(res_ty.tag() == .noreturn);
    const extra = func.air.extraData(Air.Block, ty_pl.payload);
    const body = func.air.extra[extra.end..][0..extra.data.body_len];

    const size_before = func.getSize();
    try func.genBody(body);
    const size_after = func.getSize();
    const size = size_after - size_before;

    log.debug("before: {}, after: {}, now: {}", .{ size_before, size_after, size });

    // TODO: maybe rename addInstInternal to addInstDirect or something and still allow access
    try func.addInstInternal(.jmp_abs, .{ .abs = .{ .current = .{
        .decl_index = func.getDeclIndex(),
        .offset = size,
    } } });
    func.finishAir(inst, .none, &.{});
}

fn airBreakpoint(func: *Func, inst: Air.Inst.Index) !void {
    // examples of the behavior of BRK:
    // * on the C64 it clears the screen and resets.
    // * on the C128 it prints "BREAK" followed by the values of
    //   PC (Program Counter), SR (Status Register),
    //   AC (ACcumulator), XR (X Register), YR (Y Register),
    //   and SP (Stack Pointer)
    try func.addInstImpl(.brk_impl);
    // TODO: some OSs (like the SOS written for the Apple III) apparently use BRK for system calls and the byte following BRK is the syscall number.
    //       investigate that, interrupts, and the necessity of this NOP more. for now we include it just to be sure because BRK advances the PC by 2.
    //       https://retrocomputing.stackexchange.com/questions/12291/what-are-uses-of-the-byte-after-brk-instruction-on-6502
    try func.addInstImpl(.nop_impl);
    func.finishAir(inst, .none, &.{});
}

/// Returns the size of this function up to this point.
fn getSize(func: Func) u16 {
    var byte_size: u16 = 0;
    var i: u16 = 0;
    while (i < func.mir_instructions.len) : (i += 1) {
        const inst = Mir.Inst{
            .tag = func.mir_instructions.items(.tag)[i],
            .data = func.mir_instructions.items(.data)[i],
        };
        byte_size += inst.getByteSize();
    }
    return byte_size;
}

fn getProgramCounter(func: Func) u16 {
    if (func.bin_file.cast(link.File.Prg)) |prg| {
        const load_address = prg.getLoadAddress();
        const program_size = @intCast(u16, prg.header.len) + func.getProgramSize();
        return load_address + program_size;
    } else unreachable;
}

/// Emits code to pass arguments to a function and change the program counter to continue at the function's point.
fn airCall(func: *Func, inst: Air.Inst.Index, modifier: std.builtin.CallModifier) !void {
    switch (modifier) {
        .auto, .never_inline => {},
        .compile_time => unreachable,
        .async_kw,
        .never_tail,
        .no_async,
        .always_tail,
        .always_inline,
        => return func.fail("unsupported call modifier {}", .{modifier}),
    }

    const pl_op = func.air.instructions.items(.data)[inst].pl_op;
    const extra = func.air.extraData(Air.Call, pl_op.payload);
    const args = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra.end..][0..extra.data.args_len]);
    const callee = pl_op.operand;
    const callee_ty = func.air.typeOf(callee);

    const fn_ty = switch (callee_ty.zigTypeTag()) {
        .Fn => callee_ty,
        .Pointer => callee_ty.childType(),
        else => unreachable,
    };

    log.debug("callee: {}, callee_ty: {}, args: {any}", .{ callee, fn_ty.tag(), args });

    var call_values = try func.resolveCallingConventionValues(fn_ty);
    defer call_values.deinit(func.getAllocator());

    for (call_values.args) |dst, i| {
        const src = try func.resolveInst(args[i]);
        const arg_ty = func.air.typeOf(args[i]);
        log.debug("CC: {} -> {}", .{ src, dst });
        try func.trans(src, dst, arg_ty, arg_ty);
    }

    if (func.air.value(callee)) |fn_val| {
        if (fn_val.castTag(.function)) |fn_pl| {
            log.debug("calling {s}...", .{func.getMod().declPtr(fn_pl.data.owner_decl).name});
            if (func.bin_file.cast(link.File.Prg)) |prg| {
                const blk_i = try prg.recordDecl(fn_pl.data.owner_decl);
                // TODO: maybe rename addInstInternal to addInstDirect or something and still allow access
                try func.addInstInternal(.jsr_abs, .{ .abs = .{ .unresolved = .{ .blk_i = blk_i } } });
            } else unreachable;
        } else if (fn_val.castTag(.extern_fn)) |_| {
            return func.fail("extern functions not supported", .{});
        } else if (fn_val.castTag(.decl_ref)) |_| {
            return func.fail("TODO implement calling bitcasted functions", .{});
        } else if (fn_val.castTag(.int_u64)) |int| {
            // TODO: maybe rename addInstInternal to addInstDirect or something and still allow access
            try func.addInstInternal(.jsr_abs, .{ .abs = .{ .imm = @intCast(u16, int.data) } });
        } else unreachable;
    } else {
        assert(callee_ty.zigTypeTag() == .Pointer);
        panic("TODO: handle {}", .{func.air.value(callee).?.tag()});
    }

    const res = call_values.return_value;
    var big_tomb = try func.iterateBigTomb(inst, args.len + 1);
    big_tomb.feed(pl_op.operand, res);
    for (args) |arg| big_tomb.feed(arg, res);
    return big_tomb.finishAir(res);

    // TODO from: https://www.nesdev.org/wiki/6502_assembly_optimisations
    //            avoid a JSR + RTS chain and replace it with JMP.
    //            add fix-up optimization passes that check for patterns or do it right away?
}

fn airCondBr(func: *Func, inst: Air.Inst.Index) !void {
    const pl_op = func.air.instructions.items(.data)[inst].pl_op;
    const cond = try func.resolveInst(pl_op.operand);
    const extra = func.air.extraData(Air.CondBr, pl_op.payload);
    const then_body = func.air.extra[extra.end..][0..extra.data.then_body_len];
    const else_body = func.air.extra[extra.end + then_body.len ..][0..extra.data.else_body_len];
    const liveness_condbr = func.liveness.getCondBr(inst);
    _ = liveness_condbr;

    // this assembly shows conditional execution:
    // ```
    // lda #5
    // cmp #5 ; does A equal 5? (yes)
    //
    // main:
    //   beq
    // then:
    //   lda #0
    // else:
    //   rts
    // ```

    switch (cond) {
        .none => unreachable,
        .imm => |imm| {
            if (imm == 1) {
                try func.genBody(then_body);
            } else if (imm == 0) {
                try func.genBody(else_body);
            } else unreachable;
        },
        else => panic("TODO: implement {} cond", .{cond}),
    }
}

/// Emits code based on 6502-specific input.
fn airAsm(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const extra = func.air.extraData(Air.Asm, ty_pl.payload);
    var extra_i = extra.end;
    const outputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.outputs_len]);
    assert(outputs.len == 0); // TODO: support outputting specific registers into variables (so, the other way around)
    extra_i += outputs.len;
    const inputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.inputs_len]);
    extra_i += inputs.len;

    assert(extra.data.clobbers_len() == 0); // TODO

    //const dies = !extra.data.is_volatile() and func.liveness.isUnused(inst);
    //const res: MV = if (dies)
    //    .none
    //else res: {
    const res: MV = res: {
        for (inputs) |input| {
            const input_bytes = mem.sliceAsBytes(func.air.extra[extra_i..]);
            const constraint = mem.sliceTo(input_bytes, 0);
            const name = mem.sliceTo(input_bytes[constraint.len + 1 ..], 0);

            // This equation accounts for the fact that even if we have exactly 4 bytes
            // for the string, we still use the next u32 for the null terminator.
            extra_i += (constraint.len + name.len + (2 + 3)) / 4;

            if (constraint.len < 3 or constraint[0] != '{' or constraint[constraint.len - 1] != '}') {
                return func.fail("unknown `asm` input constraint: \"{s}\"", .{constraint});
            }
            const reg_name = constraint[1 .. constraint.len - 1];
            const reg = Register.parse(reg_name) orelse {
                return func.fail("unknown register \"{s}\"", .{reg_name});
            };
            const input_val = try func.resolveInst(input);
            // TODO: support bit sizes <= 8 and support `comptime_int`s?
            const ty = func.air.typeOf(input);
            if (func.getByteSize(ty).? != 1)
                return func.fail("unable to load non-8-bit-sized into {c} register", .{std.ascii.toUpper(reg_name[0])});
            const reg_a = try func.freeReg(reg, inst);
            try func.trans(input_val, reg_a, ty, Type.u8);
        }

        const asm_source = mem.sliceAsBytes(func.air.extra[extra_i..])[0..extra.data.source_len];
        log.debug("asm_source.len: {}", .{asm_source.len});
        if (asm_source.len != 0)
            log.debug("asm source:\n```\n{s}\n```", .{asm_source});
        if (mem.eql(u8, asm_source, "rts")) {
            try func.addInstImpl(.rts_impl);
        } else if (mem.eql(u8, asm_source, "nop")) {
            try func.addInstImpl(.nop_impl);
        } else {
            assert(asm_source.len == 0);
        }

        break :res .none;
    };

    var big_tomb = try func.iterateBigTomb(inst, outputs.len + inputs.len);
    for (outputs) |output| {
        if (output == .none) continue;
        big_tomb.feed(output, res);
    }
    for (inputs) |input| {
        big_tomb.feed(input, res);
    }
    return big_tomb.finishAir(res);
}

fn airUnreach(func: *Func, inst: Air.Inst.Index) !void {
    func.finishAir(inst, .none, &.{});
}

fn lowerConstant(func: *Func, const_val: Value, ty: Type) !MV {
    var val = const_val;
    if (val.castTag(.runtime_value)) |sub_value| {
        val = sub_value.data;
    }

    if (val.castTag(.decl_ref)) |decl_ref| {
        const decl_index = decl_ref.data;
        return func.lowerDeclRef(val, ty, decl_index);
    }
    if (val.castTag(.decl_ref_mut)) |decl_ref_mut| {
        const decl_index = decl_ref_mut.data.decl_index;
        return func.lowerDeclRef(val, ty, decl_index);
    }

    // try lowering the constant to a Memory Value for efficiency.
    // if it does not fit, fall through and let the other logic handle it.
    const target = func.getTarget();
    switch (ty.zigTypeTag()) {
        .Void => return MV{ .none = {} },
        .Int => {
            const int_info = ty.intInfo(target);
            if (int_info.bits <= 8) {
                switch (int_info.signedness) {
                    .signed => return MV{ .imm = @bitCast(u8, @intCast(i8, val.toSignedInt(target))) },
                    .unsigned => return MV{ .imm = @intCast(u8, val.toUnsignedInt(target)) },
                }
            }
        },
        .Bool => return MV{ .imm = @boolToInt(val.toBool()) },
        .Pointer => switch (ty.ptrSize()) {
            .Slice => {},
            else => switch (val.tag()) {
                .int_u64, .one, .zero, .null_value => {
                    return MV{ .abs = @intCast(u16, val.toUnsignedInt(target)) };
                },
                else => {},
            },
        },

        .AnyFrame,
        .Frame,
        .ErrorSet,
        .Vector,
        .ErrorUnion,
        .Optional,
        => panic("TODO: handle {}", .{ty.zigTypeTag()}),

        .ComptimeInt => unreachable,
        .ComptimeFloat => unreachable,
        .Type => unreachable,
        .EnumLiteral => unreachable,
        .NoReturn => unreachable,
        .Undefined => unreachable, // TODO: MV.undef?
        .Null => unreachable,
        .Opaque => unreachable,

        .Array,
        .Float,
        .Struct,
        .Fn,
        .Enum,
        .Union,
        => {},
    }

    // it is not representable as a Memory Value so emit it to a read-only section in the binary
    return try func.lowerUnnamedConst(val, ty);
}

fn lowerUnnamedConst(func: *Func, val: Value, ty: Type) !MV {
    log.debug("lowerUnnamedConst: ty = {}, val = {}", .{ val.fmtValue(ty, func.getMod()), ty.fmt(func.getMod()) });
    const blk_i = @intCast(
        u16,
        func.bin_file.lowerUnnamedConst(.{ .ty = ty, .val = val }, func.getDeclIndex()) catch |err| {
            return func.fail("lowering unnamed constant failed: {}", .{err});
        },
    );
    if (func.bin_file.cast(link.File.Prg)) |_| {
        return MV{ .abs_unresolved = .{ .blk_i = blk_i } };
    } else unreachable;
}

fn lowerDeclRef(func: *Func, val: Value, ty: Type, decl_index: Decl.Index) !MV {
    _ = ty;
    _ = val;

    const mod = func.getMod();
    const decl = mod.declPtr(decl_index);
    mod.markDeclAlive(decl);

    if (func.bin_file.cast(link.File.Prg)) |prg| {
        const blk_i = try prg.recordDecl(decl_index);
        return MV{ .abs_unresolved = .{ .blk_i = blk_i } };
    } else unreachable;
}

fn lowerParentPtr(func: *Func, ptr_val: Value, ptr_child_ty: Type) !MV {
    _ = func;
    _ = ptr_child_ty;
    switch (ptr_val.tag()) {
        .elem_ptr => {
            // TODO: ptrElemPtr?
            //const elem_ptr = ptr_val.castTag(.elem_ptr).?.data;
            //const index = elem_ptr.index;
            //const offset = index * ptr_child_ty.abiSize(func.getTarget());
            //const array_ptr = try func.lowerParentPtr(elem_ptr.array_ptr, elem_ptr.elem_ty);
            //_ = array_ptr;
            @panic("TODO");
        },
        else => @panic("TODO"),
    }
}

/// A lot of instructions have a constant amount of operands, but some have an unknown amoutn,
/// so this is used when we have more operands than `Liveness.bpi - 1`
/// (`- 1` to account for the bit that is for the instruction itself).
const BigTomb = struct {
    func: *Func,
    inst: Air.Inst.Index,
    lbt: Liveness.BigTomb,

    /// Feeds a liveness operand.
    fn feed(big_tomb: *BigTomb, op: Air.Inst.Ref, res: MV) void {
        const op_i = Air.refToIndex(op) orelse return; // constants do not die
        const dies = big_tomb.lbt.feed();
        if (!dies) return;
        big_tomb.func.processDeath(op_i, res);
    }

    /// Concludes liveness analysis for a runtime-known amount of operands of an AIR instruction.
    fn finishAir(big_tomb: *BigTomb, res: MV) void {
        // TODO: check `liveness.isUnused` here?
        const is_used = !big_tomb.func.liveness.isUnused(big_tomb.inst);
        if (is_used) {
            big_tomb.func.currentBranch().inst_vals.putAssumeCapacityNoClobber(big_tomb.inst, res);
        }

        if (debug.runtime_safety)
            big_tomb.func.air_bookkeeping += 1;
    }
};

/// Serves for liveness tracking of a runtime-known, possibly larger number of operands than `Liveness.bpi - 1`
/// (`- 1` to account for the bit that is for the instruction itself).
fn iterateBigTomb(func: *Func, inst: Air.Inst.Index, operand_count: usize) !BigTomb {
    try func.ensureProcessDeathCapacity(operand_count + 1);
    return BigTomb{
        .func = func,
        .inst = inst,
        .lbt = func.liveness.iterateBigTomb(inst),
    };
}

/// Ensures that we are able to process the upcoming deaths of this many additional operands.
fn ensureProcessDeathCapacity(func: *Func, additional_count: usize) !void {
    try func.currentBranch().inst_vals.ensureUnusedCapacity(func.getAllocator(), additional_count);
}

/// Processes the death of a liveness operand of an AIR instruction.
/// The given result is the result of an AIR instruction.
/// It is used to prevent the unintentional death of an operand that happens to
/// be the result as well.
fn processDeath(func: *Func, op: Air.Inst.Index, res: MV) void {
    if (func.air.instructions.items(.tag)[op] == .constant) return; // constants do not die
    log.debug("processing death of {} (%{})", .{ func.air.instructions.items(.tag)[op], op });
    const dead = func.getResolvedInst(op);

    log.debug("the one that dies is {} and the res is {}", .{ dead, res });

    // in some cases (such as @bitCast), an operand
    // may be the same MV as the result.
    // in that case, prevent deallocation.
    if (dead.eql(res))
        return;

    switch (dead) {
        .zp, .abs => {
            const ty = func.air.typeOfIndex(op);
            const child_ty = switch (ty.zigTypeTag()) {
                .Pointer => ty.childType(),
                else => ty,
            };
            func.memory.free(dead, child_ty);
        },
        .reg => |reg| switch (reg) {
            .a => func.reg_a = null,
            .x => func.reg_x = null,
            .y => func.reg_y = null,
        },
        else => unreachable,
    }
}

/// Concludes liveness analysis for a comptime-known number of operands of an AIR instruction.
/// `- 1` in `Liveness.bpi - 1` accounts for the fact that one of the bits refers to the instruction itself.
/// If you have more than `ops.len` operands, use BigTomb.
fn finishAir(func: *Func, inst: Air.Inst.Index, res: MV, ops: []const Air.Inst.Ref) void {
    // although the number of operands is comptime-known, we prefer doing it this way
    // so that we do not have to fill every non-existent operand space with `.none`
    assert(ops.len <= Liveness.bpi - 1); // use BigTomb if you need more than this

    var tomb_bits = func.liveness.getTombBits(inst);
    // the LSB is the first operand, and so on, up to `Liveness.bpi - 1` operands
    for (ops) |op| {
        const lives = @truncate(u1, tomb_bits) == 0;
        tomb_bits >>= 1;
        if (lives) continue;
        const op_i = Air.refToIndex(op) orelse
            // it is a constant
            continue;
        func.processDeath(op_i, res);
    }

    // the MSB is whether the instruction is unreferenced
    const is_used = @truncate(u1, tomb_bits) == 0;
    if (is_used) {
        func.currentBranch().inst_vals.putAssumeCapacityNoClobber(inst, res);
    }

    if (debug.runtime_safety)
        func.air_bookkeeping += 1;
}

fn resolveInst(func: *Func, inst_ref: Air.Inst.Ref) !MV {
    const ty = func.air.typeOf(inst_ref);

    // check whether the value is a static comptime-known constant, such as `false`, `true`, a type, etc.
    if (func.air.value(inst_ref)) |val| {
        return func.lowerConstant(val, ty);
    }

    // the value is runtime-known
    const inst_index = Air.refToIndex(inst_ref).?;
    switch (func.air.instructions.items(.tag)[inst_index]) {
        .constant => {
            if (true) panic("when does this happen {}", .{inst_ref});
            // this is different in that the value is also comptime-known but we do not know the exact
            // value and we have to generate that. this happens for comptime-known strings for example.
            // the instruction is constant and we emit it to the binary and then we will memoize it
            // in the outermost, first branch, similarly to how constants show up at the top in AIR.
            const branch = &func.branches.items[0];
            const gop = try branch.inst_vals.getOrPut(func.gpa, inst_index);
            if (!gop.found_existing) {
                const ty_pl = func.air.instructions.items(.data)[inst_index].ty_pl;
                gop.value_ptr.* = try func.genTypedValue(.{
                    .ty = ty,
                    .val = func.air.values[ty_pl.payload],
                });
            }
            return gop.value_ptr.*;
        },
        .const_ty => unreachable,
        else => return func.getResolvedInst(inst_index),
    }
}

fn getResolvedInst(func: *Func, inst: Air.Inst.Index) MV {
    log.debug("getting resolved instruction for %{}", .{inst});
    // check whether the instruction refers to a value in one of our local branches.
    // example for visualization:
    // ```
    // %1 = x;
    // {
    //      %2 = x;
    //      // -> if we are here we have access to instruction %1 and %2
    // }
    // // -> if we are here we have access to instruction %1 but not %2
    // ```
    var i = func.branches.items.len - 1;
    while (true) : (i -= 1) {
        if (func.branches.items[i].inst_vals.get(inst)) |val| {
            assert(val != .none);
            return val;
        }
    }
}

/// Adds an MIR instruction to the output.
/// See `Mir.Inst.checkCombo` for an explanation on why `data` is `anytype`.
/// This function is the only point that adds instructions to the MIR output.
/// It has safety that ensures the following:
/// * It is impossible to generate invalid instructions.
/// * It is impossible to clobber registers unintentionally.
fn addInstInternal(func: *Func, tag: Mir.Inst.Tag, data: anytype) !void {
    // TODO: if we migrate to no longer using `addInst` but only the other specific `addInst` variants,
    //       we no longer need this safety
    // TODO: try doing this at comptime by making `tag` and `data` comptime
    //       (as of writing, making `data` into comptime crashes the compiler)
    const inst = Mir.Inst{ .tag = tag, .data = @as(Mir.Inst.Data, data) };
    // technically we could also do this check manually in each addInst* variant
    // but it is more central this way
    if (debug.runtime_safety) {
        // prevent generating invalid instructions
        Mir.Inst.checkCombo(tag, data);

        // prevent clobbering registers unintentionally
        if (inst.tag.getAffectedRegister()) |affected_reg| {
            switch (affected_reg) {
                .a => if (func.reg_a != null) assert(func.reg_a == func.air_current_inst),
                .x => if (func.reg_x != null) assert(func.reg_x == func.air_current_inst),
                .y => if (func.reg_y != null) assert(func.reg_y == func.air_current_inst),
            }
        }
    }
    try func.mir_instructions.append(func.getAllocator(), inst);
    log.debug("added instruction .{{ .tag = {}, .data = {} }}", .{ tag, data });
}
fn addInstImpl(func: *Func, tag: Mir.Inst.Tag) !void {
    try func.addInstInternal(tag, .{ .none = {} });
}
fn addInstImm(func: *Func, tag: Mir.Inst.Tag, value: u8) !void {
    try func.addInstInternal(tag, .{ .imm = value });
}
/// Adds an MIR instruction to the output that operates on zero page.
/// Use this over `addInstMem` if you do not expect absolute memory to be used.
/// Use MV.index() to get an index into the value.
fn addInstZp(func: *Func, tag: Mir.Inst.Tag, mv: MV) !void {
    try func.addInstInternal(tag, .{ .zp = mv.zp });
}
/// Adds an MIR instruction to the output that operates on absolute memory.
/// Use this over `addInstMem` if you do not expect the zero page to be used.
/// Use MV.index() to get an index into the value.
fn addInstAbs(func: *Func, tag: Mir.Inst.Tag, val: MV) !void {
    switch (val) {
        .abs => |addr| try func.addInstInternal(
            tag,
            .{ .abs = .{ .imm = addr } },
        ),
        .abs_unresolved => |unresolved| try func.addInstInternal(
            tag,
            .{ .abs = .{ .unresolved = .{ .blk_i = unresolved.blk_i, .offset = unresolved.offset } } },
        ),
        else => unreachable,
    }
}
/// Adds an MIR instruction to the output that operates on addressable memory.
/// Whenever you see an opcode mnemonic in double quotes, you can expect it to be used with multiple possible addressing modes.
///
/// Use MV.index() to get an index into the value.
///
/// Use `addInstAbs` instead if you do not expect the zero page to be used.
/// Use `addInstZp` instead if you do not expect absolute memory to be used.
fn addInstMem(func: *Func, comptime mnemonic: []const u8, val: MV) !void {
    // TODO: once we hit possible 100 instructions in Mir.Inst.Tag, stringToEnum will stop working.
    switch (val) {
        .zp => |addr| try func.addInstInternal(
            std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_zp").?,
            .{ .zp = addr },
        ),
        .abs => |addr| try func.addInstInternal(
            std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_abs").?,
            .{ .abs = .{ .imm = addr } },
        ),
        .abs_unresolved => |unresolved| {
            try func.addInstInternal(
                std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_abs").?,
                .{ .abs = .{ .unresolved = .{ .blk_i = unresolved.blk_i, .offset = unresolved.offset } } },
            );
        },
        else => unreachable,
    }
}
fn addInstRel(func: *Func, tag: Mir.Inst.Tag, offset: i8) !void {
    try func.addInstInternal(tag, .{ .rel = offset });
}
/// Adds an MIR instruction to the output that operates using the appropriate addressing mode based on the given value.
/// Whenever you see an opcode mnemonic in double quotes, you can expect it to be used with multiple possible addressing modes.
///
/// Use MV.index() to get an index into the value.
///
/// Use a more specific addInst* function if possible.
fn addInstAny(func: *Func, comptime mnemonic: []const u8, val: MV) !void {
    switch (val) {
        .none => try func.addInstImpl(std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_impl").?),
        .imm => |imm| try func.addInstImm(std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_imm").?, imm),
        .reg => {
            const temp = .{ .zp = (func.memory.zp_res orelse return func.fail("zero page depleted", .{}))[0] };
            try func.trans(val, temp, Type.u8, Type.u8);
            try func.addInstAny(mnemonic, temp);
        },
        .zp, .abs, .abs_unresolved => try func.addInstMem(mnemonic, val),
        .abs_imm => unreachable, // TODO
        .zp_imm => |imm| try func.addInstImm(std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_imm").?, imm),
    }
}

/// Returns a pointer to the last MIR instruction added.
fn currentInst(func: *Func) struct { tag: *Mir.Inst.Tag, data: *Mir.Inst.Data } {
    return .{
        .tag = &func.mir_instructions.items(.tag)[func.mir_instructions.len - 1],
        .data = &func.mir_instructions.items(.data)[func.mir_instructions.len - 1],
    };
}

fn fail(func: *Func, comptime fmt: []const u8, args: anytype) error{ CodegenFail, OutOfMemory } {
    @setCold(true);
    func.err_msg = try Module.ErrorMsg.create(func.getAllocator(), func.src_loc, fmt, args);
    return error.CodegenFail;
}
