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
const FnResult = codegen.FnResult;
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
/// The allocator we will be using throughout.
gpa: mem.Allocator,

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
/// Holds the index of the current instruction generated.
/// Used to prevent clobbering registers unintentionally.
current_inst: if (debug.runtime_safety) Air.Inst.Index else void = if (debug.runtime_safety) undefined else {},

// TODO: remove these ref links eventually
// https://github.com/cc65/cc65/blob/master/asminc/c64.inc
// https://github.com/cc65/cc65/blob/master/asminc/cbm_kernal.inc

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
) GenerateSymbolError!FnResult {
    _ = debug_output;

    const gpa = bin_file.allocator;
    const memory = Memory.init(bin_file.options.target, bin_file);
    var func = Func{
        .bin_file = bin_file,
        .src_loc = src_loc,
        .props = props,
        .air = air,
        .liveness = liveness,
        .gpa = gpa,
        .memory = memory,
    };
    defer func.deinit();

    func.gen() catch |err| switch (err) {
        error.CodegenFail => return FnResult{ .fail = func.err_msg },
        error.OutOfRegisters => unreachable,
        error.OutOfMemory => |other| return other,
    };

    var mir = Mir{
        .instructions = func.mir_instructions.slice(),
    };

    var emit = Emit{
        .mir = mir,
        .bin_file = bin_file,
        .code = code,
    };

    try emit.emitMir();

    return .{ .appended = {} };
}

fn deinit(func: *Func) void {
    for (func.branches.items) |*branch|
        branch.deinit(func.gpa);
    func.branches.deinit(func.gpa);
    func.mir_instructions.deinit(func.gpa);
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
    imm: u8,
    /// The value is in a register.
    reg: Register,
    /// The value is in the first page of memory.
    /// This becomes Mir.Inst.Data.zp.
    zp: u8,
    /// The value is in one of the 256 pages of memory (64 KiB),
    /// excluding the first one (the zero page).
    /// This becomes Mir.Inst.Data.abs.imm.
    abs: u16,
    /// Same as `abs` except that this is a block index and
    /// the absolute memory address is yet to be resolved by the linker.
    /// This becomes Mir.Inst.Data.abs.unresolved.
    // TODO: rename to block?
    abs_unresolved: struct {
        blk_i: u16,
        offset: u16 = 0, // TODO: i32
    },

    /// Returns the given value in memory subscripted by the index in form of an offset.
    fn index(mv: MV, offset: u16) MV {
        switch (mv) {
            .none => unreachable,
            .imm => |_| {
                assert(offset == 0);
                return mv;
            },
            .reg => {
                assert(offset == 0);
                return mv;
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
fn freeReg(func: *Func, reg: Register, inst: Air.Inst.Index) !MV {
    const spillReg = struct {
        fn spillReg(func_inn: *Func, reg_inn: Register, inst_inn: Air.Inst.Index) !void {
            const val = func_inn.getResolvedInstValue(Air.indexToRef(inst_inn));
            if (val == .reg)
                assert(reg_inn == val.reg);
            const new_home = try func_inn.allocMem(Type.u8);
            try func_inn.trans(val, new_home, Type.u8, Type.u8);
            try func_inn.currentBranch().values.put(func_inn.gpa, Air.indexToRef(inst_inn), new_home);
        }
    }.spillReg;

    switch (reg) {
        .a => {
            if (func.reg_a) |old_inst| {
                assert(inst != old_inst); // TODO: can it happen that they're the same?
                // TODO: use PHA and PLA to spill using the hardware stack?
                //       if yes, how do we best associate the PLA with the inst and emit it when the inst dies?
                //       also, can we TXA and TYA below and do it there too? measure total cycles of each solution
                try spillReg(func, reg, old_inst);
            }
            func.reg_a = inst;
        },
        .x => {
            if (func.reg_x) |old_inst| {
                assert(inst != old_inst); // TODO: can it happen that they're the same?
                try spillReg(func, reg, old_inst);
            }
            func.reg_x = inst;
        },
        .y => {
            if (func.reg_y) |old_inst| {
                assert(inst != old_inst); // TODO: can it happen that they're the same?
                try spillReg(func, reg, old_inst);
            }
            func.reg_y = inst;
        },
    }
    return .{ .reg = reg };
}

/// Assumes the given register is now owned by the new instruction and associates it with the new instruction.
fn takeReg(func: *Func, reg: Register, inst: Air.Inst.Index) MV {
    switch (reg) {
        .a => func.reg_a = inst,
        .x => func.reg_x = inst,
        .y => func.reg_y = inst,
    }
    return .{ .reg = reg };
}

/// This represents the system's addressable memory where each byte has an address.
const Memory = struct {
    /// Memory addresses in the first page of memory available for storage.
    /// This is expensive, sparsely available memory and very valuable because
    /// writing and reading from it is faster than for absolute memory.
    /// This must always contain at least two free addresses for pointer derefencing.
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

    fn init(target: std.Target, bin_file: *link.File) Memory {
        if (bin_file.cast(link.File.Prg)) |prg| {
            if (prg.zp_free != null and prg.abs_offset != null) {
                const zp_free = prg.zp_free.?;
                const abs_offset = prg.abs_offset.?;
                return .{
                    .zp_free = zp_free,
                    .abs_offset = abs_offset,
                };
            }
        } else unreachable;

        const zp_free = abi.getZeroPageAddresses(target);
        const abs_offset = abi.getAbsoluteMemoryOffset(target);
        return .{
            .zp_free = zp_free,
            .abs_offset = abs_offset,
        };
    }

    /// Allocates two bytes specifically from the zero page for the purpose of storing a pointer address and dereferencing that pointer.
    fn allocZeroPageDerefAddress(memory: *Memory) !MV {
        if (memory.allocZeroPageMemory(2)) |addr|
            return addr;
        const func = @fieldParentPtr(Func, "memory", memory);
        return func.fail("unable to dereference pointer due to zero page shortage", .{});
    }

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
            // but we must still make sure the zero page has at least 2 addresses free
            (byte_count == 1 and memory.zp_free.len > 2);

        if (use_zero_page) {
            if (memory.allocZeroPageMemory(byte_count)) |addr|
                return addr;
        }

        if (memory.allocAbsoluteMemory(byte_count)) |addr|
            return addr;

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

    /// Returns null if OOM or if no contiguous bytes were found.
    fn allocZeroPageMemory(memory: *Memory, byte_count: u16) ?MV {
        assert(byte_count != 0);
        const addrs = memory.zp_free.constSlice();
        if (debug.runtime_safety and !std.sort.isSorted(u8, addrs, {}, std.sort.asc(u8)))
            @panic("free zero page addresses must be sorted low to high");
        // find a matching number of free bytes, each with an address only 1 apart from the one before it (contiguous)
        var contig_count: u8 = 0;
        var contig_addr_i: u8 = 0;
        var i: u8 = 0;
        while (true) {
            if (i >= addrs.len)
                break;
            const addr1 = addrs[i];
            i += 1;
            if (i >= addrs.len)
                break;
            const addr2 = addrs[i];
            i += 1;
            const contig = addr1 == addr2 - 1;
            if (contig) {
                contig_count += 2;
                if (contig_count == byte_count)
                    break contig_addr_i;
            } else {
                // would this be enough?
                if (contig_count + 1 == byte_count) {
                    contig_count += 1;
                    break contig_addr_i;
                } else {
                    // no; reset
                    contig_count = 1;
                    contig_addr_i = i;
                }
            }
        } else return null;
        if (contig_addr_i + byte_count <= memory.zp_free.len) {
            // TODO: what's generally faster?
            //       1. multiple `swapRemove`s and then one `std.sort.sort` at the end
            //       or 2. multiple `orderedRemove` and no `std.sort.sort`?
            const start_addr = memory.zp_free.swapRemove(contig_addr_i);
            var times: u16 = 1;
            while (times < byte_count) : (times += 1)
                _ = memory.zp_free.swapRemove(contig_addr_i + times);
            std.sort.sort(u8, memory.zp_free.slice(), {}, std.sort.asc(u8));
            return MV{ .zp = start_addr };
        }
        return null;
    }

    /// Returns null if OOM.
    fn allocAbsoluteMemory(memory: *Memory, byte_count: u16) ?MV {
        assert(byte_count != 0);
        // TODO:
        //if (addr == abs.abs_end) {
        //    return null;
        //}
        memory.abs_offset -= byte_count; // grow the stack downwards
        return MV{ .abs = memory.abs_offset + 1 };
    }

    fn free(memory: *Memory, value: MV, ty: Type) void {
        const func = @fieldParentPtr(Func, "memory", memory);
        const byte_size = func.getByteSize(ty).?;
        log.debug("I want to free {} bytes", .{byte_size});
        switch (value) {
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
        };
        try testing.expectEqual(MV{ .zp = 0 }, memory.allocZeroPageMemory(3).?);
        try testing.expectEqual(MV{ .zp = 4 }, memory.allocZeroPageMemory(2).?);
        try testing.expectEqual(@as(?MV, null), memory.allocZeroPageMemory(2));
        try testing.expectEqual(MV{ .zp = 0x80 }, memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?MV, null), memory.allocZeroPageMemory(10));
        try testing.expectEqual(MV{ .zp = 0x90 }, memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));

        memory.zp_free.appendSliceAssumeCapacity(&[_]u8{ 0xfe, 0xff });
        try testing.expectEqual(MV{ .zp = 0xfe }, memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?MV, null), memory.allocZeroPageMemory(0xff));
        try testing.expectEqual(MV{ .zp = 0xff }, memory.allocZeroPageMemory(1).?);
        try testing.expectEqual(@as(?MV, null), memory.allocZeroPageMemory(1));
        try testing.expectEqual(@as(u8, 0), @intCast(u8, memory.zp_free.len));
    }

    test allocAbsoluteMemory {
        var memory = Memory{
            .zp_free = undefined,
            .abs_offset = 0xffff,
            //.abs_end = 0xfeff,
        };
        try testing.expectEqual(MV{ .abs = 0xffff }, memory.allocAbsoluteMemory(1).?);
        try testing.expectEqual(MV{ .abs = 0xfffe }, memory.allocAbsoluteMemory(1).?);
        try testing.expectEqual(MV{ .abs = 0xfff0 }, memory.allocAbsoluteMemory(0xe).?);
        try testing.expectEqual(MV{ .abs = 0xffee }, memory.allocAbsoluteMemory(2).?);
        try testing.expectEqual(MV{ .abs = 0xff6e }, memory.allocAbsoluteMemory(128).?);
        //try testing.expectEqual(@as(?MV,null), memory.allocAbsoluteMemory(0xffff).?);
        try testing.expectEqual(@as(u16, 0xff6d), memory.abs_offset);
    }
};

// TODO: are these tests run as part of `zig build test`?
comptime {
    _ = Memory;
}

const Branch = struct {
    values: std.AutoArrayHashMapUnmanaged(Air.Inst.Ref, MV) = .{},

    fn deinit(branch: *Branch, allocator: mem.Allocator) void {
        branch.values.deinit(allocator);
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
    const param_types = try func.gpa.alloc(Type, fn_ty.fnParamLen());
    defer func.gpa.free(param_types);
    fn_ty.fnParamTypes(param_types);
    var values = CallMVs{
        .args = try func.gpa.alloc(MV, param_types.len),
        .return_value = undefined,
    };
    errdefer func.gpa.free(values.args);

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
    var call_values = try func.resolveCallingConventionValues(func.getType());
    defer call_values.deinit(func.gpa);
    func.args = call_values.args;
    func.ret_val = call_values.return_value;

    try func.branches.append(func.gpa, .{});
    defer {
        var outer_branch = func.branches.pop();
        outer_branch.deinit(func.gpa);
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
            func.current_inst = inst;

        try func.currentBranch().values.ensureUnusedCapacity(func.gpa, Liveness.bpi);
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

fn genInst(func: *Func, inst: Air.Inst.Index) error{ CodegenFail, OutOfMemory, OutOfRegisters }!void {
    const tag = func.air.instructions.items(.tag)[inst];
    log.debug("lowering {}...", .{tag});
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
        .loop => func.airLoop(inst),
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
        .ptr_elem_val => func.airPtrElemVal(inst),
        .ptr_elem_ptr => func.airPtrElemPtr(inst),
        .array_to_slice,
        .float_to_int,
        .float_to_int_optimized,
        .int_to_float,

        .reduce,
        .reduce_optimized,
        .splat,
        .shuffle,
        .select,

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
    const bin_op = func.air.extraData(Air.Bin, ty_pl.payload).data;
    const ty = func.air.getRefType(ty_pl.ty);
    const lhs = try func.resolveInst(bin_op.lhs);
    const rhs = try func.resolveInst(bin_op.rhs);
    const lhs_ty = func.air.typeOf(bin_op.lhs);
    const rhs_ty = func.air.typeOf(bin_op.rhs);
    assert(lhs_ty.eql(ty, func.getMod()));
    const res = try func.binOp(tag, lhs, rhs, lhs_ty, rhs_ty, null, inst);
    return func.finishAir(inst, res, &.{ bin_op.lhs, bin_op.rhs });
}

/// Emits code to perform the given binary operation on two operands of possible differing types, resulting in a value of either of the given types.
/// A binary operation is a rule for combining two operands to produce another value.
fn binOp(
    func: *Func,
    tag: Air.Inst.Tag,
    lhs: MV,
    rhs: MV,
    lhs_ty: Type,
    rhs_ty: Type,
    dst: ?MV,
    inst: Air.Inst.Index,
) !MV {
    switch (tag) {
        .add,
        .addwrap,
        .sub,
        .subwrap,
        => {
            assert(lhs_ty.eql(rhs_ty, func.getMod()));
            const ty = lhs_ty;
            switch (ty.zigTypeTag()) {
                .Int => {
                    return try func.intBinOp(lhs, rhs, ty, tag, dst, inst);
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
                        return try func.binOp(op_tag, lhs, rhs, Type.usize, Type.usize, dst, inst);
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
    bin_op: Air.Inst.Tag,
    dst: ?MV,
    inst: Air.Inst.Index,
) !MV {
    assert(ty.zigTypeTag() == .Int);

    const byte_size = func.getByteSize(ty).?;

    switch (bin_op) {
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
        .add, .addwrap => {
            try func.decimal.clear();
            try func.carry.clear();
            defer func.carry.state = .unknown;

            var res = dst;
            var reg_a: ?MV = null;

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // take advantage of the fact that addition is commutative,
                // meaning we can add the operands in any order,
                // so get either of the operands into the accumulator.
                var values = std.BoundedArray(MV, 2){};
                values.appendAssumeCapacity(lhs.index(i));
                values.appendAssumeCapacity(rhs.index(i));
                for (values.constSlice()) |val, val_i| {
                    // is either LHS or RHS already in the accumulator?
                    if (val == .reg and val.reg == .a) {
                        if (dst == null and byte_size == 1) {
                            // we can do this if LHS or RHS is the same as the result location
                            reg_a = func.takeReg(.a, inst);
                            log.debug("inst: {}, current_inst: {}", .{ inst, func.current_inst });
                            try func.addInstAny("adc", values.pop());
                            return reg_a.?;
                        }
                        _ = values.swapRemove(val_i);
                        break;
                    }
                } else {
                    // neither LHS or RHS is in the accumulator
                    if (reg_a == null)
                        reg_a = try func.freeReg(.a, inst);
                    //try func.load(reg_a, values.pop(), Type.u8);
                    try func.trans(values.pop(), reg_a.?, Type.u8, Type.u8);
                }
                try func.addInstAny("adc", values.pop());
                assert(values.len == 0);
                //try func.load(res.index(i), reg_a, Type.u8);
                if (res == null)
                    res = try func.allocMem(ty);
                try func.trans(reg_a.?, res.?.index(i), Type.u8, Type.u8);
            }
            return res.?;
        },
        .sub, .subwrap => {
            try func.decimal.clear();
            // for subtraction we have to do the opposite of what we do for addition:
            // we set carry, which for subtraction means we *clear borrow*.
            try func.carry.set();
            defer func.carry.state = .unknown;

            if (lhs == .reg and lhs.reg == .a) {
                if (dst == null and byte_size == 1) {
                    try func.addInstAny("sbc", rhs);
                    const reg_a = func.takeReg(.a, inst);
                    return reg_a;
                }
            }

            const res = dst orelse try func.allocMem(ty);
            const reg_a = try func.freeReg(.a, inst);

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // get the LHS byte into the accumulator and subtract the RHS byte from it
                try func.trans(lhs.index(i), reg_a, Type.u8, Type.u8);
                //try func.load(reg_a, lhs.index(i), Type.u8);
                try func.addInstAny("sbc", rhs.index(i));
                try func.trans(reg_a, res.index(i), Type.u8, Type.u8);
                //try func.load(res.index(i), reg_a, Type.u8);
            }

            return res;
        },
        else => unreachable,
    }
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
    log.debug("allocated ptr: {}", .{res});
    func.finishAir(inst, res, &.{});
}

fn airRetPtr(func: *Func, inst: Air.Inst.Index) !void {
    // this is equivalent to airAlloc if we do not choose to pass by reference
    const ptr_ty = func.air.instructions.items(.data)[inst].ty;
    const res = try func.allocMem(ptr_ty.childType());
    log.debug("(ret_ptr) allocated ptr: {}", .{res});
    func.finishAir(inst, res, &.{});
}

/// Allocates addressable memory capable of storing a value of the given type.
fn allocMem(func: *Func, ty: Type) !MV {
    const byte_size = func.getByteSize(ty) orelse return func.fail("type `{}` too big to fit in address space", .{ty.fmt(func.getMod())});
    return try func.memory.alloc(byte_size);
}

/// Emits code to write a value to a pointer.
fn airStore(func: *Func, inst: Air.Inst.Index) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;

    const ptr = try func.resolveInst(bin_op.lhs);
    const ptr_ty = func.air.typeOf(bin_op.lhs);
    const val = try func.resolveInst(bin_op.rhs);
    const val_ty = func.air.typeOf(bin_op.rhs);

    log.debug("store {} ({}) at {} ({})", .{ val, val_ty.tag(), ptr, ptr_ty.tag() });

    //try func.store(val, ptr, val_ty, ptr_ty);
    try func.trans(val, ptr, val_ty, ptr_ty.childType());

    func.finishAir(inst, .none, &.{ bin_op.lhs, bin_op.rhs });
}

/// Stores the given value at the given pointer.
/// This means we write the given value to the pointer.
/// Preserves all registers not specified in the operation.
/// The parameter order intentionally resembles the ST* instructions.
/// If the value is a memory value, *it is the address* that is stored at the pointer, *not the value at that address*.
fn store(func: *Func, val: MV, ptr: MV, val_ty: Type, ptr_ty: Type) !void {
    const ptr_size = func.getByteSize(val_ty).?;
    const child_ptr_ty = switch (ptr_ty.zigTypeTag()) {
        .Pointer => ptr_ty.childType(),
        else => ptr_ty,
    };
    const val_size = func.getByteSize(child_ptr_ty).?;

    log.debug("storing {} bytes of {} to {} which is capable of holding {} bytes", .{ val_size, val, ptr, ptr_size });

    if (val_size == 0)
        return;

    assert(val_size <= ptr_size);
    switch (ptr) {
        .none => unreachable, // cannot write something to nothing
        .imm => unreachable, // cannot write something to an immediate value. should be `zp` if anything.
        .reg => |reg| {
            // write the value to the register
            switch (val) {
                .none => {
                    // write nothing to the register
                },
                .imm => |imm| {
                    // write immediate value to the register
                    switch (reg) {
                        .a => try func.addInstImm(.lda_imm, imm),
                        .x => try func.addInstImm(.ldx_imm, imm),
                        .y => try func.addInstImm(.ldy_imm, imm),
                    }
                },
                .reg => {
                    assert(val.reg == ptr.reg);
                },
                .zp => {
                    // write zero page value to the register
                    switch (reg) {
                        .a => try func.addInstZp(.lda_zp, val),
                        .x => try func.addInstZp(.ldx_zp, val),
                        .y => try func.addInstZp(.ldy_zp, val),
                    }
                },
                .abs, .abs_unresolved => unreachable, // cannot write 2-byte address to 1-byte register
            }
        },
        .zp, .abs, .abs_unresolved => {
            // write the value to addressable memory
            switch (val) {
                .none => {
                    // write nothing to addressable memory
                },
                .imm => |imm| {
                    // write immediate value to addressable memory
                    // TODO: make this A reg temporary saving thing into a function
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    try func.addInstImm(.lda_imm, imm);
                    try func.addInstMem("sta", ptr);
                    try func.addInstImpl(.pla_impl);
                },
                .reg => |reg| {
                    switch (reg) {
                        .a => try func.addInstMem("sta", ptr),
                        .x => try func.addInstMem("stx", ptr),
                        .y => try func.addInstMem("sty", ptr),
                    }
                },
                .zp => |addr| {
                    // write zero page address to addressable memory
                    try func.addInstImm(.lda_imm, addr);
                    try func.addInstMem("sta", ptr);
                },
                .abs, .abs_unresolved => {
                    // write absolute address to addressable memory
                    try func.addInstMem("lda", val);
                    try func.addInstMem("sta", ptr);
                    try func.addInstMem("lda", val.index(1));
                    try func.addInstMem("sta", ptr.index(1));
                },
            }
        },
    }
}

/// Emits code to read a value from a pointer.
fn airLoad(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const val_ty = func.air.getRefType(ty_op.ty);
    const ptr = try func.resolveInst(ty_op.operand);
    const ptr_ty = func.air.typeOf(ty_op.operand);
    assert(val_ty.eql(ptr_ty.childType(), func.getMod()));
    const dst = try func.allocMem(ptr_ty.childType());
    //const dst_ty = ptr_ty;
    try func.trans(ptr, dst, ptr_ty.childType(), val_ty);
    //try func.load(dst, ptr, ptr_ty);
    //try func.trans(ptr,dst,ptr_ty,dst_ty);
    func.finishAir(inst, dst, &.{ty_op.operand});
}

/// Transfers from source to destination.
fn trans(func: *Func, src: MV, dst: MV, src_ty: Type, dst_ty: Type) !void {
    //const child_src_ty = switch (src_ty.zigTypeTag()) {
    //    .Pointer => switch (src) {
    //        .none, .imm => unreachable,
    //        .reg, .zp, .abs, .abs_unresolved => src_ty.childType(),
    //    },
    //    else => switch (src) {
    //        .none, .imm => src_ty,
    //        .reg, .zp, .abs, .abs_unresolved => unreachable,
    //    },
    //};
    //const child_dst_ty = switch (dst_ty.zigTypeTag()) {
    //    .Pointer => switch (dst) {
    //        .none, .imm => unreachable,
    //        .reg, .zp, .abs, .abs_unresolved => dst_ty.childType(),
    //    },
    //    else => switch (dst) {
    //        .none, .imm => dst_ty,
    //        .reg, .zp, .abs, .abs_unresolved => unreachable,
    //    },
    //};
    const src_byte_size = func.getByteSize(src_ty).?; //child_src_ty).?;
    const dst_byte_size = func.getByteSize(dst_ty).?; //child_dst_ty).?;
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
                    assert(dst_byte_size == 1); // TODO: could be more
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    try func.addInstImm(.lda_imm, src_imm);
                    try func.addInstMem("sta", dst);
                    try func.addInstImpl(.pla_impl);
                },
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
            }
        },
        //else => panic("TODO: handle {}", .{src}),
    }
}

/// Loads into the destination the value that the pointer points at.
/// This means we read the value from the pointer and write it to the destination.
/// Preserves all registers not specified in the operation.
/// Asserts destination size is >= size of pointer type.
/// The parameter order intentionally resembles the LD* instructions.
fn load(func: *Func, dst: MV, ptr: MV, ptr_ty: Type) !void {
    const child_ptr_ty = ptr_ty.childType();
    log.debug("loading value of type {} from {} into {}...", .{ child_ptr_ty.tag(), ptr, dst });
    const byte_size = func.getByteSize(child_ptr_ty).?;
    switch (ptr) {
        .none => {
            // write nothing to the destination
        },
        .imm => |imm| {
            // load the immediate value to the destination
            assert(byte_size == 1);
            switch (dst) {
                .none => unreachable, // cannot write register value to nothing
                .imm => unreachable, // cannot write immediate value to immediate value. should be `zp` if anything.
                .reg => |reg| {
                    switch (reg) {
                        .a => try func.addInstImm(.lda_imm, imm),
                        .x => try func.addInstImm(.ldx_imm, imm),
                        .y => try func.addInstImm(.ldy_imm, imm),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    // TODO: make this A reg temporary saving thing into a function
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    try func.addInstImm(.lda_imm, imm);
                    try func.addInstMem("sta", dst);
                    try func.addInstImpl(.pla_impl);
                },
            }
        },
        .reg => {
            // load the register's value to the destination
            switch (dst) {
                .none => unreachable, // cannot write register value to nothing
                .imm => unreachable, // cannot write register value to immediate value. should be `zp` if anything.
                .reg => {
                    assert(ptr.reg == dst.reg);
                },
                else => panic("TODO: handle {}", .{dst}),
            }
        },
        .zp, .abs, .abs_unresolved => {
            // load the value from the address to the destination
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |reg| {
                    switch (reg) {
                        .a => try func.addInstMem("lda", ptr),
                        .x => try func.addInstMem("ldx", ptr),
                        .y => try func.addInstMem("ldy", ptr),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    // TODO: if ReleaseSmall, do this loop at runtime at a certain threshold
                    // TODO: make this A reg temporary saving thing into a function
                    const reg_a = func.reg_a;
                    func.reg_a = null;
                    defer func.reg_a = reg_a;
                    try func.addInstImpl(.pha_impl);
                    var i: u16 = 0;
                    while (i < byte_size) : (i += 1) {
                        try func.addInstMem("lda", ptr.index(i));
                        try func.addInstMem("sta", dst.index(i));
                    }
                    try func.addInstImpl(.pla_impl);
                },
            }
        },
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
            //try func.load(res.index(0), operand, Type.u8);
            //try func.load(res.index(1), .{ .imm = 0 }, Type.u8);
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
                        @panic("TODO");
                    },
                };
            },
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
    const res: MV = res: {
        break :res try func.ptrElemVal(ptr, index, ptr_ty, inst);
    };
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
    try func.trans(addr, dst, ptr_ty.childType(), child_ty);
    //try func.load(dst, addr, ptr_ty);
    return dst;
}

fn airBitCast(func: *Func, inst: Air.Inst.Index) !void {
    // this is only a change at the type system level
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const dst_ty = func.air.getRefType(ty_op.ty);
    _ = dst_ty;
    const res = try func.resolveInst(ty_op.operand);
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
    try func.load(dst, ptr, ptr_ty);
    const val = dst;
    try func.load(func.ret_val, val, ret_ty);
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
    defer call_values.deinit(func.gpa);

    for (call_values.args) |dst, i| {
        const src = try func.resolveInst(args[i]);
        const arg_ty = func.air.typeOf(args[i]);
        try func.trans(src, dst, arg_ty, arg_ty);
        //try func.load(dst, src, arg_ty);
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

    const result = call_values.return_value;

    var big_tomb = try func.iterateBigTomb(inst, args.len + 1);
    big_tomb.feed(pl_op.operand);
    for (args) |arg| big_tomb.feed(arg);
    return big_tomb.finishAir(result);

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

    log.debug("ty_pl ty: {}", .{func.air.typeOf(ty_pl.ty).tag()});

    var extra_i = extra.end;
    const outputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.outputs_len]);
    // TODO: support outputting specific registers into variables (so, the other way around)
    assert(outputs.len == 0);
    extra_i += outputs.len;
    const inputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.inputs_len]);
    extra_i += inputs.len;

    assert(extra.data.clobbers_len() == 0);

    const dies = !extra.data.is_volatile() and func.liveness.isUnused(inst);
    const result: MV = if (dies)
        .none
    else result: {
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
            //try func.load(func.freeReg(reg, inst), input_mv, ty);
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

        break :result .none;
    };

    var big_tomb = try func.iterateBigTomb(inst, outputs.len + inputs.len);
    for (outputs) |output| {
        if (output == .none) continue;
        big_tomb.feed(output);
    }
    for (inputs) |input| {
        big_tomb.feed(input);
    }
    return big_tomb.finishAir(result);
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

    const module = func.bin_file.options.module.?;
    const decl = module.declPtr(decl_index);
    module.markDeclAlive(decl);

    if (func.bin_file.cast(link.File.Prg)) |prg| {
        const blk_i = try prg.recordDecl(decl_index);
        return MV{ .abs_unresolved = .{ .blk_i = blk_i } };
    } else unreachable;
}

fn lowerParentPtr(func: *Func, ptr_val: Value, ptr_child_ty: Type) !MV {
    switch (ptr_val.tag()) {
        .elem_ptr => {
            // TODO: ptrElemPtr?
            const elem_ptr = ptr_val.castTag(.elem_ptr).?.data;
            const index = elem_ptr.index;
            const offset = index * ptr_child_ty.abiSize(func.getTarget());
            const array_ptr = try func.lowerParentPtr(elem_ptr.array_ptr, elem_ptr.elem_ty);
            _ = array_ptr;

            log.debug("{} {} , offset: {},index:{}", .{ elem_ptr.array_ptr.fmtValue(elem_ptr.elem_ty, func.getMod()), elem_ptr.elem_ty.fmt(func.getMod()), offset, index });
            @panic("TODO");
            //return WValue{ .memory_offset = .{
            //    .pointer = array_ptr.memory,
            //    .offset = @intCast(u32, offset),
            //}, };
        },
        else => @panic("TODO"),
    }
}

const BigTomb = struct {
    func: *Func,
    inst: Air.Inst.Index,
    lbt: Liveness.BigTomb,

    fn feed(big_tomb: *BigTomb, op_ref: Air.Inst.Ref) void {
        _ = Air.refToIndex(op_ref) orelse return; // constants do not have to be freed regardless
        const dies = big_tomb.lbt.feed();
        if (!dies) return;
        processDeath(big_tomb.func, op_ref);
    }

    fn finishAir(big_tomb: *BigTomb, result: MV) void {
        if (result != .none) {
            big_tomb.func.currentBranch().values.putAssumeCapacityNoClobber(Air.indexToRef(big_tomb.inst), result);
        }
        if (debug.runtime_safety) {
            big_tomb.func.air_bookkeeping += 1;
        }
    }
};

// TODO: `operand_count: u16`?
fn iterateBigTomb(func: *Func, inst: Air.Inst.Index, operand_count: usize) !BigTomb {
    try func.currentBranch().values.ensureUnusedCapacity(func.gpa, operand_count + 1);
    return BigTomb{
        .func = func,
        .inst = inst,
        .lbt = func.liveness.iterateBigTomb(inst),
    };
}

fn processDeath(func: *Func, ref: Air.Inst.Ref) void {
    const inst = Air.refToIndex(ref) orelse return;
    if (func.air.instructions.items(.tag)[inst] == .constant) return; // constants do not die
    log.debug("processing death of {}", .{func.air.instructions.items(.tag)[inst]});
    const value = func.getResolvedInstValue(ref);
    switch (value) {
        .zp, .abs => func.memory.free(value, func.air.typeOfIndex(inst)),
        .reg => |reg| switch (reg) {
            .a => func.reg_a = null,
            .x => func.reg_x = null,
            .y => func.reg_y = null,
        },
        else => unreachable,
    }
}

fn finishAir(func: *Func, inst: Air.Inst.Index, result: MV, operands: []const Air.Inst.Ref) void {
    assert(operands.len <= Liveness.bpi - 1);
    var tomb_bits = func.liveness.getTombBits(inst);
    for (operands) |operand| {
        const lives = @truncate(u1, tomb_bits) == 0;
        tomb_bits >>= 1;
        if (lives) continue;
        func.processDeath(operand);
    }
    const lives = @truncate(u1, tomb_bits) == 0;
    if (lives and result != .none) {
        func.currentBranch().values.putAssumeCapacityNoClobber(Air.indexToRef(inst), result);
    }
    if (debug.runtime_safety) {
        func.air_bookkeeping += 1;
    }
}

fn resolveInst(func: *Func, ref: Air.Inst.Ref) !MV {
    log.debug("resolving instruction {} of type {}...", .{ ref, func.air.typeOf(ref).tag() });

    // does the instruction refer to a value in one of our local branches?
    // example for visualization:
    // ```
    // {
    //     a = 5;
    //     {
    //          b = 10;
    //          {
    //              // -> if we are here we have access to instruction a and b
    //          }
    //     }
    //     // -> if we are here we have access to instruction a but not b
    // }
    // ```
    var branch_index = func.branches.items.len;
    while (branch_index > 0) : (branch_index -= 1) {
        const last_branch = func.branches.items[branch_index - 1];
        if (last_branch.values.get(ref)) |value| {
            log.debug("resolved instruction {} as {}", .{ ref, value });
            return value;
        }
    }

    // otherwise, it is a constant

    const ty = func.air.typeOf(ref);
    log.debug("resolveInst: const ty: {}", .{ty.tag()});

    // if (!ty.hasRuntimeBits())
    //     return MV{ .none = {} };

    const val = func.air.value(ref).?;
    log.debug("lowering constant of type {} with tag {}...", .{ ty.tag(), ty.zigTypeTag() });

    return func.lowerConstant(val, ty);
}

fn getResolvedInstValue(func: *Func, inst: Air.Inst.Ref) MV {
    var i: usize = func.branches.items.len - 1;
    while (true) : (i -= 1) {
        if (func.branches.items[i].values.get(inst)) |mv| {
            assert(mv != .none);
            return mv;
        }
    }
}

/// Adds an MIR instruction to the output.
/// See `Mir.Inst.checkCombo` for an explanation on why `data` is `anytype`.
/// This function is the only point that adds instructions to the MIR output.
/// It has safety that ensures the following:
/// * It is impossible to generate invalid instructions.
/// * It is impossible to clobber registers unintentionally.
fn addInstInternal(func: *Func, tag: Mir.Inst.Tag, data: anytype) error{OutOfMemory}!void {
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
                .a => if (func.reg_a != null) assert(func.reg_a == func.current_inst),
                .x => if (func.reg_x != null) assert(func.reg_x == func.current_inst),
                .y => if (func.reg_y != null) assert(func.reg_y == func.current_inst),
            }
        }
    }
    try func.mir_instructions.append(func.gpa, inst);
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
            const temp = func.memory.allocZeroPageMemory(1).?; // TODO: handle error
            defer func.memory.free(temp, Type.u8);
            //try func.load(temp, val, Type.u8);
            try func.trans(val, temp, Type.u8, Type.u8);
            try func.addInstAny(mnemonic, temp);
        },
        .zp, .abs, .abs_unresolved => try func.addInstMem(mnemonic, val),
    }
}

fn fail(func: *Func, comptime fmt: []const u8, args: anytype) error{ CodegenFail, OutOfMemory } {
    @setCold(true);
    func.err_msg = try Module.ErrorMsg.create(func.gpa, func.src_loc, fmt, args);
    return error.CodegenFail;
}
