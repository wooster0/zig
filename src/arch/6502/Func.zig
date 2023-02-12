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

const std = @import("std");
const builtin = @import("builtin");
const debug = std.debug;
const panic = debug.panic;
const assert = debug.assert;
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
const Reg = bits.Reg;
const abi = @import("abi.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");
const AddrMem = @import("AddrMem.zig");
const RegMem = @import("RegMem.zig");

// TODO: open issues for these:
// * cycle-stepped Emulator.zig
// * basic pattern-based optimizations:
//   e.g. eliminate the second LDA in `sta $9FFF; lda $9FFF`
//   e.g. eliminate the first LDA in `lda #5; lda #6`
//   e.g. reduce `sta $02; sta $02; sta $02` to `sta $02`
//   e.g. replace JSR + RTS chain with JMP.
// * 3-byte exe with `pub fn main() void {}` and `-fno-basic-bootstrap`

const Func = @This();

//
// Input
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
// Call info
//

/// The locations of the arguments to this function.
args: []MV = undefined,
/// The index of the current argument that is being resolved in `airArg`.
arg_i: u16 = 0,
/// The result location of the return value of the current function.
ret_val: MV = undefined,

/// An error message for if codegen fails.
/// This is set if error.CodegenFail happens.
err_msg: *Module.ErrorMsg = undefined,

/// Runtime branches containing operands.
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
// Safety
//

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
// Memory
//
// TODO: let the user provide information about the register and flags' initial state so we don't have to assume the initial state is unknown?
//       what if we're embedded into something else? isn't there a 6502 CPU implemented in Minecraft? that way we can avoid setting these flags.

/// Addressable memory.
addr_mem: AddrMem,
/// Register memory.
reg_mem: RegMem = .{},

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

    const addr_mem = AddrMem.init(bin_file.options.target, bin_file);
    var func = Func{
        .bin_file = bin_file,
        .src_loc = src_loc,
        .props = props,
        .air = air,
        .liveness = liveness,
        .addr_mem = addr_mem,
    };
    defer func.deinit();

    func.gen() catch |err| switch (err) {
        error.CodegenFail => return Result{ .fail = func.err_msg },
        error.OutOfMemory => |other| return other,
    };

    const mir = Mir{
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

//
// Getters
//

/// Returns the allocator we will be using throughout.
pub fn getAllocator(func: Func) std.mem.Allocator {
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

/// Returns the byte size of a type or null if the byte size is too big.
/// Assert non-null in contexts where a type is involved where the compiler made sure the bit width is <= 65535.
fn getSize(func: Func, ty: Type) ?u16 {
    return std.math.cast(u16, ty.abiSize(func.getTarget()));
}

//
// Memory representation
//

/// Memory Value. This represents the location of a value in memory.
const MV = union(enum) {
    /// The value has no bits that we could represent at runtime.
    none: void,
    /// The value is this immediately available byte.
    imm: u8,
    /// The value is in this one-byte register.
    reg: Reg,
    /// The value is at this zero page address in memory.
    zp: u8,
    /// The value is at this absolute memory address in memory.
    abs: u16,
    /// The value is pointed to by an absolute memory address at this zero page address in memory.
    zp_abs: u8,
    /// The value is at an absolute memory address in memory that is yet to be resolved by the linker.
    abs_unres: Mir.Inst.UnresAbsAddr,

    fn eql(lhs: MV, rhs: MV) bool {
        return switch (lhs) {
            .none => rhs == .none,
            .imm => |imm| rhs == .imm and rhs.imm == imm,
            .reg => |reg| rhs == .reg and rhs.reg == reg,
            .zp => |addr| rhs == .zp and rhs.zp == addr,
            .abs => |addr| rhs == .abs and rhs.abs == addr,
            .zp_abs => |addr| rhs == .zp_abs and rhs.zp_abs == addr,
            .abs_unres => |abs_unres| rhs == .abs_unres and
                rhs.abs_unres.block_index == abs_unres.block_index and
                rhs.abs_unres.addend == abs_unres.addend,
        };
    }
};

const Branch = struct {
    inst_vals: std.AutoArrayHashMapUnmanaged(Air.Inst.Index, MV) = .{},

    fn deinit(branch: *Branch, allocator: std.mem.Allocator) void {
        branch.inst_vals.deinit(allocator);
    }
};
/// Returns the runtime branch we're in.
fn getCurrentBranch(func: *Func) *Branch {
    return &func.branches.items[func.branches.items.len - 1];
}

const CallMVs = struct {
    args: []MV,
    ret_val: MV,

    fn deinit(values: *CallMVs, allocator: std.mem.Allocator) void {
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
    // Reference: https://llvm-mos.org/wiki/C_calling_convention
    const cc = fn_ty.fnCallingConvention();
    const allocator = func.getAllocator();
    const param_types = try allocator.alloc(Type, fn_ty.fnParamLen());
    defer allocator.free(param_types);
    fn_ty.fnParamTypes(param_types);
    var values = CallMVs{
        .args = try allocator.alloc(MV, param_types.len),
        .ret_val = undefined,
    };
    errdefer allocator.free(values.args);

    switch (cc) {
        .Naked => {
            // Naked functions are not callable in normal code.
            assert(values.args.len == 0);
            values.ret_val = .{ .none = {} };
            return values;
        },
        .Unspecified, .C => {
            for (values.args) |*arg, i| {
                const param_ty = param_types[i];
                const param_size = func.getSize(param_ty).?;
                // TODO: preserve zero page addresses for this stuff once we have more of them
                // TODO: if size is > 3 bytes, allocate and pass pointer as A:X or something
                if (param_size != 1)
                    return try func.fail("TODO: support parameters bigger than 1 byte", .{});
                arg.* = switch (i) {
                    0 => .{ .reg = .a },
                    1 => .{ .reg = .x },
                    2 => .{ .reg = .y },
                    else => return try func.fail("TODO: support more than 3 parameters", .{}),
                };
            }
        },
        else => {
            const err = try func.fail("unsupported calling convention {}", .{cc});
            try func.failAddNote("supported are .Naked, .Unspecified, and .C", .{});
            return err;
        },
    }

    const ret_ty = fn_ty.fnReturnType();
    if (ret_ty.zigTypeTag() == .NoReturn or !ret_ty.hasRuntimeBits()) {
        values.ret_val = .{ .none = {} };
    } else {
        const ret_ty_size = func.getSize(ret_ty).?;
        if (ret_ty_size != 1)
            return try func.fail("TODO: support return values bigger than 1 byte", .{});
        values.ret_val = .{ .reg = .a };
    }

    return values;
}

//
// Memory allocation
//

/// Allocates addressable memory capable of storing a value of the given type, in any suitable address space.
fn allocAddrMem(func: *Func, ty: Type) !MV {
    if (!ty.hasRuntimeBitsIgnoreComptime()) {
        // TODO: return 0 address? think about this when implementing optionals
        return try func.fail("TODO: handle non-sized alloc", .{});
    }

    if (ty.zigTypeTag() == .Pointer) {
        // Pointers can only be dereferenced in the zero page so we need to put pointers there
        // and cannot allow the pointer to be allocated elsewhere.
        const val = func.addr_mem.alloc(2, .zp) orelse {
            const err = try func.fail("unable to allocate pointer due to zero page shortage", .{});
            try func.failAddNote("try reducing your program's pointer allocations", .{});
            return err;
        };
        log.debug("allocated zp ptr to {}", .{val});
        return .{ .zp_abs = val.zp };
    }

    // The type system guards against a number of cases where a type is too big to be represented:
    // * `var arr: [65536]u8 = undefined`
    // * TODO: list more
    // TODO: open issue for ` var a: u65535 = undefined;` and `var a: [65535]u65535 = undefined;` etc. overflowing here.
    //       fix this in src/type.zig and then see if we can assert this getSize to not fail.
    const size = func.getSize(ty) orelse return try func.fail("type {} too big to fit in address space", .{ty.fmt(func.getMod())});
    const val = func.addr_mem.alloc(size, .any) orelse {
        // TODO: test this error
        const err = try func.fail("program depleted all {} bytes of addressable memory", .{func.addr_mem.getMax(func.getTarget())});
        try func.failAddNote("try reducing your program's stack memory allocations", .{});
        return err;
    };
    log.debug("allocated {}B to {}", .{ size, val });
    const res = switch (val) {
        .zp => |addr| MV{ .zp = addr },
        .abs => |addr| MV{ .abs = addr },
    };
    return res;
}
/// Allocates register memory for storing a byte or null if the type is too big.
fn allocRegMem(
    func: *Func,
    ty: Type,
    /// The newly-allocated memory's owner.
    inst: Air.Inst.Index,
) !?MV {
    // TODO: remove error union from this function's return type after all TODOs are done
    if (!ty.hasRuntimeBitsIgnoreComptime()) {
        // TODO: return 0 address? think about this when implementing optionals
        return try func.fail("TODO: handle non-sized alloc", .{});
    }
    const bit_size = ty.bitSize(func.getTarget());
    if (bit_size > 8)
        return null;
    if (bit_size != 8) {
        // TODO: truncate bits (set to 0)
        return try func.fail("TODO: allow allocating non-8-bit type to reg", .{});
    }
    const reg = func.reg_mem.alloc(inst) orelse return null;
    return .{ .reg = reg };
}
/// Allocates addressable or register memory, depending on which is convenient.
fn allocAddrOrRegMem(
    func: *Func,
    ty: Type,
    /// The newly-allocated memory's owner.
    inst: Air.Inst.Index,
) !MV {
    // We need to make sure to allocate registers to the zero page as zp_abs
    // but the type will be 2 bytes (usize) so allocRegMem won't work and
    // it will happen in allocAddrMem.
    if (try func.allocRegMem(ty, inst)) |reg_mem|
        return reg_mem;
    return try func.allocAddrMem(ty);
}

//
// Register management
//

/// Spills the value in a register to addressable memory to free it for a different purpose.
pub fn spillReg(func: *Func, reg: Reg, inst: Air.Inst.Index) !void {
    if (func.liveness.isUnused(inst))
        return;
    const reg_val = func.getResolvedInst(inst);
    assert(reg_val == .reg);
    const ty = Type.u8;
    const new_home = try func.allocAddrOrRegMem(ty, inst);
    log.debug("spilling register {}'s {} into {}", .{ reg, reg_val, new_home });
    // DEPRECATED: try func.setAddrOrRegMem(ty, new_home, reg_val);
    try func.trans(reg_val, new_home, ty);
    func.getCurrentBranch().inst_vals.putAssumeCapacity(inst, new_home);
}
fn takeReg(func: *Func, reg: Reg, maybe_inst: ?Air.Inst.Index) MV {
    func.reg_mem.takeReg(reg, maybe_inst);
    return .{ .reg = reg };
}
fn freeReg(func: *Func, reg: Reg, maybe_inst: ?Air.Inst.Index) !MV {
    try func.reg_mem.freeReg(reg, maybe_inst);
    return .{ .reg = reg };
}
fn saveReg(func: *Func, reg: Reg) !RegMem.RegSave {
    return func.reg_mem.saveReg(reg);
}

//
// Memory interaction
//

//
// NOTE: the following is the new memory load-store architecture which unifies all memory operations into a single function.
//       This was due to a lot of logic repeating itself in the old memory load-store architecture (find it below `trans`).

/// Transfers from source to destination a value of the given type.
fn trans(func: *Func, src: MV, dst: MV, ty: Type) !void {
    const size = func.getSize(ty).?;
    log.debug("transferring {} to {} (size={}B,ty={})", .{ src, dst, size, ty.tag() });
    if (src.eql(dst))
        return;
    switch (src) {
        .none => {},
        .imm => |src_imm| {
            assert(size == 1);
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    switch (dst_reg) {
                        .a => try func.addInst(.lda_imm, .{ .imm = src_imm }),
                        .x => try func.addInst(.ldx_imm, .{ .imm = src_imm }),
                        .y => try func.addInst(.ldy_imm, .{ .imm = src_imm }),
                    }
                },
                .zp => |dst_addr| {
                    const reg_a = try func.saveReg(.a);
                    defer reg_a.restore();
                    try func.addInst(.lda_imm, .{ .imm = src_imm });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr });
                },
                .abs => |dst_addr| {
                    const reg_a = try func.saveReg(.a);
                    defer reg_a.restore();
                    try func.addInst(.lda_imm, .{ .imm = src_imm });
                    try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr } });
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() != .Pointer);
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    // TODO: either X or Y is fine here as both just have to be 0 for this to work.
                    //       take advantage of that and use the one that's most convenient.
                    const reg_x_save = try func.saveReg(.x);
                    defer reg_x_save.restore();
                    try func.addInst(.ldx_imm, .{ .imm = 0 });
                    try func.addInst(.lda_imm, .{ .imm = src_imm });
                    try func.addInst(.sta_x_ind_zp, .{ .zp = dst_addr });
                },
                .abs_unres => |dst_abs_unres| {
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    try func.addInst(.lda_imm, .{ .imm = src_imm });
                    try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_abs_unres } });
                },
            }
        },
        .reg => |src_reg| {
            assert(size == 1);
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    switch (src_reg) {
                        .a => {
                            switch (dst_reg) {
                                .a => {},
                                .x => try func.addInst(.tax_impl, .{ .none = {} }),
                                .y => try func.addInst(.tay_impl, .{ .none = {} }),
                            }
                        },
                        .x => {
                            switch (dst_reg) {
                                .a => try func.addInst(.txa_impl, .{ .none = {} }),
                                .x => {},
                                .y => unreachable, // TODO
                            }
                        },
                        .y => {
                            switch (dst_reg) {
                                .a => try func.addInst(.tya_impl, .{ .none = {} }),
                                .x => unreachable, // TODO
                                .y => {},
                            }
                        },
                    }
                },
                .zp => |dst_addr| {
                    switch (src_reg) {
                        .a => try func.addInst(.sta_zp, .{ .zp = dst_addr }),
                        .x => try func.addInst(.stx_zp, .{ .zp = dst_addr }),
                        .y => try func.addInst(.sty_zp, .{ .zp = dst_addr }),
                    }
                },
                .abs => |dst_addr| {
                    switch (src_reg) {
                        .a => try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr } }),
                        .x => try func.addInst(.stx_abs, .{ .abs = .{ .fixed = dst_addr } }),
                        .y => try func.addInst(.sty_abs, .{ .abs = .{ .fixed = dst_addr } }),
                    }
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() == .Pointer);
                    _ = dst_addr;
                    unreachable; // TODO
                },
                .abs_unres => |dst_unres| {
                    switch (src_reg) {
                        .a => try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_unres } }),
                        .x => try func.addInst(.stx_abs, .{ .abs = .{ .unres = dst_unres } }),
                        .y => try func.addInst(.sty_abs, .{ .abs = .{ .unres = dst_unres } }),
                    }
                },
            }
        },
        .zp => |src_addr| {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(size == 1);
                    switch (dst_reg) {
                        .a => try func.addInst(.lda_zp, .{ .zp = src_addr }),
                        .x => try func.addInst(.ldx_zp, .{ .zp = src_addr }),
                        .y => try func.addInst(.ldy_zp, .{ .zp = src_addr }),
                    }
                },
                .zp => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.lda_zp, .{ .zp = src_addr + i });
                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
                    }
                },
                .abs => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.lda_zp, .{ .zp = src_addr + i });
                        try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr + i } });
                    }
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() == .Pointer);
                    try func.addInst(.lda_imm, .{ .imm = src_addr });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
                    // Zero-extend the zero page address to be pointer-sized.
                    try func.addInst(.lda_imm, .{ .imm = 0 });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
                },
                .abs_unres => |dst_unres| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.lda_zp, .{ .zp = src_addr + i });
                        try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_unres.index(i) } });
                    }
                },
            }
        },
        .abs => |src_addr| {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(size == 1);
                    switch (dst_reg) {
                        .a => try func.addInst(.lda_abs, .{ .abs = .{ .fixed = src_addr } }),
                        .x => try func.addInst(.ldx_abs, .{ .abs = .{ .fixed = src_addr } }),
                        .y => try func.addInst(.ldy_abs, .{ .abs = .{ .fixed = src_addr } }),
                    }
                },
                .zp => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .fixed = src_addr + i } });
                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
                    }
                },
                .abs => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u16 = 0;
                    while (i < size) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .fixed = src_addr + i } });
                        try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr + i } });
                    }
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() == .Pointer);
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    const addr_bytes = @bitCast([2]u8, src_addr);
                    // Low byte first.
                    try func.addInst(.lda_imm, .{ .imm = addr_bytes[0] });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
                    // High byte second.
                    try func.addInst(.lda_imm, .{ .imm = addr_bytes[1] });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
                },
                .abs_unres => |dst_unres| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u16 = 0;
                    while (i < size) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .fixed = src_addr + i } });
                        try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_unres.index(i) } });
                    }
                },
            }
        },
        .zp_abs => |src_addr| {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(size == 1);
                    // TODO: either X or Y is fine here as both just have to be 0 for this to work.
                    //       take advantage of that and use the one that's most convenient.
                    const reg_y_save = try func.saveReg(.y);
                    defer reg_y_save.restore();
                    try func.addInst(.ldy_imm, .{ .imm = 0 });
                    switch (dst_reg) {
                        .a => try func.addInst(.lda_ind_y_zp, .{ .zp = src_addr }),
                        .x => unreachable, // TODO: try func.addInst(.ldx_ind_y_zp, .{ .zp = src_addr }),
                        .y => unreachable, // TODO: try func.addInst(.ldy_ind_y_zp, .{ .zp = src_addr }),
                    }
                },
                .zp => |dst_addr| {
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    const reg_y_save = try func.saveReg(.y);
                    defer reg_y_save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.ldy_imm, .{ .imm = i });
                        try func.addInst(.lda_ind_y_zp, .{ .zp = src_addr });
                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
                    }
                },
                .abs => |dst_addr| {
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    const reg_y_save = try func.saveReg(.y);
                    defer reg_y_save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.ldy_imm, .{ .imm = i });
                        try func.addInst(.lda_ind_y_zp, .{ .zp = src_addr });
                        try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr + i } });
                    }
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() == .Pointer);
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    try func.addInst(.lda_zp, .{ .zp = src_addr + 0 });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
                    try func.addInst(.lda_zp, .{ .zp = src_addr + 1 });
                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
                },
                .abs_unres => |dst_unres| {
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    const reg_y_save = try func.saveReg(.y);
                    defer reg_y_save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.ldy_imm, .{ .imm = i });
                        try func.addInst(.lda_ind_y_zp, .{ .zp = src_addr });
                        try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_unres.index(i) } });
                    }
                },
            }
        },
        .abs_unres => |src_unres| {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |dst_reg| {
                    assert(size == 1);
                    switch (dst_reg) {
                        .a => try func.addInst(.lda_abs, .{ .abs = .{ .unres = src_unres } }),
                        .x => try func.addInst(.ldx_abs, .{ .abs = .{ .unres = src_unres } }),
                        .y => try func.addInst(.ldy_abs, .{ .abs = .{ .unres = src_unres } }),
                    }
                },
                .zp => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .unres = src_unres.index(i) } });
                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
                    }
                },
                .abs => |dst_addr| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u16 = 0;
                    while (i < size) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .unres = src_unres.index(i) } });
                        try func.addInst(.sta_abs, .{ .abs = .{ .fixed = dst_addr + i } });
                    }
                },
                .zp_abs => |dst_addr| {
                    assert(ty.zigTypeTag() != .Pointer);
                    const reg_a_save = try func.saveReg(.a);
                    defer reg_a_save.restore();
                    const reg_y_save = try func.saveReg(.y);
                    defer reg_y_save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u8 = 0;
                    while (i < @intCast(u8, size)) : (i += 1) {
                        try func.addInst(.ldy_imm, .{ .imm = i });
                        try func.addInst(.lda_abs, .{ .abs = .{ .unres = src_unres.index(i) } });
                        try func.addInst(.sta_ind_y_zp, .{ .zp = dst_addr });
                    }
                },
                .abs_unres => |dst_unres| {
                    const save = try func.saveReg(.a);
                    defer save.restore();
                    // TODO: do this inlined memcpy at runtime at a certain threshold depending on optimize mode
                    var i: u16 = 0;
                    while (i < size) : (i += 1) {
                        try func.addInst(.lda_abs, .{ .abs = .{ .unres = src_unres.index(i) } });
                        try func.addInst(.sta_abs, .{ .abs = .{ .unres = dst_unres.index(i) } });
                    }
                },
            }
        },
    }
}

//
// NOTE: the following is the old, deprecated memory load-store architecture
//

//fn setAddrOrRegMem(func: *Func, ty: Type, dst: MV, val: MV) !void {
//    switch (dst) {
//        .none => {},
//        .imm => unreachable,
//        .reg => |reg| try func.setRegMem(ty, reg, val),
//        .zp, .abs => try func.setAddrMem(ty, dst, val),
//        .zp_abs => unreachable,
//        .abs_unresolved => unreachable,
//    }
//}
///// Sets addressable memory to a value of any size.
//fn setAddrMem(func: *Func, ty: Type, dst: MV, val: MV) !void {
//    const size = func.getSize(ty).?;
//    switch (dst) {
//        .none => unreachable,
//        .imm => unreachable,
//        .reg => unreachable,
//        .zp => |addr| {
//            switch (val) {
//                .none => assert(size == 0),
//                .imm => |imm| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    try func.addInst(.lda_imm, .{ .imm = imm });
//                    try func.addInst(.sta_zp, .{ .zp = addr });
//                },
//                .reg => |reg| {
//                    switch (reg) {
//                        .a => try func.addInst(.sta_zp, .{ .zp = addr }),
//                        .x => try func.addInst(.stx_zp, .{ .zp = addr }),
//                        .y => try func.addInst(.sty_zp, .{ .zp = addr }),
//                    }
//                },
//                .zp => unreachable, // TODO
//                .abs => unreachable, // TODO
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .abs => |dst_addr| {
//            switch (val) {
//                .none => assert(size == 0),
//                .imm => unreachable, // TODO
//                .reg => |reg| {
//                    switch (reg) {
//                        .a => try func.addInst(.sta_abs, .{ .abs = .{ .imm = dst_addr } }),
//                        .x => try func.addInst(.stx_abs, .{ .abs = .{ .imm = dst_addr } }),
//                        .y => try func.addInst(.sty_abs, .{ .abs = .{ .imm = dst_addr } }),
//                    }
//                },
//                .zp => unreachable, // TODO
//                .abs => unreachable, // TODO
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .zp_abs => unreachable, // TODO
//        .abs_unresolved => unreachable, // TODO
//    }
//}
///// Sets a register to a one-byte value.
//fn setRegMem(func: *Func, ty: Type, reg: Reg, val: MV) !void {
//    const bit_size = ty.bitSize(func.getTarget());
//    assert(bit_size <= 8);
//    switch (reg) {
//        .a => {
//            switch (val) {
//                .none => {},
//                .imm => |imm| try func.addInst(.lda_imm, .{ .imm = imm }),
//                .reg => |val_reg| {
//                    switch (val_reg) {
//                        .a => {},
//                        else => unreachable, // TODO
//                    }
//                },
//                .zp => |addr| try func.addInst(.lda_zp, .{ .zp = addr }),
//                .abs => |addr| try func.addInst(.lda_abs, .{ .abs = .{ .imm = addr } }),
//                .zp_abs => |addr| {
//                    // TODO: either X or Y is fine here as both just have to be 0 for this to work.
//                    //       take advantage of that and use the one that's most convenient.
//                    const reg_x_save = try func.saveReg(.x);
//                    defer reg_x_save.restore();
//                    try func.addInst(.ldx_imm, .{ .imm = 0 });
//                    try func.addInst(.lda_x_ind_zp, .{ .zp = addr });
//                },
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        else => unreachable, // TODO
//    }
//}
///// Writes a value to a pointer.
//fn storeToPtr(func: *Func, ptr: MV, val: MV, ptr_ty: Type, val_ty: Type) !void {
//    const size = func.getSize(val_ty).?;
//    assert(ptr_ty.childType().eql(val_ty, func.getMod()));
//    //_ = ptr_ty;
//    switch (ptr) {
//        .none => unreachable,
//        .imm => unreachable,
//        .reg => unreachable,
//        .zp => |ptr_addr| {
//            switch (val) {
//                .none => {},
//                .imm => |imm| {
//                    assert(size == 1);
//                    const reg_a_save = try func.saveReg(.a);
//                    defer reg_a_save.restore();
//                    try func.addInst(.lda_imm, .{ .imm = imm });
//                    //log.debug("val_ty: {}", .{val_ty.tag()});
//                    //if (val_ty.isPtrAtRuntime()) {
//                    //    const reg_x_save = try func.saveReg(.x);
//                    //    defer reg_x_save.restore();
//                    //    try func.addInst(.sta_x_ind_zp, .{ .zp = ptr_addr });
//                    //} else {
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr });
//                    //}
//                },
//                .reg => |reg| {
//                    switch (reg) {
//                        .a => try func.addInst(.sta_zp, .{ .zp = ptr_addr }),
//                        .x => try func.addInst(.stx_zp, .{ .zp = ptr_addr }),
//                        .y => try func.addInst(.sty_zp, .{ .zp = ptr_addr }),
//                    }
//                },
//                .zp => |val_addr| {
//                    // the problem is that zero page address (1 bytes) != usize (2 bytes)
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    //if (val_ty.isPtrAtRuntime()) {
//                    //    assert(size == 2);
//                    //    try func.addInst(.lda_imm, .{ .imm = val_addr });
//                    //    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 0 });
//                    //    // The size is 2 because of the usize type but it really is 1 because it's a zero page address,
//                    //    // so we zero-extend the zero page address to be pointer-sized.
//                    //    try func.addInst(.lda_imm, .{ .imm = 0 });
//                    //    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                    //} else {
//                    var i: u8 = 0;
//                    while (i < @intCast(u8, size)) : (i += 1) {
//                        //if (ptr_ty.childType().isPtrAtRuntime()) {
//                        //    try func.addInst(.lda_imm, .{ .imm = val_addr + i });
//                        //} else {
//                        //    try func.addInst(.lda_zp, .{ .zp = val_addr + i });
//                        //}
//                        //try func.addInst(.sta_zp, .{ .zp = ptr_addr + i });
//                        try func.addInst(.lda_zp, .{ .zp = val_addr + i });
//                        try func.addInst(.sta_zp, .{ .zp = ptr_addr + i });
//                    }
//                    //}
//                },
//                .abs => unreachable, // TODO
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .abs => |ptr_addr| {
//            switch (val) {
//                .none => {},
//                .imm => |imm| {
//                    assert(size == 1);
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    try func.addInst(.lda_imm, .{ .imm = imm });
//                    try func.addInst(.sta_abs, .{ .abs = .{ .imm = ptr_addr } });
//                },
//                .reg => unreachable, // TODO
//                .zp => |val_addr| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    var i: u8 = 0;
//                    while (i < @intCast(u8, size)) : (i += 1) {
//                        try func.addInst(.lda_zp, .{ .zp = val_addr + i });
//                        try func.addInst(.sta_abs, .{ .abs = .{ .imm = ptr_addr + i } });
//                    }
//                },
//                .abs => |val_addr| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    var i: u16 = 0;
//                    while (i < size) : (i += 1) {
//                        try func.addInst(.lda_abs, .{ .abs = .{ .imm = val_addr + i } });
//                        try func.addInst(.sta_abs, .{ .abs = .{ .imm = ptr_addr + i } });
//                    }
//                },
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => |abs_unresolved| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    var i: u16 = 0;
//                    while (i < size) : (i += 1) {
//                        try func.addInst(.lda_abs, .{ .abs = .{ .unresolved = abs_unresolved.index(i) } });
//                        try func.addInst(.sta_abs, .{ .abs = .{ .imm = ptr_addr + i } });
//                    }
//                },
//            }
//        },
//        .zp_abs => |ptr_addr| {
//            switch (val) {
//                .none => {},
//                .imm => |imm| {
//                    assert(size == 1);
//                    const reg_a_save = try func.saveReg(.a);
//                    defer reg_a_save.restore();
//                    // TODO: either X or Y is fine here as both just have to be 0 for this to work.
//                    //       take advantage of that and use the one that's most convenient.
//                    const reg_x_save = try func.saveReg(.x);
//                    defer reg_x_save.restore();
//                    try func.addInst(.ldx_imm, .{ .imm = 0 });
//                    try func.addInst(.lda_imm, .{ .imm = imm });
//                    try func.addInst(.sta_x_ind_zp, .{ .zp = ptr_addr });
//                },
//                .reg => unreachable, // TODO
//                .zp => |val_addr| {
//                    try func.addInst(.lda_imm, .{ .imm = val_addr });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 0 });
//                    // The size is 2 because of the usize type but it really is 1 because it's a zero page address,
//                    // so we zero-extend the zero page address to be pointer-sized.
//                    try func.addInst(.lda_imm, .{ .imm = 0 });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                },
//                .abs => |val_addr| {
//                    const addr_bytes = @bitCast([2]u8, val_addr);
//                    // Low byte first.
//                    try func.addInst(.lda_imm, .{ .imm = addr_bytes[1] });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 0 });
//                    // High byte second.
//                    try func.addInst(.lda_imm, .{ .imm = addr_bytes[0] });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                },
//                .zp_abs => |val_addr| {
//                    try func.addInst(.lda_zp, .{ .zp = val_addr + 0 });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 0 });
//                    try func.addInst(.lda_zp, .{ .zp = val_addr + 1 });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                },
//                .abs_unresolved => |val_abs_unresolved| {
//                    try func.addInst(.lda_abs, .{ .abs = .{ .unresolved = val_abs_unresolved.index(0) } });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 0 });
//                    try func.addInst(.lda_abs, .{ .abs = .{ .unresolved = val_abs_unresolved.index(1) } });
//                    try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                },
//            }
//        },
//        .abs_unresolved => unreachable, // TODO
//    }
//}
///// Reads a value from a pointer.
//fn loadFromPtr(func: *Func, dst: MV, ptr: MV, ptr_ty: Type) !void {
//    const child_ty = ptr_ty.childType();
//    const size = func.getSize(child_ty).?;
//    switch (ptr) {
//        .none => unreachable,
//        .imm => unreachable,
//        .reg => unreachable,
//        .zp => |ptr_addr| {
//            switch (dst) {
//                .none => assert(size == 0),
//                .imm => unreachable,
//                .reg => |reg| {
//                    assert(size == 1);
//                    switch (reg) {
//                        .a => try func.addInst(.lda_zp, .{ .zp = ptr_addr }),
//                        else => unreachable, // TODO
//                    }
//                },
//                .zp => |dst_addr| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    //if (child_ty.isPtrAtRuntime()) {
//                    //    assert(size == 2);
//                    //    // TODO: care about size
//                    //    //try func.addInst(.lda_x_ind_zp, .{ .zp = ptr_addr });
//                    //    //try func.addInst(.sta_zp, .{ .zp = dst_addr });
//                    //    //try func.addInst(.lda
//                    //    try func.addInst(.lda_imm, .{ .imm = ptr_addr });
//                    //    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
//                    //    try func.addInst(.lda_imm, .{ .imm = ptr_addr + 1 });
//                    //    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
//                    //} else {
//                    var i: u8 = 0;
//                    while (i < @intCast(u8, size)) : (i += 1) {
//                        //if (child_ty.isPtrAtRuntime()) {
//                        //    try func.addInst(.lda_imm, .{ .imm = ptr_addr + 1 });
//                        //} else {
//                        //    try func.addInst(.lda_zp, .{ .zp = ptr_addr + 1 });
//                        //}
//                        //try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
//                        try func.addInst(.lda_zp, .{ .zp = ptr_addr + i });
//                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
//                    }
//                    //}
//                },
//                .abs => unreachable, // TODO
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .abs => |ptr_addr| {
//            switch (dst) {
//                .none => assert(size == 0),
//                .imm => unreachable,
//                .reg => unreachable, // TODO
//                .zp => unreachable, // TODO
//                .abs => |dst_addr| {
//                    const save = try func.saveReg(.a);
//                    defer save.restore();
//                    var i: u16 = 0;
//                    while (i < size) : (i += 1) {
//                        try func.addInst(.lda_abs, .{ .abs = .{ .imm = ptr_addr + i } });
//                        try func.addInst(.sta_abs, .{ .abs = .{ .imm = dst_addr + i } });
//                    }
//                },
//                .zp_abs => unreachable, // TODO
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .zp_abs => |ptr_addr| {
//            switch (dst) {
//                .none => assert(size == 0),
//                .imm => unreachable,
//                .reg => unreachable, // TODO
//                .zp => |dst_addr| {
//                    const reg_a_save = try func.saveReg(.a);
//                    defer reg_a_save.restore();
//                    const reg_y_save = try func.saveReg(.y);
//                    defer reg_y_save.restore();
//                    //if (child_ty.isPtrAtRuntime()) {
//                    //    assert(size == 2);
//                    //    // TODO: care about size
//                    //    //try func.addInst(.lda_x_ind_zp, .{ .zp = ptr_addr });
//                    //    //try func.addInst(.sta_zp, .{ .zp = dst_addr });
//                    //    //try func.addInst(.lda
//                    //    try func.addInst(.lda_imm, .{ .imm = ptr_addr });
//                    //    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
//                    //    try func.addInst(.lda_imm, .{ .imm = ptr_addr + 1 });
//                    //    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
//                    //} else {
//                    var i: u8 = 0;
//                    while (i < @intCast(u8, size)) : (i += 1) {
//                        //if (child_ty.isPtrAtRuntime()) {
//                        //    try func.addInst(.lda_imm, .{ .imm = ptr_addr + 1 });
//                        //} else {
//                        //    try func.addInst(.lda_zp, .{ .zp = ptr_addr + 1 });
//                        //}}
//
//                        //try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
//                        try func.addInst(.ldy_imm, .{ .imm = i });
//                        try func.addInst(.lda_ind_y_zp, .{ .zp = ptr_addr });
//                        try func.addInst(.sta_zp, .{ .zp = dst_addr + i });
//                    }
//                    //}
//                },
//                .abs => unreachable, // TODO
//                .zp_abs => |dst_addr| {
//                    try func.addInst(.lda_zp, .{ .zp = ptr_addr + 0 });
//                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 0 });
//                    try func.addInst(.lda_zp, .{ .zp = ptr_addr + 1 });
//                    try func.addInst(.sta_zp, .{ .zp = dst_addr + 1 });
//                    //// The size is 2 because of the usize type but it really is 1 because it's a zero page address,
//                    //// so we zero-extend the zero page address to be pointer-sized.
//                    //try func.addInst(.lda_imm, .{ .imm = 0 });
//                    //try func.addInst(.sta_zp, .{ .zp = ptr_addr + 1 });
//                },
//                .abs_unresolved => unreachable, // TODO
//            }
//        },
//        .abs_unresolved => unreachable, // TODO
//    }
//}

//
// AIR instruction operand management
//

/// A lot of instructions have a constant amount of operands, but some have an unknown amount,
/// so this is used when we have more operands than `Liveness.bpi - 1`
/// (`- 1` to account for the bit that is for the instruction itself).
const BigTomb = struct {
    func: *Func,
    inst: Air.Inst.Index,
    lbt: Liveness.BigTomb,

    /// Feeds a liveness operand.
    fn feed(big_tomb: *BigTomb, op: Air.Inst.Ref, res: MV) void {
        const op_i = Air.refToIndex(op) orelse return; // Constants do not die.
        const dies = big_tomb.lbt.feed();
        if (!dies) return;
        big_tomb.func.processDeath(op_i, res);
    }

    /// Concludes liveness analysis for a runtime-known amount of operands of an AIR instruction.
    fn finishAir(big_tomb: *BigTomb, res: MV) void {
        const is_used = !big_tomb.func.liveness.isUnused(big_tomb.inst);
        if (is_used) {
            log.debug("%{} = {}", .{ big_tomb.inst, res });
            big_tomb.func.getCurrentBranch().inst_vals.putAssumeCapacityNoClobber(big_tomb.inst, res);
        }
        big_tomb.func.finishAirBookkeeping();
    }
};
fn finishAirBookkeeping(func: *Func) void {
    if (debug.runtime_safety)
        func.air_bookkeeping += 1;
}
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
    try func.getCurrentBranch().inst_vals.ensureUnusedCapacity(func.getAllocator(), additional_count);
}
/// Processes the death of a liveness operand of an AIR instruction.
/// The given result is the result of an AIR instruction.
/// It is used to prevent the unintentional death of an operand that happens to
/// be the result as well.
fn processDeath(func: *Func, op: Air.Inst.Index, res: MV) void {
    if (func.air.instructions.items(.tag)[op] == .constant) return; // Constants do not die.

    const dead = func.getResolvedInst(op);

    // In some cases (such as @bitCast), an operand
    // may be the same MV as the result.
    // in that case, prevent death and subsequent deallocation.
    if (dead.eql(res))
        return;

    func.getCurrentBranch().inst_vals.putAssumeCapacity(op, .none);
    log.debug("%{} = {}", .{ op, MV{ .none = {} } });

    switch (dead) {
        .none => unreachable,
        .imm => unreachable,
        .reg => |reg| _ = func.takeReg(reg, null),
        .zp, .abs => {
            // TODO: confirm that this is correct
            const ty = func.air.typeOfIndex(op);
            const size = func.getSize(ty).?;
            log.debug("deallocating {} ({}B of type {})", .{ dead, size, ty.tag() });
            const addr: AddrMem.Addr = switch (dead) {
                inline .zp, .zp_abs => |addr| .{ .zp = addr },
                .abs => |addr| .{ .abs = addr },
                else => unreachable,
            };
            func.addr_mem.free(addr, size);
        },
        .zp_abs => |addr| {
            // TODO: confirm that this is correct
            log.debug("deallocating ptr {}", .{dead});
            func.addr_mem.free(.{ .zp = addr }, 2);
        },
        .abs_unres => unreachable,
    }
}
/// Concludes liveness analysis for a comptime-known number of operands of an AIR instruction.
/// `- 1` in `Liveness.bpi - 1` accounts for the fact that one of the bits refers to the instruction itself.
/// If you have more than `ops.len` operands, use BigTomb.
fn finishAir(func: *Func, inst: Air.Inst.Index, res: MV, ops: []const Air.Inst.Ref) void {
    // Although the number of operands is comptime-known, we prefer doing it this way
    // so that we do not have to fill every non-existent operand space with `.none`.
    assert(ops.len <= Liveness.bpi - 1); // Use BigTomb if you need more than this.

    // The LSB is the first operand, and so on, up to `Liveness.bpi - 1` operands.
    var tomb_bits = func.liveness.getTombBits(inst);
    for (ops) |op| {
        const lives = @truncate(u1, tomb_bits) == 0;
        tomb_bits >>= 1;
        if (lives) continue;
        const op_i = Air.refToIndex(op) orelse
            // It's a constant.
            continue;
        func.processDeath(op_i, res);
    }

    // Shift more if we have not shifted enough yet.
    tomb_bits >>= @intCast(u2, (Liveness.bpi - 1) - ops.len);

    // The MSB is whether the instruction is unreferenced.
    const is_used = @truncate(u1, tomb_bits) == 0;
    if (is_used) {
        log.debug("%{} = {}", .{ inst, res });
        func.getCurrentBranch().inst_vals.putAssumeCapacityNoClobber(inst, res);
    }

    func.finishAirBookkeeping();
}
fn lowerConstant(func: *Func, const_val: Value, ty: Type) !MV {
    var val = const_val;
    if (val.castTag(.runtime_value)) |sub_value| {
        val = sub_value.data;
    }

    if (val.castTag(.decl_ref)) |decl_ref| {
        //const decl_index = decl_ref.data;
        //return func.lowerDeclRef(val, ty, decl_index);
        _ = decl_ref;
        unreachable; // TODO
    }
    if (val.castTag(.decl_ref_mut)) |decl_ref_mut| {
        //const decl_index = decl_ref_mut.data.decl_index;
        //return func.lowerDeclRef(val, ty, decl_index);
        _ = decl_ref_mut;
        unreachable; // TODO
    }

    // Try lowering the constant directly to an MV or else if it doesn't fit,
    // fall through and let the other logic handle it.
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
                    const addr = val.toUnsignedInt(target);
                    if (std.math.cast(u8, addr)) |zp_addr|
                        return MV{ .zp = zp_addr }
                    else
                        return MV{ .abs = @intCast(u16, addr) };
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

    // The value is not immediately representable as an MV so emit it to the binary.
    return try func.lowerUnnamedConst(val, ty);
}
fn lowerUnnamedConst(func: *Func, val: Value, ty: Type) !MV {
    log.debug("lowerUnnamedConst: ty = {}, val = {}", .{ val.fmtValue(ty, func.getMod()), ty.fmt(func.getMod()) });
    const block_index = @intCast(
        u16,
        func.bin_file.lowerUnnamedConst(.{ .ty = ty, .val = val }, func.getDeclIndex()) catch |err| {
            return try func.fail("lowering unnamed constant failed: {}", .{err});
        },
    );
    if (func.bin_file.cast(link.File.Prg)) |_| {
        return MV{ .abs_unres = .{ .block_index = block_index } };
    } else unreachable;
}
fn resolveInst(func: *Func, inst: Air.Inst.Ref) !MV {
    // The first section of indexes correspond to a set number of constant values.
    const ref_int = @enumToInt(inst);
    if (ref_int < Air.Inst.Ref.typed_value_map.len) {
        const tv = Air.Inst.Ref.typed_value_map[ref_int];
        if (!tv.ty.hasRuntimeBitsIgnoreComptime()) {
            return MV{ .none = {} };
        }
        return func.lowerConstant(tv.val, tv.ty);
    }

    // If the type has no codegen bits, no need to store it.
    const inst_ty = func.air.typeOf(inst);
    if (!inst_ty.hasRuntimeBitsIgnoreComptime())
        return MV{ .none = {} };

    const inst_index = @intCast(Air.Inst.Index, ref_int - Air.Inst.Ref.typed_value_map.len);
    switch (func.air.instructions.items(.tag)[inst_index]) {
        .constant => {
            // Constants have static lifetimes, so they are always memoized in the outer most table.
            const branch = &func.branches.items[0];
            const gop = try branch.inst_vals.getOrPut(func.getAllocator(), inst_index);
            if (!gop.found_existing) {
                const ty_pl = func.air.instructions.items(.data)[inst_index].ty_pl;
                assert(inst_ty.eql(func.air.getRefType(ty_pl.ty), func.getMod()));
                gop.value_ptr.* = try func.lowerConstant(func.air.values[ty_pl.payload], inst_ty);
            }
            return gop.value_ptr.*;
        },
        .const_ty => unreachable,
        else => return func.getResolvedInst(inst_index),
    }
}
fn getResolvedInst(func: *Func, inst: Air.Inst.Index) MV {
    // Check whether the instruction refers to a value in one of our local branches.
    // Example for visualization:
    // ```
    // %1 = x;
    // {
    //      %2 = x;
    //      // -> If we are here we have access to instruction %1 and %2
    // }
    // // -> If we are here we have access to instruction %1 but not %2
    // ```
    var i = func.branches.items.len - 1;
    while (true) : (i -= 1) {
        if (func.branches.items[i].inst_vals.get(inst)) |val| {
            assert(val != .none);
            return val;
        }
    }
}
/// Attempts to reuse an operand for a different purpose if it would die anyway.
fn reuseOperand(func: *Func, inst: Air.Inst.Index, operand: Air.Inst.Ref, op_index: Liveness.OperandInt, val: MV) bool {
    if (!func.liveness.operandDies(inst, op_index))
        return false;

    switch (val) {
        .none => unreachable,
        .imm => unreachable,
        .reg => |reg| {
            _ = func.takeReg(reg, inst);
            log.debug("%{} = {} (reused)", .{ inst, reg });
        },
        .zp, .abs => {
            log.debug("%{} = {} (reused)", .{ inst, val });
        },
        .zp_abs => {
            log.debug("%{} = {} (reused)", .{ inst, val });
        },
        .abs_unres => unreachable,
    }

    // Prevent the operand death's processing code from deallocating it.
    func.liveness.clearOperandDeath(inst, op_index);

    // This makes us responsible for doing the rest of the stuff that processDeath would have done.
    func.getCurrentBranch().inst_vals.putAssumeCapacity(Air.refToIndex(operand).?, .none);

    return true;
}

//
// MIR instruction output
//

/// Adds an MIR instruction to the output.
/// See `Mir.Inst.checkCombo` for an explanation on why `data` is `anytype`.
/// It has safety that ensures the following:
/// * It is impossible to generate invalid instructions.
/// * It is impossible to clobber registers unintentionally.
pub fn addInst(func: *Func, tag: Mir.Inst.Tag, data: anytype) !void {
    // TODO: try doing the checkCombo at comptime by making `tag` and `data` comptime
    //       (as of writing, making `tag` and `data` comptime crashes the compiler)
    const inst = Mir.Inst{ .tag = tag, .data = @as(Mir.Inst.Data, data) };
    if (debug.runtime_safety) {
        // Prevent generating invalid instructions.
        Mir.Inst.checkCombo(tag, data);
        // Prevent clobbering registers unintentionally.
        func.reg_mem.checkInst(inst);
    }
    try func.mir_instructions.append(func.getAllocator(), inst);
}
/// Returns the last MIR instruction added.
fn getCurrentInst(func: *Func) struct { tag: *Mir.Inst.Tag, data: *Mir.Inst.Data } {
    return .{
        .tag = &func.mir_instructions.items(.tag)[func.mir_instructions.len - 1],
        .data = &func.mir_instructions.items(.data)[func.mir_instructions.len - 1],
    };
}

fn fail(func: *Func, comptime fmt: []const u8, args: anytype) !error{CodegenFail} {
    @setCold(true);
    func.err_msg = try Module.ErrorMsg.create(func.getAllocator(), func.src_loc, fmt, args);
    return error.CodegenFail;
}
fn failAddNote(func: *Func, comptime fmt: []const u8, args: anytype) !void {
    @setCold(true);
    func.err_msg.notes = try func.getAllocator().realloc(func.err_msg.notes, func.err_msg.notes.len + 1);
    func.err_msg.notes[func.err_msg.notes.len - 1] = (try Module.ErrorMsg.create(func.getAllocator(), func.src_loc, fmt, args)).*;
}

//
// MIR generation
//

fn gen(func: *Func) !void {
    const allocator = func.getAllocator();
    var call_vals = try func.resolveCallingConventionValues(func.getType());
    defer call_vals.deinit(allocator);
    func.args = call_vals.args;
    func.ret_val = call_vals.ret_val;

    try func.branches.append(allocator, .{});
    defer {
        var outer_branch = func.branches.pop();
        outer_branch.deinit(allocator);
    }

    try func.genBody(func.air.getMainBody());

    // Preserve memory state for next function codegen.
    func.addr_mem.preserve();
}

fn genBody(func: *Func, body: []const Air.Inst.Index) !void {
    for (body) |inst| {
        const old_air_bookkeeping = func.air_bookkeeping;

        if (debug.runtime_safety)
            func.air_current_inst = inst;

        var i: u16 = @intCast(u16, func.mir_instructions.len);
        try func.genInst(inst);
        while (i < func.mir_instructions.len) : (i += 1) {
            const mir_inst = Mir.Inst{
                .tag = func.mir_instructions.items(.tag)[i],
                .data = func.mir_instructions.items(.data)[i],
            };
            var buf: [20]u8 = undefined;
            log.debug("-> {s: <20} (.{s})", .{ mir_inst.getTextRepr(&buf), @tagName(mir_inst.tag) });
        }
        log.debug("", .{});

        if (debug.runtime_safety) {
            if (func.air_bookkeeping <= old_air_bookkeeping) {
                debug.panic("missing `finishAir` call for AIR instruction `{}`", .{
                    func.air.instructions.items(.tag)[inst],
                });
            }
        }
    }
}

fn genInst(func: *Func, inst: Air.Inst.Index) !void {
    try func.ensureProcessDeathCapacity(Liveness.bpi);
    const tag = func.air.instructions.items(.tag)[inst];
    log.debug("lowering {} (%{})...", .{ tag, inst });
    return switch (tag) {
        .arg => try func.airArg(inst),
        .add => try func.fail("TODO: implement add", .{}),
        .add_optimized => try func.fail("TODO: implement add_optimized", .{}),
        .addwrap => try func.fail("TODO: implement addwrap", .{}),
        .addwrap_optimized => try func.fail("TODO: implement addwrap_optimized", .{}),
        .add_sat => try func.fail("TODO: implement add_sat", .{}),
        .sub => try func.fail("TODO: implement sub", .{}),
        .sub_optimized => try func.fail("TODO: implement sub_optimized", .{}),
        .subwrap => try func.fail("TODO: implement subwrap", .{}),
        .subwrap_optimized => try func.fail("TODO: implement subwrap_optimized", .{}),
        .sub_sat => try func.fail("TODO: implement sub_sat", .{}),
        .mul => try func.fail("TODO: implement mul", .{}),
        .mul_optimized => try func.fail("TODO: implement mul_optimized", .{}),
        .mulwrap => try func.fail("TODO: implement mulwrap", .{}),
        .mulwrap_optimized => try func.fail("TODO: implement mulwrap_optimized", .{}),
        .mul_sat => try func.fail("TODO: implement mul_sat", .{}),
        .div_float => try func.fail("TODO: implement div_float", .{}),
        .div_float_optimized => try func.fail("TODO: implement div_float_optimized", .{}),
        .div_trunc => try func.fail("TODO: implement div_trunc", .{}),
        .div_trunc_optimized => try func.fail("TODO: implement div_trunc_optimized", .{}),
        .div_floor => try func.fail("TODO: implement div_floor", .{}),
        .div_floor_optimized => try func.fail("TODO: implement div_floor_optimized", .{}),
        .div_exact => try func.fail("TODO: implement div_exact", .{}),
        .div_exact_optimized => try func.fail("TODO: implement div_exact_optimized", .{}),
        .rem => try func.fail("TODO: implement rem", .{}),
        .rem_optimized => try func.fail("TODO: implement rem_optimized", .{}),
        .mod => try func.fail("TODO: implement mod", .{}),
        .mod_optimized => try func.fail("TODO: implement mod_optimized", .{}),
        .ptr_add => try func.fail("TODO: implement ptr_add", .{}),
        .ptr_sub => try func.fail("TODO: implement ptr_sub", .{}),
        // TODO: wait for https://github.com/ziglang/zig/issues/14039 to be implemented first
        //       before lowering the following two.
        .max => try func.fail("TODO: implement max", .{}),
        .min => try func.fail("TODO: implement min", .{}),
        .add_with_overflow => try func.fail("TODO: implement add_with_overflow", .{}),
        .sub_with_overflow => try func.fail("TODO: implement sub_with_overflow", .{}),
        .mul_with_overflow => try func.fail("TODO: implement mul_with_overflow", .{}),
        .shl_with_overflow => try func.fail("TODO: implement shl_with_overflow", .{}),
        .alloc => try func.airAlloc(inst),
        .ret_ptr => try func.fail("TODO: implement ret_ptr", .{}),
        // TODO: this tag should be called .asm
        .assembly => try func.airAsm(inst),
        .bit_and => try func.fail("TODO: implement bit_and", .{}),
        .bit_or => try func.fail("TODO: implement bit_or", .{}),
        .shr => try func.fail("TODO: implement shr", .{}),
        .shr_exact => try func.fail("TODO: implement shr_exact", .{}),
        .shl => try func.fail("TODO: implement shl", .{}),
        .shl_exact => try func.fail("TODO: implement shl_exact", .{}),
        .shl_sat => try func.fail("TODO: implement shl_sat", .{}),
        .xor => try func.fail("TODO: implement xor", .{}),
        .not => try func.fail("TODO: implement not", .{}),
        // TODO: this tag should be called .bit_cast
        .bitcast => try func.airBitCast(inst),
        .block => try func.fail("TODO: implement block", .{}),
        .loop => try func.fail("TODO: implement loop", .{}),
        .br => try func.fail("TODO: implement br", .{}),
        .breakpoint => try func.airBreakpoint(inst),
        .ret_addr => try func.fail("TODO: implement ret_addr", .{}),
        .frame_addr => try func.fail("TODO: implement frame_addr", .{}),
        .call => try func.airCall(inst, .auto),
        .call_always_tail => try func.airCall(inst, .always_tail),
        .call_never_tail => try func.airCall(inst, .never_tail),
        .call_never_inline => try func.airCall(inst, .never_inline),
        // TODO: can't we simply use compiler_rt for a lot of these?
        //       on another note, am I supposed to use the
        //       https://github.com/ziglang/zig/blob/master/lib/std/math/big/int.zig
        //       stuff to implement integers with bit count > 128?
        .clz => try func.fail("TODO: implement clz", .{}),
        .ctz => try func.fail("TODO: implement ctz", .{}),
        // TODO: this tag should be called .pop_count
        .popcount => try func.fail("TODO: implement popcount", .{}), // TODO: for this one can't we use lib/compiler_rt/popcount.zig for example?
        .byte_swap => try func.fail("TODO: implement byte_swap", .{}),
        .bit_reverse => try func.fail("TODO: implement bit_reverse", .{}),

        .sqrt => try func.fail("TODO: implement sqrt", .{}),
        .sin => try func.fail("TODO: implement sin", .{}),
        .cos => try func.fail("TODO: implement cos", .{}),
        .tan => try func.fail("TODO: implement tan", .{}),
        .exp => try func.fail("TODO: implement exp", .{}),
        .exp2 => try func.fail("TODO: implement exp2", .{}),
        .log => try func.fail("TODO: implement log", .{}),
        .log2 => try func.fail("TODO: implement log2", .{}),
        .log10 => try func.fail("TODO: implement log10", .{}),
        .fabs => try func.fail("TODO: implement fabs", .{}),
        .floor => try func.fail("TODO: implement floor", .{}),
        .ceil => try func.fail("TODO: implement ceil", .{}),
        .round => try func.fail("TODO: implement round", .{}),
        .trunc_float => try func.fail("TODO: implement trunc_float", .{}),
        .neg => try func.fail("TODO: implement neg", .{}),
        .neg_optimized => try func.fail("TODO: implement neg_optimized", .{}),

        .cmp_lt => try func.fail("TODO: implement cmp_lt", .{}),
        .cmp_lt_optimized => try func.fail("TODO: implement cmp_lt_optimized", .{}),
        .cmp_lte => try func.fail("TODO: implement cmp_lte", .{}),
        .cmp_lte_optimized => try func.fail("TODO: implement cmp_lte_optimized", .{}),
        .cmp_eq => try func.fail("TODO: implement cmp_eq", .{}),
        .cmp_eq_optimized => try func.fail("TODO: implement cmp_eq_optimized", .{}),
        .cmp_gte => try func.fail("TODO: implement cmp_gte", .{}),
        .cmp_gte_optimized => try func.fail("TODO: implement cmp_gte_optimized", .{}),
        .cmp_gt => try func.fail("TODO: implement cmp_gt", .{}),
        .cmp_gt_optimized => try func.fail("TODO: implement cmp_gt_optimized", .{}),
        .cmp_neq => try func.fail("TODO: implement cmp_neq", .{}),
        .cmp_neq_optimized => try func.fail("TODO: implement cmp_neq_optimized", .{}),
        .cmp_vector => try func.fail("TODO: implement cmp_vector", .{}),
        .cmp_vector_optimized => try func.fail("TODO: implement cmp_vector_optimized", .{}),

        .cond_br => try func.fail("TODO: implement cond_br", .{}),
        .switch_br => try func.fail("TODO: implement switch_br", .{}),
        .@"try" => try func.fail("TODO: implement try", .{}),
        .try_ptr => try func.fail("TODO: implement try_ptr", .{}),
        .constant => unreachable, // Excluded from function bodies.
        .const_ty => unreachable, // Excluded from function bodies.
        .dbg_stmt => try func.airDbgStmt(inst),
        .dbg_block_begin => try func.airDbgBlock(inst),
        .dbg_block_end => try func.airDbgBlock(inst),
        .dbg_inline_begin => try func.airDbgInline(inst),
        .dbg_inline_end => try func.airDbgInline(inst),
        .dbg_var_ptr => try func.airDbgLocal(inst),
        .dbg_var_val => try func.airDbgLocal(inst),
        .is_null => try func.fail("TODO: implement is_null", .{}),
        .is_non_null => try func.fail("TODO: implement is_non_null", .{}),
        .is_null_ptr => try func.fail("TODO: implement is_null_ptr", .{}),
        .is_non_null_ptr => try func.fail("TODO: implement is_non_null_ptr", .{}),
        .is_err => try func.fail("TODO: implement is_err", .{}),
        .is_non_err => try func.fail("TODO: implement is_non_err", .{}),
        .is_err_ptr => try func.fail("TODO: implement is_err_ptr", .{}),
        .is_non_err_ptr => try func.fail("TODO: implement is_non_err_ptr", .{}),
        .bool_and => try func.fail("TODO: implement bool_and", .{}),
        .bool_or => try func.fail("TODO: implement bool_or", .{}),
        .load => try func.airLoad(inst),
        .ptrtoint => try func.fail("TODO: implement ptrtoint", .{}),
        .bool_to_int => try func.fail("TODO: implement bool_to_int", .{}),
        .ret => try func.airRet(inst),
        .ret_load => try func.fail("TODO: implement ret_load", .{}),
        .store => try func.airStore(inst),
        .unreach => try func.airUnreach(inst),
        .fptrunc => try func.fail("TODO: implement fptrunc", .{}),
        .fpext => try func.fail("TODO: implement fpext", .{}),
        // TODO: this tag should be called .int_cast
        .intcast => try func.airIntCast(inst),
        .trunc => try func.fail("TODO: implement trunc", .{}),
        .optional_payload => try func.fail("TODO: implement optional_payload", .{}),
        .optional_payload_ptr => try func.fail("TODO: implement optional_payload_ptr", .{}),
        .optional_payload_ptr_set => try func.fail("TODO: implement optional_payload_ptr_set", .{}),
        .wrap_optional => try func.fail("TODO: implement wrap_optional", .{}),
        .unwrap_errunion_payload => try func.fail("TODO: implement unwrap_errunion_payload", .{}),
        .unwrap_errunion_err => try func.fail("TODO: implement unwrap_errunion_err", .{}),
        .unwrap_errunion_payload_ptr => try func.fail("TODO: implement unwrap_errunion_payload_ptr", .{}),
        .unwrap_errunion_err_ptr => try func.fail("TODO: implement unwrap_errunion_err_ptr", .{}),
        .errunion_payload_ptr_set => try func.fail("TODO: implement errunion_payload_ptr_set", .{}),
        .wrap_errunion_payload => try func.fail("TODO: implement wrap_errunion_payload", .{}),
        .wrap_errunion_err => try func.fail("TODO: implement wrap_errunion_err", .{}),
        .struct_field_ptr => try func.fail("TODO: implement struct_field_ptr", .{}),
        .struct_field_ptr_index_0 => try func.fail("TODO: implement struct_field_ptr_index_0", .{}),
        .struct_field_ptr_index_1 => try func.fail("TODO: implement struct_field_ptr_index_1", .{}),
        .struct_field_ptr_index_2 => try func.fail("TODO: implement struct_field_ptr_index_2", .{}),
        .struct_field_ptr_index_3 => try func.fail("TODO: implement struct_field_ptr_index_3", .{}),
        .struct_field_val => try func.fail("TODO: implement struct_field_val", .{}),
        .set_union_tag => try func.fail("TODO: implement set_union_tag", .{}),
        .get_union_tag => try func.fail("TODO: implement get_union_tag", .{}),
        .slice => try func.fail("TODO: implement slice", .{}),
        .slice_len => try func.fail("TODO: implement slice_len", .{}),
        .slice_ptr => try func.fail("TODO: implement slice_ptr", .{}),
        .ptr_slice_len_ptr => try func.fail("TODO: implement ptr_slice_len_ptr", .{}),
        .ptr_slice_ptr_ptr => try func.fail("TODO: implement ptr_slice_ptr_ptr", .{}),
        .array_elem_val => try func.fail("TODO: implement array_elem_val", .{}),
        .slice_elem_val => try func.fail("TODO: implement slice_elem_val", .{}),
        .slice_elem_ptr => try func.fail("TODO: implement slice_elem_ptr", .{}),
        .ptr_elem_val => try func.fail("TODO: implement ptr_elem_val", .{}),
        .ptr_elem_ptr => try func.fail("TODO: implement ptr_elem_ptr", .{}),
        .array_to_slice => try func.fail("TODO: implement array_to_slice", .{}),
        .float_to_int => try func.fail("TODO: implement float_to_int", .{}),
        .float_to_int_optimized => try func.fail("TODO: implement float_to_int_optimized", .{}),
        .int_to_float => try func.fail("TODO: implement int_to_float", .{}),

        .reduce => try func.fail("TODO: implement reduce", .{}),
        .reduce_optimized => try func.fail("TODO: implement reduce_optimized", .{}),
        .splat => try func.fail("TODO: implement splat", .{}),
        .shuffle => try func.fail("TODO: implement shuffle", .{}),
        .select => try func.fail("TODO: implement select", .{}),

        // TODO: wait for https://github.com/ziglang/zig/issues/14040 to be implemented first
        //       before lowering the following two.
        .memset => try func.fail("TODO: implement memset", .{}),
        .memcpy => try func.fail("TODO: implement memcpy", .{}),

        .cmpxchg_weak => try func.fail("TODO: implement cmpxchg_weak", .{}),
        .cmpxchg_strong => try func.fail("TODO: implement cmpxchg_strong", .{}),
        .fence => try func.fail("TODO: implement fence", .{}),
        .atomic_load => try func.fail("TODO: implement atomic_load", .{}),
        .atomic_store_unordered => try func.fail("TODO: implement atomic_store_unordered", .{}),
        .atomic_store_monotonic => try func.fail("TODO: implement atomic_store_monotonic", .{}),
        .atomic_store_release => try func.fail("TODO: implement atomic_store_release", .{}),
        .atomic_store_seq_cst => try func.fail("TODO: implement atomic_store_seq_cst", .{}),
        .atomic_rmw => try func.fail("TODO: implement atomic_rmw", .{}),

        .is_named_enum_value => try func.fail("TODO: implement is_named_enum_value", .{}),

        .tag_name => try func.fail("TODO: implement tag_name", .{}),

        .error_name => try func.fail("TODO: implement error_name", .{}),

        .error_set_has_value => try func.fail("TODO: implement error_set_has_value", .{}),

        .aggregate_init => try func.fail("TODO: implement aggregate_init", .{}),

        .union_init => try func.fail("TODO: implement union_init", .{}),

        .prefetch => try func.fail("TODO: implement prefetch", .{}),

        .mul_add => try func.fail("TODO: implement mul_add", .{}),

        .field_parent_ptr => try func.fail("TODO: implement field_parent_ptr", .{}),

        .wasm_memory_size => try func.fail("TODO: implement wasm_memory_size", .{}),
        .wasm_memory_grow => try func.fail("TODO: implement wasm_memory_grow", .{}),

        .cmp_lt_errors_len => try func.fail("TODO: implement cmp_lt_errors_len", .{}),

        .err_return_trace => try func.fail("TODO: implement err_return_trace", .{}),

        .set_err_return_trace => try func.fail("TODO: implement set_err_return_trace", .{}),

        .addrspace_cast => try func.fail("TODO: implement addrspace_cast", .{}),

        .save_err_return_trace_index => try func.fail("TODO: implement save_err_return_trace_index", .{}),

        .vector_store_elem => try func.fail("TODO: implement vector_store_elem", .{}),

        .c_va_arg => try func.fail("TODO: implement c_va_arg", .{}),
        .c_va_copy => try func.fail("TODO: implement c_va_copy", .{}),
        .c_va_end => try func.fail("TODO: implement c_va_end", .{}),
        .c_va_start => try func.fail("TODO: implement c_va_start", .{}),
    };
}

fn airArg(func: *Func, inst: Air.Inst.Index) !void {
    const air_arg = func.air.instructions.items(.data)[inst].arg;
    const arg_ty = func.air.typeOf(air_arg.ty);
    _ = arg_ty;
    const arg_src_index = air_arg.src_index;
    _ = arg_src_index;

    const mv = func.args[func.arg_i];
    func.arg_i += 1;

    switch (mv) {
        .reg => |reg| _ = func.takeReg(reg, inst),
        else => {},
    }

    func.finishAir(inst, mv, &.{});
}

fn airAlloc(func: *Func, inst: Air.Inst.Index) !void {
    // We have two options here:
    //
    // 1. Allocate technically correct stack-local memory by using only the hardware stack and registers,
    //    which only gives us a bit more than 256 bytes per function.
    //    To do this, we would use PHA and PLA to push data and store/load bytes using specific offsets into the second page of memory,
    //    using TSX and TXS to transfer from and to the stack pointer, so the system itself would "clean up" us because it's true stack memory.
    //    This is how all other codegen backends do this and it works for them because of how big their stack is.
    //    Choosing this option means we do not have access to the zero page which is crucial for
    //    efficient code generation and necessary for dereferencing pointers.
    // 2. Allocate from zero page, absolute memory, and registers,
    //    which gives us access to all memory at the cost that it is harder to manage
    //    and we need to keep memory state intact across all function codegens and we have to clean up memory.
    //    This creates a unique problem no other codegen backend has.
    //    See also: AddrMem.preserve
    //    TODO: talk about how heap memory allocation plays into this.
    //          how does one write a heap memory allocator for a 6502 target?
    //          1. use the OS? does the C64 or any OS have some CBM kernal routine to allocate heap memory?
    //          2. provide a way for the user to reserve a specific number of bytes that it can use in its heap.
    //             for example, how do I tell the backend I want 0x0800 bytes of the 64 KiB available ones for my heap?
    //             would we need a new builtin for this? ref: @wasmMemorySize and @wasmMemoryGrow
    //
    // We choose the second option.
    const ptr_ty = func.air.instructions.items(.data)[inst].ty;
    const child_ty = ptr_ty.childType();
    // Allocate to addressable memory but not registers because they're already sparsely available,
    // and it would violate the specifications of the AIR instructions because they talk about pointers and not registers.
    const res = try func.allocAddrMem(child_ty);
    func.finishAir(inst, res, &.{});
}

fn airAsm(func: *Func, inst: Air.Inst.Index) !void {
    const ty_pl = func.air.instructions.items(.data)[inst].ty_pl;
    const ty = func.air.getRefType(ty_pl.ty);
    assert(ty.tag() == .void);
    const extra = func.air.extraData(Air.Asm, ty_pl.payload);
    var extra_i = extra.end;
    const outputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.outputs_len]);
    assert(outputs.len == 0); // TODO: support outputting specific registers into variables (so, the other way around)
    extra_i += outputs.len;
    const inputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.inputs_len]);
    extra_i += inputs.len;

    assert(extra.data.clobbers_len() == 0); // TODO

    const res: MV = if (!extra.data.is_volatile() and func.liveness.isUnused(inst))
        .none
    else res: {
        for (inputs) |input| {
            const input_bytes = std.mem.sliceAsBytes(func.air.extra[extra_i..]);
            const constraint = std.mem.sliceTo(input_bytes, 0);
            const name = std.mem.sliceTo(input_bytes[constraint.len + 1 ..], 0);

            // This equation accounts for the fact that even if we have exactly 4 bytes
            // for the string, we still use the next u32 for the null terminator.
            extra_i += (constraint.len + name.len + (2 + 3)) / 4;

            if (constraint.len < 3 or constraint[0] != '{' or constraint[constraint.len - 1] != '}') {
                return try func.fail("unknown `asm` input constraint: \"{s}\"", .{constraint});
            }
            const reg_name = constraint[1 .. constraint.len - 1];
            const reg = Reg.parse(reg_name) orelse {
                const err = try func.fail("unknown register \"{s}\"", .{reg_name});
                try func.failAddNote("possible registers are A, X, and Y", .{});
                return err;
            };
            const input_val = try func.resolveInst(input);
            // TODO: support bit sizes <= 8 and support `comptime_int`s?
            const input_ty = func.air.typeOf(input);
            // TODO: maybe move this error into setRegMem after supporting types with bit width <= 8
            if (func.getSize(input_ty).? != 1)
                return try func.fail("unable to load non-8-bit-sized into {c} register", .{std.ascii.toUpper(reg_name[0])});
            if (!(input_val == .reg and input_val.reg == reg)) {
                const reg_val = try func.freeReg(reg, inst);
                // DEPRECATED: try func.setRegMem(input_ty, reg, input_val);
                try func.trans(input_val, reg_val, input_ty);
            }
        }

        const asm_source = std.mem.sliceAsBytes(func.air.extra[extra_i..])[0..extra.data.source_len];
        log.debug("asm_source.len: {}", .{asm_source.len});
        if (asm_source.len != 0)
            log.debug("asm source:\n```\n{s}\n```", .{asm_source});
        if (std.mem.eql(u8, asm_source, "rts")) {
            try func.addInst(.rts_impl, .{ .none = {} });
        } else if (std.mem.eql(u8, asm_source, "nop")) {
            try func.addInst(.nop_impl, .{ .none = {} });
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

fn airBitCast(func: *Func, inst: Air.Inst.Index) !void {
    // This is only a change at the type system level.
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    _ = ty_op.ty;
    const res = try func.resolveInst(ty_op.operand);
    func.finishAir(inst, res, &.{ty_op.operand});
}

fn airBreakpoint(func: *Func, inst: Air.Inst.Index) !void {
    // Examples of the behavior of BRK:
    // * On the C64 it clears the screen and resets.
    // * On the C128 it prints "BREAK" followed by the values of
    //   PC (Program Counter), SR (Status Register),
    //   AC (ACcumulator), XR (X Register), YR (Y Register),
    //   and SP (Stack Pointer)
    try func.addInst(.brk_impl, .{ .none = {} });
    // TODO: some OSs (like the SOS written for the Apple III) apparently use BRK for system calls and the byte following BRK is the syscall number.
    //       investigate that, interrupts, and the necessity of this NOP more. for now we include it just to be sure because BRK advances the PC by 2.
    //       https://retrocomputing.stackexchange.com/questions/12291/what-are-uses-of-the-byte-after-brk-instruction-on-6502
    try func.addInst(.nop_impl, .{ .none = {} });
    func.finishAir(inst, .none, &.{});
}

fn airCall(func: *Func, inst: Air.Inst.Index, modifier: std.builtin.CallModifier) !void {
    switch (modifier) {
        .auto, .never_inline => {},
        .compile_time => unreachable,
        .async_kw,
        .never_tail,
        .no_async,
        .always_tail,
        .always_inline,
        => return try func.fail("unsupported call modifier {}", .{modifier}),
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

    var call_vals = try func.resolveCallingConventionValues(fn_ty);
    defer call_vals.deinit(func.getAllocator());

    for (call_vals.args) |dst, i| {
        // `dst` is where we want the argument according to the calling convention (the "specification") and
        // `src` is where the argument currently is.
        const src = try func.resolveInst(args[i]);
        const arg_ty = func.air.typeOf(args[i]);
        // DEPRECATED: try func.setAddrOrRegMem(arg_ty, dst, src);
        try func.trans(src, dst, arg_ty);
    }

    if (func.air.value(callee)) |fn_val| {
        if (fn_val.castTag(.function)) |fn_pl| {
            log.debug("Calling {s}...", .{func.getMod().declPtr(fn_pl.data.owner_decl).name});
            if (func.bin_file.cast(link.File.Prg)) |prg| {
                const blk_i = try prg.recordDecl(fn_pl.data.owner_decl);
                try func.addInst(.jsr_abs, .{ .abs = .{ .unres = .{ .block_index = blk_i } } });
            } else unreachable;
        } else if (fn_val.castTag(.extern_fn)) |_| {
            return try func.fail("extern functions not supported", .{});
        } else if (fn_val.castTag(.decl_ref)) |_| {
            return try func.fail("TODO implement calling bitcasted functions", .{});
        } else if (fn_val.castTag(.int_u64)) |int| {
            try func.addInst(.jsr_abs, .{ .abs = .{ .fixed = @intCast(u16, int.data) } });
        } else unreachable;
    } else {
        assert(callee_ty.zigTypeTag() == .Pointer);
        panic("TODO: handle {}", .{func.air.value(callee).?.tag()});
    }

    const res = call_vals.ret_val;
    var big_tomb = try func.iterateBigTomb(inst, args.len + 1);
    big_tomb.feed(pl_op.operand, res);
    for (args) |arg| big_tomb.feed(arg, res);
    return big_tomb.finishAir(res);
}

fn airDbgStmt(func: *Func, inst: Air.Inst.Index) !void {
    // TODO: emit debug info
    func.finishAir(inst, .none, &.{});
}

fn airDbgBlock(func: *Func, inst: Air.Inst.Index) !void {
    // TODO: emit debug info
    func.finishAir(inst, .none, &.{});
}

fn airDbgInline(func: *Func, inst: Air.Inst.Index) !void {
    // TODO: emit debug info
    func.finishAir(inst, .none, &.{});
}

fn airDbgLocal(func: *Func, inst: Air.Inst.Index) !void {
    const pl_op = func.air.instructions.items(.data)[inst].pl_op;
    // TODO: emit debug info
    func.finishAir(inst, .none, &.{pl_op.operand});
}

fn airLoad(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const elem_ty = func.air.getRefType(ty_op.ty);
    const res: MV = res: {
        if (!elem_ty.hasRuntimeBitsIgnoreComptime())
            break :res .none;

        const ptr = try func.resolveInst(ty_op.operand);
        const ptr_ty = func.air.typeOf(ty_op.operand);
        const is_volatile = ptr_ty.isVolatilePtr();
        if (!is_volatile and func.liveness.isUnused(inst))
            break :res .none;

        const dst = dst: {
            if (func.reuseOperand(inst, ty_op.operand, 0, ptr)) {
                // The MV that holds the pointer can be reused as the value.
                break :dst ptr;
            } else {
                break :dst try func.allocAddrOrRegMem(elem_ty, inst);
            }
        };
        // DEPRECATED: try func.loadFromPtr(dst, ptr, ptr_ty);
        try func.trans(ptr, dst, elem_ty);
        break :res dst;
    };
    func.finishAir(inst, res, &.{ty_op.operand});
}

fn airRet(func: *Func, inst: Air.Inst.Index) !void {
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const val = try func.resolveInst(un_op);
    const ret_ty = func.getType().fnReturnType();
    // DEPRECATED: try func.setAddrOrRegMem(ret_ty, func.ret_val, val);
    try func.trans(val, func.ret_val, ret_ty);
    try func.addInst(.rts_impl, .{ .none = {} });
    func.finishAir(inst, .none, &.{un_op});
}

fn airStore(func: *Func, inst: Air.Inst.Index) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;
    const ptr = try func.resolveInst(bin_op.lhs);
    const ptr_ty = func.air.typeOf(bin_op.lhs);
    const val = try func.resolveInst(bin_op.rhs);
    const val_ty = func.air.typeOf(bin_op.rhs);
    assert(ptr_ty.childType().eql(val_ty, func.getMod()));
    // DEPRECATED: try func.storeToPtr(ptr, val, ptr_ty, val_ty);
    try func.trans(val, ptr, val_ty);
    func.finishAir(inst, .none, &.{ bin_op.lhs, bin_op.rhs });
}

fn airUnreach(func: *Func, inst: Air.Inst.Index) !void {
    func.finishAir(inst, .none, &.{});
}

fn airIntCast(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    if (func.liveness.isUnused(inst))
        return func.finishAir(inst, .none, &.{ty_op.operand});

    const dst_ty = func.air.getRefType(ty_op.ty);

    const op_ty = func.air.typeOf(ty_op.operand);
    const op = try func.resolveInst(ty_op.operand);
    const op_ty_info = op_ty.intInfo(func.getTarget());
    const dst_ty_info = dst_ty.intInfo(func.getTarget());

    const op_size = op_ty.abiSize(func.getTarget());
    const dst_size = dst_ty.abiSize(func.getTarget());
    const dst: MV = dst: {
        if (op_ty_info.signedness != dst_ty_info.signedness)
            return try func.fail("TODO: handle @intCast with different signedness", .{});

        if (op_ty_info.bits == dst_ty_info.bits) {
            break :dst op;
        }
        assert(op_ty_info.bits % 8 == 0 and dst_ty_info.bits % 8 == 0); // TODO: truncate bits

        if (try func.allocRegMem(dst_ty, inst)) |reg| {
            // DEPRECATED: try func.setRegMem(dst_ty, reg.reg, op);
            try func.trans(op, reg, Type.u8);
            break :dst reg;
        }

        assert(dst_size >= op_size); // TODO: truncate bytes

        const dst = try func.allocAddrMem(dst_ty);
        // DEPRECATED: try func.setAddrMem(dst_ty, dst, op);
        try func.trans(op, dst, op_ty);
        break :dst dst;
    };

    func.finishAir(inst, dst, &.{ty_op.operand});
}
