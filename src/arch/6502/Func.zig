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
const log = std.log.scoped(.codegen);
const link = @import("../../link.zig");
const Module = @import("../../Module.zig");
const Type = @import("../../type.zig").Type;
const Value = @import("../../value.zig").Value;
const Air = @import("../../Air.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");
const Liveness = @import("../../Liveness.zig");
const codegen = @import("../../codegen.zig");
const GenerateSymbolError = codegen.GenerateSymbolError;
const FnResult = codegen.FnResult;
const DebugInfoOutput = codegen.DebugInfoOutput;
const Decl = Module.Decl;

const bits = @import("bits.zig");
const abi = @import("abi.zig");
const Register = bits.Register;
const RegisterManager = abi.RegisterManager;
const RegisterLock = RegisterManager.RegisterLock;
const regs = abi.regs;
const gp = abi.RegisterClass.gp;

const Func = @This();

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

args: []MValue = &.{},
/// The index of the current argument that is being resolved in `airArg`.
arg_index: u16 = 0,
/// The result location of the return value of the current function.
ret_mv: MValue = undefined,

/// An error message for if codegen fails.
/// This is set if error.CodegenFail happens.
err_msg: *Module.ErrorMsg = undefined,

/// Contains a list of current branches.
/// When we return from a branch, the branch will be popped from this list,
/// which means branches can only contain references from within its own branch,
/// or a branch higher (lower index) in the tree.
branches: std.ArrayListUnmanaged(Branch) = .{},

/// The output of this codegen.
mir_instructions: std.MultiArrayList(Mir.Inst) = .{},
//mir_extra: std.ArrayListUnmanaged(u32) = .{},

/// After we finished lowering something we need to call `airFinish`.
/// This is equal to the amount of times we called `airFinish`.
/// We can use this to check if we forgot to call `airFinish`, which helps finding bugs.
air_bookkeeping: if (debug.runtime_safety) usize else void = if (debug.runtime_safety) 0 else {},

// https://github.com/cc65/cc65/blob/master/asminc/c64.inc
// https://github.com/cc65/cc65/blob/master/asminc/cbm_kernal.inc

register_manager: RegisterManager = .{},

// TODO: the original Furby used a 6502 chip with 128 bytes of RAM and no Y register.
//       allow compatibility with such variants of the 6502 using a constraint config like this
//       based on which we generate code
//       This also applies to the NES. to be compatible we basically just don't have to use decimal mode (it's the 2A03)
//       https://en.wikipedia.org/wiki/MOS_Technology_6502#Variations_and_derivatives
//constraints: struct {},

memory: Memory,

// TODO: let the user provide information about the register and flags' initial state so we don't have to assume the initial state is unknown?
//       what if we're embedded into something else? that way we can avoid setting these flags
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
        .args = undefined,
        .err_msg = undefined,
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

/// Memory Value (MV). This represents the location of a value in memory.
const MValue = union(enum) {
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
    fn index(mv: MValue, offset: u16) MValue {
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
            .zp => |addr| return .{ .zp = addr + @intCast(u8, offset) },
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

fn Flag(comptime set_inst: Mir.Inst.Tag, comptime clear_inst: Mir.Inst.Tag, comptime field_name: []const u8) type {
    return struct {
        const Self = @This();

        const State = enum { set, clear, unknown };

        state: State = .unknown,

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

const Memory = struct {
    /// Memory addresses in the first page of memory available for storage.
    /// This is expensive, sparsely available memory and very valuable because
    /// writing and reading from it is faster than for absolute memory.
    /// This must always contain at least one address for general-purpose.
    zp_free: std.BoundedArray(u8, 256) = .{},
    /// The offset at which free absolute memory starts.
    /// Must not start in zero page. Must be contiguously free memory.
    /// This is like a stack because it grows downwards.
    ///
    /// You can call this the software stack.
    abs_offset: u16,
    // TODO(solution): record all alloc() calls and where they came from (src_loc) and then if the last record's abs_offset is <= program_size,
    //                 report an error for the specific alloc that crashed the stack into the program
    ///// The absolute memory address at which memory ends.
    //abs_end: u16,

    fn init(target: std.Target, bin_file: *link.File) Memory {
        if (bin_file.cast(link.File.Prg)) |prg| {
            if (prg.zp_free != null and prg.abs_offset != null) {
                const zp_free = prg.zp_free.?;
                const abs_offset = prg.abs_offset.?;
                return Memory{
                    .zp_free = zp_free,
                    .abs_offset = abs_offset,
                };
            }
        } else unreachable;

        const zp_free = abi.getZeroPageAddresses(target);

        if (debug.runtime_safety and !std.sort.isSorted(u8, zp_free.constSlice(), {}, std.sort.asc(u8)))
            @panic("zero page addresses must be sorted low to high");

        const abs_offset = abi.getAbsoluteMemoryOffset(target);

        //const gp_zp = zp_free.pop();

        return Memory{
            .zp_free = zp_free,
            .abs_offset = abs_offset,
        };
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
    /// Returns either MValue.zp or MValue.abs.
    fn alloc(memory: *Memory, byte_count: u16) !MValue {
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
            // but we must still not allow the zero page to deplete
            (byte_count == 1 and memory.zp_free.len > 1);

        if (use_zero_page) {
            // find a matching number of free bytes, each with an address only 1 apart from the one before it (contiguous)
            const addrs = memory.zp_free.constSlice();
            log.debug("addrs: {any}", .{addrs});
            var maybe_contig_addr_i: ?u8 = null;
            for (addrs) |curr_addr, i| {
                if (i + 1 == addrs.len)
                    break;
                const next_addr = addrs[i + 1];
                const diff = next_addr - curr_addr;
                assert(diff != 0); // otherwise we have a duplicate zero page address
                const is_beside = diff == 1;
                if (is_beside) {
                    if (maybe_contig_addr_i == null)
                        maybe_contig_addr_i = @intCast(u8, i);
                } else {
                    // no longer contiguous; try again
                    maybe_contig_addr_i = null;
                }
            }
            if (maybe_contig_addr_i) |contig_addr_i| {
                if (contig_addr_i + byte_count >= memory.zp_free.len) {
                    log.debug("removing {} bytes from the ZP", .{byte_count});
                    const start = memory.zp_free.swapRemove(contig_addr_i);
                    var i: u16 = 1;
                    while (i < byte_count) : (i += 1)
                        _ = memory.zp_free.swapRemove(contig_addr_i);
                    std.sort.sort(u8, memory.zp_free.slice(), {}, std.sort.asc(u8));
                    return MValue{ .zp = start };
                }
            }
        }

        //if (addr == abs.abs_end) {
        //    const func = @fieldParentPtr(Func, "memory", memory);
        //    return func.fail("program depleted all registers and all 64 KiB of memory", .{}),
        //}
        memory.abs_offset -= byte_count; // grow the stack downwards
        return MValue{ .abs = memory.abs_offset };
    }

    fn free(memory: *Memory, value: MValue, ty: Type) void {
        const func = @fieldParentPtr(Func, "memory", memory);
        const byte_size = func.getByteSize(ty).?;
        log.debug("I want to free {} bytes", .{byte_size});
        switch (value) {
            .zp => |addr| {
                var i: u8 = 0;
                while (i < byte_size) : (i += 1)
                    memory.zp_free.appendAssumeCapacity(addr + i);
            },
            .abs => |addr| {
                // TODO: free memory!
                //       do it this way: at the start of a branch record the offset and when that branch dies, reset
                _ = addr;
            },
            else => unreachable,
        }
    }
};

const Branch = struct {
    values: std.AutoArrayHashMapUnmanaged(Air.Inst.Ref, MValue) = .{},

    fn deinit(branch: *Branch, allocator: mem.Allocator) void {
        branch.values.deinit(allocator);
    }
};

fn currentBranch(func: *Func) *Branch {
    return &func.branches.items[func.branches.items.len - 1];
}


/// Returns the byte size of a type or null if the byte size is too big.
/// Assert non-null in contexts where a type is involved where the compiler made sure the bit width is <= 65535.
fn getByteSize(func: Func, ty: Type) ?u16 {
    return math.cast(u16, ty.abiSize(func.getTarget()));
}

// /// Allocates stack-local memory and returns it.
// fn alloc(func: *Func, size: u16) MValue {
//     _ = size;
//     _ = func;
// }

// /// Loads a value at the given stack offset.
// fn load(func: *Func, offset: u16) MValue {
//     _ = offset;
//     _ = func;
// }

// /// Stores a value at the given stack offset.
// fn store(func: *Func, offset: u16, mv: MValue) void {
//     _ = mv;
//     _ = offset;
//     _ = func;
// }

const CallMValues = struct {
    args: []MValue,
    return_value: MValue,

    fn deinit(values: *CallMValues, allocator: mem.Allocator) void {
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
fn resolveCallingConventionValues(func: *Func, fn_ty: Type) !CallMValues {
    // ref: https://llvm-mos.org/wiki/C_calling_convention
    const cc = fn_ty.fnCallingConvention();
    const param_types = try func.gpa.alloc(Type, fn_ty.fnParamLen());
    defer func.gpa.free(param_types);
    fn_ty.fnParamTypes(param_types);
    var values = CallMValues{
        .args = try func.gpa.alloc(MValue, param_types.len),
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
    func.ret_mv = call_values.return_value;

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
        .add => func.airBinOp(inst, .add),
        .add_optimized => func.fail("TODO: handle {}", .{tag}),
        .addwrap => func.airBinOp(inst, .addwrap),
        .addwrap_optimized,
        .add_sat,
        => func.fail("TODO: handle {}", .{tag}),
        .sub => func.airBinOp(inst, .sub),
        .sub_optimized => func.fail("TODO: handle {}", .{tag}),
        .subwrap => func.airBinOp(inst, .subwrap),
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
        .ptr_add => func.airBinOp(inst, .ptr_add),
        .ptr_sub => func.airBinOp(inst, .ptr_sub),
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
        .clz,
        .ctz,
        .popcount,
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
        .array_elem_val,
        .slice_elem_val,
        .slice_elem_ptr,
        .ptr_elem_val,
        .ptr_elem_ptr,
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
        .reg => |reg| {
            func.register_manager.getRegAssumeFree(reg, inst);
        },
        else => {},
    }

    log.debug("lowering arg in {} of type {} with src index {}", .{ mv, arg_ty.tag(), arg_src_index });

    func.finishAir(inst, mv, &.{});
}

/// Emits code to perform the given binary operation on two operands of the same type resulting in the same type.
/// Commutativeness may be taken into account.
fn airBinOp(func: *Func, inst: Air.Inst.Index, tag: Air.Inst.Tag) !void {
    const bin_op = func.air.instructions.items(.data)[inst].bin_op;
    log.debug("bin_op: {}", .{bin_op});

    const lhs = try func.resolveInst(bin_op.lhs);
    const rhs = try func.resolveInst(bin_op.rhs);
    const lhs_ty = func.air.typeOf(bin_op.lhs);
    const rhs_ty = func.air.typeOf(bin_op.rhs);

    const result = try func.binOp(tag, inst, lhs, rhs, lhs_ty, rhs_ty);
    func.finishAir(inst, result, &.{ bin_op.lhs, bin_op.rhs });
}

/// A binary operation is a rule for combining two operands to produce another value.
fn binOp(
    func: *Func,
    tag: Air.Inst.Tag,
    maybe_inst: ?Air.Inst.Index,
    lhs: MValue,
    rhs: MValue,
    lhs_ty: Type,
    rhs_ty: Type,
) !MValue {
    _ = maybe_inst;
    switch (tag) {
        .add,
        .sub,
        => {
            switch (lhs_ty.zigTypeTag()) {
                .Int => {
                    assert(lhs_ty.eql(rhs_ty, func.getMod()));
                    const ty = lhs_ty;
                    return try func.binOpInt(lhs, rhs, ty, tag);
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
                    const op_tag: Air.Inst.Tag = switch (tag) {
                        .ptr_add => .add,
                        .ptr_sub => .sub,
                        else => unreachable,
                    };
                    assert(lhs_ty.eql(rhs_ty, func.getMod()));
                    const ty = lhs_ty;
                    return try func.binOpInt(lhs, rhs, ty, op_tag);
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

/// Emits code to perform the given binary operation on two operands of the same size resulting in the same size.
/// Supports the native 8-bit integer size as well as any other size.
/// Commutativeness may be taken into account.
fn binOpInt(func: *Func, lhs: MValue, rhs: MValue, ty: Type, bin_op: Air.Inst.Tag) !MValue {
    const byte_size = func.getByteSize(ty).?;

    assert(ty.zigTypeTag() == .Int);

    switch (bin_op) {
        // this assembly showcases the addition of two words:
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

            const res = try func.allocMem(ty);

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            try func.register_manager.getReg(.a, null);
            defer func.register_manager.freeReg(.a);
            const reg_a = MValue{ .reg = .a };
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // take advantage of the fact that addition is commutative,
                // meaning we can add the operands in any order;
                // get either of the operands into the accumulator
                var values = std.BoundedArray(MValue, 2){};
                values.appendAssumeCapacity(lhs.index(i));
                values.appendAssumeCapacity(rhs.index(i));
                for (values.constSlice()) |value, index| {
                    switch (value) {
                        .reg => |reg| {
                            if (reg == .a) {
                                _ = values.swapRemove(index);
                                break;
                            }
                        },
                        else => {},
                    }
                } else {
                    try func.store(reg_a, values.pop(), Type.Tag.init(.u8), Type.Tag.init(.u8));
                }
                try func.addInstAny("adc", values.pop(), i);
                assert(values.len == 0);
                try func.store(res.index(i), reg_a, Type.Tag.init(.u8), Type.Tag.init(.u8));
            }
            return res;
        },
        .sub, .subwrap => {
            try func.decimal.clear();
            try func.carry.clear();
            defer func.carry.state = .unknown;

            const res = try func.allocMem(ty);

            log.debug("lhs: {}, rhs: {}", .{ lhs, rhs });
            try func.register_manager.getReg(.a, null);
            defer func.register_manager.freeReg(.a);
            const reg_a = .{ .reg = .a };
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                // get LHS into the accumulator and subtract RHS from it
                try func.store(reg_a, lhs, ty, ty);
                try func.addInstAny("sbc", rhs, i);
                try func.store(res.index(i), reg_a, ty, ty);
            }

            return res;
        },
        else => unreachable,
    }
}

/// Allocates memory local to the stack (does not have to be cleaned up) however possible.
fn airAlloc(func: *Func, inst: Air.Inst.Index) !void {
    const mv = try func.allocMemPtr(inst);
    log.debug("allocated ptr: {}", .{mv});
    func.finishAir(inst, mv, &.{});
}

fn airRetPtr(func: *Func, inst: Air.Inst.Index) !void {
    const mv = try func.allocMemPtr(inst);
    log.debug("(ret_ptr) allocated ptr: {}", .{mv});
    func.finishAir(inst, mv, &.{});
}

/// Uses a pointer instruction as the basis for allocating runtime-mutable memory.
fn allocMemPtr(func: *Func, inst: Air.Inst.Index) !MValue {
    const ptr_ty: Type = func.air.instructions.items(.data)[inst].ty;
    const child_ty = ptr_ty.childType();
    const byte_size = func.getByteSize(child_ty) orelse {
        return func.fail("type `{}` too big to fit in stack frame", .{child_ty.fmt(func.getMod())});
    };
    const abs = try func.memory.alloc(byte_size);
    return abs;
}

fn allocMem(func: *Func, ty: Type) !MValue {
    const byte_size = func.getByteSize(ty).?;
    if (byte_size == 1) {
        if (func.register_manager.tryAllocReg(null, gp)) |reg|
            return .{ .reg = reg };
    }
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

    try func.store(ptr, val, ptr_ty, val_ty);

    func.finishAir(inst, .none, &.{ bin_op.lhs, bin_op.rhs });
}

fn store(func: *Func, ptr: MValue, val: MValue, ptr_ty: Type, val_ty: Type) !void {
    const src = val;
    const src_size = func.getByteSize(val_ty).?;
    const dst = ptr;
    //const dst_size = func.getByteSize(if (ptr.isMem()) ptr_ty.childType() else ptr_ty).?;
    const child_ptr_ty = switch (ptr_ty.zigTypeTag()) {
        .Pointer => ptr_ty.childType(),
        else => ptr_ty,
    };
    const dst_size = func.getByteSize(child_ptr_ty).?;
    log.debug("I want to write {} bytes to a ptr that can hold at most {} bytes", .{ src_size, dst_size });

    try func.copy(src, src_size, dst, dst_size);
}

/// Copies LHS to RHS, preserving all registers not specified in the operation.
// TODO: register_manager.getReg
fn copy(func: *Func, src: MValue, src_size: u16, dst: MValue, dst_size: u16) !void {
    if (src_size == 0)
        return;
    assert(src_size <= dst_size);
    log.debug("copying {} bytes of {} to {} which is capable of holding {} bytes", .{ src_size, src, dst, dst_size });
    switch (src) {
        .none => unreachable,
        .imm => |imm| {
            assert(src_size == 1);
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |reg| {
                    assert(dst_size == 1);
                    switch (reg) {
                        .a => try func.addInstImm(.lda_imm, imm),
                        .x => try func.addInstImm(.ldx_imm, imm),
                        .y => try func.addInstImm(.ldy_imm, imm),
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    assert(src_size == 1);
                    try func.addInstImpl(.pha_impl);
                    try func.addInstImm(.lda_imm, imm);
                    try func.addInstMem("sta", dst, 0);
                    try func.addInstImpl(.pla_impl);
                },
            }
        },
        .reg => |src_reg| {
            // there are no "size equals 1" asserts here because registers can contain values of types such as u128
            // as long as the value itself fits in eight bits (e.g. @as(u128, 32))
            switch (src_reg) {
                .a => {
                    switch (dst) {
                        .none => unreachable,
                        .imm => unreachable,
                        .reg => |dst_reg| {
                            switch (dst_reg) {
                                .a => {},
                                .x => try func.addInstImpl(.tax_impl),
                                .y => try func.addInstImpl(.tay_impl),
                            }
                        },
                        .zp, .abs, .abs_unresolved => try func.addInstMem("sta", dst, 0),
                    }
                },
                .x => {
                    switch (dst) {
                        .none => unreachable,
                        .imm => unreachable,
                        .reg => |dst_reg| {
                            switch (dst_reg) {
                                .a => try func.addInstImpl(.txa_impl),
                                .x => {},
                                .y => {
                                    try func.addInstImpl(.pha_impl);
                                    try func.addInstImpl(.txa_impl);
                                    try func.addInstImpl(.tay_impl);
                                    try func.addInstImpl(.pla_impl);
                                },
                            }
                        },
                        .zp, .abs, .abs_unresolved => try func.addInstMem("stx", dst, 0),
                    }
                },
                .y => {
                    switch (dst) {
                        .none => unreachable,
                        .imm => unreachable,
                        .reg => |dst_reg| {
                            switch (dst_reg) {
                                .a => try func.addInstImpl(.tya_impl),
                                .x => {
                                    try func.addInstImpl(.pha_impl);
                                    try func.addInstImpl(.tya_impl);
                                    try func.addInstImpl(.tax_impl);
                                    try func.addInstImpl(.pla_impl);
                                },
                                .y => {},
                            }
                        },
                        .zp, .abs, .abs_unresolved => try func.addInstMem("sty", dst, 0),
                    }
                },
            }
        },
        .zp, .abs, .abs_unresolved => {
            switch (dst) {
                .none => unreachable,
                .imm => unreachable,
                .reg => |reg| {
                    switch (reg) {
                        .a => {
                            assert(src_size == 1);
                            assert(dst_size == 1);
                            try func.addInstMem("lda", src, 0);
                        },
                        .x => {
                            assert(src_size == 1);
                            assert(dst_size == 1);
                            try func.addInstMem("ldx", src, 0);
                        },
                        .y => {
                            assert(src_size == 1);
                            assert(dst_size == 1);
                            try func.addInstMem("ldy", src, 0);
                        },
                    }
                },
                .zp, .abs, .abs_unresolved => {
                    try func.addInstImpl(.pha_impl);
                    // TODO: if ReleaseSmall, do this loop at runtime at a certain threshold
                    var i: @TypeOf(src_size) = 0;
                    while (i < src_size) : (i += 1) {
                        try func.addInstMem("lda", src, @intCast(u16, i));
                        try func.addInstMem("sta", dst, @intCast(u16, i));
                    }
                    try func.addInstImpl(.pla_impl);
                },
            }
        },
    }
}

/// Emits code to read a value from a pointer.
fn airLoad(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const val_ty = func.air.getRefType(ty_op.ty);
    _ = val_ty;
    const ptr = try func.resolveInst(ty_op.operand);
    const ptr_ty = func.air.typeOf(ty_op.operand);

    const dst = try func.allocMem(ptr_ty);

    try func.load(dst, ptr, ptr_ty);
    func.finishAir(inst, dst, &.{ty_op.operand});
}

fn load(func: *Func, dst: MValue, ptr: MValue, ptr_ty: Type) !void {
    const child_ty = ptr_ty.childType();
    log.debug("loading value of type {} from {} into {}...", .{ child_ty.tag(), ptr, dst });
    const byte_size = func.getByteSize(child_ty).?;
    switch (ptr) {
        .zp, .abs, .abs_unresolved => {
            var i: u16 = 0;
            while (i < byte_size) : (i += 1) {
                try func.addInstMem("lda", ptr, i);
                try func.addInstMem("sta", dst, i);
            }
        },
        else => panic("TODO: handle {}", .{ptr}),
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
    const res: MValue = res: {
        if (operand_info.bits == dst_info.bits) {
            break :res operand;
        }

        const res = try func.allocMem(dst_ty);
        try func.store(
            res,
            operand,
            dst_ty,
            // note: not operand_ty.
            // this AIR instruction guarantees that the same integer value fits in both types.
            dst_ty,
        );
        break :res res;
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

fn structFieldPtr(func: *Func, inst: Air.Inst.Index, operand: Air.Inst.Ref, index: u32) !MValue {
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

fn airBitCast(func: *Func, inst: Air.Inst.Index) !void {
    const ty_op = func.air.instructions.items(.data)[inst].ty_op;
    const dst_ty = func.air.getRefType(ty_op.ty);
    _ = dst_ty;
    const res = try func.resolveInst(ty_op.operand);
    func.finishAir(inst, res, &.{ty_op.operand});
}

fn airPtrToInt(func: *Func, inst: Air.Inst.Index) !void {
    // nothing to do because pointers are represented as integers, anyway
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
    const val_ty = ret_ty;
    try func.store(func.ret_mv, val, ret_ty, val_ty);
    try func.addInstImpl(.rts_impl);
    func.finishAir(inst, .none, &.{un_op});
}

fn airRetLoad(func: *Func, inst: Air.Inst.Index) !void {
    const un_op = func.air.instructions.items(.data)[inst].un_op;
    const ptr = try func.resolveInst(un_op);
    const ptr_ty = func.air.typeOf(un_op);
    const ret_ty = func.getType().fnReturnType();
    const byte_size = func.getByteSize(ret_ty).?;
    const dst = try func.memory.alloc(byte_size);
    try func.load(dst, ptr, ptr_ty);
    const val = dst;
    try func.store(func.ret_mv, val, ptr_ty, ret_ty);
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

    try func.addInst(.jmp_abs, .{ .abs = .{ .current = .{
        .decl_index = func.getDeclIndex(),
        .offset = size,
    } } });
    func.finishAir(inst, .none, &.{});
}

fn airBreakpoint(func: *Func, inst: Air.Inst.Index) !void {
    try func.addInstImpl(.brk_impl);
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

    for (call_values.args) |expected, i| {
        const actual = try func.resolveInst(args[i]);
        const arg_ty = func.air.typeOf(args[i]);
        const byte_size = func.getByteSize(arg_ty).?;
        try func.copy(actual, byte_size, expected, byte_size);
    }

    if (func.air.value(callee)) |fn_val| {
        if (fn_val.castTag(.function)) |fn_pl| {
            log.debug("calling {s}...", .{func.getMod().declPtr(fn_pl.data.owner_decl).name});
            if (func.bin_file.cast(link.File.Prg)) |prg| {
                const blk_i = try prg.recordDecl(fn_pl.data.owner_decl);
                try func.addInst(.jsr_abs, .{ .abs = .{ .unresolved = .{ .blk_i = blk_i } } });
            } else unreachable;
        } else if (fn_val.castTag(.extern_fn)) |_| {
            return func.fail("extern functions are not supported", .{});
        } else if (fn_val.castTag(.decl_ref)) |_| {
            return func.fail("TODO implement calling bitcasted functions", .{});
        } else if (fn_val.castTag(.int_u64)) |int| {
            try func.addInst(.jsr_abs, .{ .abs = .{ .imm = @intCast(u16, int.data) } });
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

    // this assembly showcases conditional execution:
    // ```
    // lda #5
    // cmp #5 ; does A equal 5? (yes)
    //
    // main:
    //   beq
    // then:
    //   lda #0
    // else:
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
    // TODO: support outputting specific registers into variables (the other way around)
    assert(outputs.len == 0);
    extra_i += outputs.len;
    const inputs = @ptrCast([]const Air.Inst.Ref, func.air.extra[extra_i..][0..extra.data.inputs_len]);
    extra_i += inputs.len;

    assert(extra.data.clobbers_len() == 0);

    const dies = !extra.data.is_volatile() and func.liveness.isUnused(inst);
    const result: MValue = if (dies)
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
                return func.fail("unrecognized `asm` input constraint: \"{s}\"", .{constraint});
            }
            const reg_name = constraint[1 .. constraint.len - 1];
            const reg = parseRegName(reg_name) orelse {
                return func.fail("unrecognized register \"{s}\"", .{reg_name});
            };

            const input_mv = try func.resolveInst(input);
            try func.register_manager.getReg(reg, null);
            const ty = func.air.typeOf(input);
            // TODO: make this work with comptime_ints?
            if (func.getByteSize(ty).? != 1)
                return func.fail("unable to load non-8-bit-sized into {c} register", .{std.ascii.toUpper(reg_name[0])});
            try func.store(.{ .reg = reg }, input_mv, ty, ty);
        }

        const asm_source = mem.sliceAsBytes(func.air.extra[extra_i..])[0..extra.data.source_len];
        log.debug("asm_source.len: {}", .{asm_source.len});
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

fn lowerConstant(func: *Func, const_val: Value, ty: Type) !MValue {
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
        .Void => return MValue{ .none = {} },
        .Int => {
            const int_info = ty.intInfo(target);
            if (int_info.bits <= 8) {
                switch (int_info.signedness) {
                    .signed => return MValue{ .imm = @bitCast(u8, @intCast(i8, val.toSignedInt(target))) },
                    .unsigned => return MValue{ .imm = @intCast(u8, val.toUnsignedInt(target)) },
                }
            }
        },
        .Bool => return MValue{ .imm = @boolToInt(val.toBool()) },
        .Pointer => switch (ty.ptrSize()) {
            .Slice => {},
            else => switch (val.tag()) {
                .int_u64, .one, .zero, .null_value => {
                    return MValue{ .abs = @intCast(u16, val.toUnsignedInt(target)) };
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
        .Undefined => unreachable, // TODO: MValue.undef?
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

fn lowerUnnamedConst(func: *Func, val: Value, ty: Type) !MValue {
    log.debug("lowerUnnamedConst: ty = {}, val = {}", .{ val.fmtValue(ty, func.getMod()), ty.fmt(func.getMod()) });
    const blk_i = @intCast(
        u16,
        func.bin_file.lowerUnnamedConst(.{ .ty = ty, .val = val }, func.getDeclIndex()) catch |err| {
            return func.fail("lowering unnamed constant failed: {}", .{err});
        },
    );
    if (func.bin_file.cast(link.File.Prg)) |_| {
        return MValue{ .abs_unresolved = .{ .blk_i = blk_i } };
    } else unreachable;
}

fn lowerDeclRef(func: *Func, val: Value, ty: Type, decl_index: Decl.Index) !MValue {
    _ = ty;
    _ = val;

    const module = func.bin_file.options.module.?;
    const decl = module.declPtr(decl_index);
    module.markDeclAlive(decl);

    if (func.bin_file.cast(link.File.Prg)) |prg| {
        const blk_i = try prg.recordDecl(decl_index);
        return MValue{ .abs_unresolved = .{ .blk_i = blk_i } };
    } else unreachable;
}

fn lowerParentPtr(func: *Func, ptr_val: Value, ptr_child_ty: Type) !MValue {
    switch (ptr_val.tag()) {
        .elem_ptr => {
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

    fn finishAir(big_tomb: *BigTomb, result: MValue) void {
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
    log.warn("and this is not a constant! it is {!}!", .{value}); // TODO
    switch (value) {
        .zp, .abs => func.memory.free(value, func.air.typeOfIndex(inst)),
        .reg => |reg| {
            func.register_manager.freeReg(reg);
        },
        else => unreachable,
    }
}

fn finishAir(func: *Func, inst: Air.Inst.Index, result: MValue, operands: []const Air.Inst.Ref) void {
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

fn resolveInst(func: *Func, ref: Air.Inst.Ref) !MValue {
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
    //     return MValue{ .none = {} };

    const val = func.air.value(ref).?;
    log.debug("lowering constant of type {} with tag {}...", .{ ty.tag(), ty.zigTypeTag() });

    return func.lowerConstant(val, ty);
}

fn getResolvedInstValue(func: *Func, inst: Air.Inst.Ref) MValue {
    var i: usize = func.branches.items.len - 1;
    while (true) : (i -= 1) {
        if (func.branches.items[i].values.get(inst)) |mv| {
            assert(mv != .none);
            return mv;
        }
    }
}

/// Spills the given register to non-register memory.
pub fn spillInstruction(func: *Func, reg: Register, inst: Air.Inst.Index) !void {
    const reg_mv = func.getResolvedInstValue(Air.indexToRef(inst));
    assert(reg == reg_mv.reg);
    // TODO: use PHA and PLA here if reg == .a?
    //       how do we best associate the PLA with the inst and emit it when the inst dies?
    const branch = &func.branches.items[func.branches.items.len - 1];
    const mem_mv = try func.memory.alloc(1);
    try branch.values.put(func.gpa, Air.indexToRef(inst), mem_mv);
    try func.copy(reg_mv, 1, mem_mv, 1);
}
fn parseRegName(name: []const u8) ?Register {
    if (@hasDecl(Register, "parseRegName")) {
        return Register.parseRegName(name);
    }
    return std.meta.stringToEnum(Register, name);
}

/// Adds an MIR instruction to the output.
fn addInst(func: *Func, tag: Mir.Inst.Tag, data: anytype) error{OutOfMemory}!void {
    // TODO: shouldn't we be able to comptime-force this? `tag` can be `comptime`
    if (debug.runtime_safety)
        Mir.Inst.checkCombo(tag, data);
    try func.mir_instructions.append(func.gpa, .{ .tag = tag, .data = @as(Mir.Inst.Data, data) });
    log.debug("added instruction .{{ .tag = {}, .data = {} }}", .{ tag, data });
}
fn addInstImpl(func: *Func, tag: Mir.Inst.Tag) !void {
    try func.addInst(tag, .{ .none = {} });
}
fn addInstImm(func: *Func, tag: Mir.Inst.Tag, value: u8) !void {
    try func.addInst(tag, .{ .imm = value });
}
/// Adds an MIR instruction to the output that operates on memory.
///
/// Whenever you see an opcode mnemonic in double quotes, you can expect it to be used with multiple possible addressing modes.
fn addInstMem(func: *Func, comptime mnemonic: []const u8, mv: MValue, offset: u16) !void {
    switch (mv.index(offset)) {
        .zp => |addr| try func.addInst(
            std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_zp").?,
            .{ .zp = addr },
        ),
        .abs => |addr| try func.addInst(
            std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_abs").?,
            .{ .abs = .{ .imm = addr } },
        ),
        .abs_unresolved => |unresolved| {
            try func.addInst(
                std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_abs").?,
                .{ .abs = .{ .unresolved = .{ .blk_i = unresolved.blk_i, .offset = unresolved.offset } } },
            );
        },
        else => unreachable,
    }
}
/// Adds an MIR instruction to the output that operates using the appropriate addressing mode based on the given value.
/// The given offset is used only if an absolute memory or zero page memory addressing mode is used.
///
/// Whenever you see an opcode mnemonic in double quotes, you can expect it to be used with multiple possible addressing modes.
fn addInstAny(func: *Func, comptime mnemonic: []const u8, mv: MValue, offset: u16) !void {
    switch (mv.index(offset)) {
        .none => {
            try func.addInstImpl(std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_impl").?);
        },
        .imm => |imm| {
            try func.addInstImm(std.meta.stringToEnum(Mir.Inst.Tag, mnemonic ++ "_imm").?, imm);
        },
        .reg => {
            const temp = func.memory.zp_free.pop();
            defer func.memory.zp_free.appendAssumeCapacity(temp);
            const ptr = .{ .zp = temp };
            try func.copy(mv, 1, ptr, 1);
            try func.addInstAny(mnemonic, ptr, 0);
        },
        .zp, .abs, .abs_unresolved => try func.addInstMem(mnemonic, mv, offset),
    }
}

fn fail(func: *Func, comptime fmt: []const u8, args: anytype) error{ CodegenFail, OutOfMemory } {
    @setCold(true);
    func.err_msg = try Module.ErrorMsg.create(func.gpa, func.src_loc, fmt, args);
    return error.CodegenFail;
}
