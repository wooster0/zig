//! Non-addressable register memory.

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
const Reg = bits.Reg;
const abi = @import("abi.zig");
const Mir = @import("Mir.zig");
const Emit = @import("Emit.zig");
const Func = @import("Func.zig");

const RegMem = @This();

// Main registers

/// The AIR instruction that uses the accumulator register, if any.
reg_a_owner: ?Air.Inst.Index = null,
/// The AIR instruction that uses the X index register, if any.
reg_x_owner: ?Air.Inst.Index = null,
/// The AIR instruction that uses the Y index register, if any.
reg_y_owner: ?Air.Inst.Index = null,

// Status register

/// Whether binary-coded decimal mode is on.
decimal: Flag(.sed_impl, .cld_impl, "decimal") = .{},
/// Whether we have a carry bit.
carry: Flag(.sec_impl, .clc_impl, "carry") = .{},

fn Flag(comptime set_inst: Mir.Inst.Tag, comptime clear_inst: Mir.Inst.Tag, comptime field_name: []const u8) type {
    return struct {
        const Self = @This();

        state: State = .unknown,

        const State = enum { set, clear, unknown };

        pub fn set(flag: *Self) !void {
            switch (flag.state) {
                .set => {},
                .clear, .unknown => {
                    const reg_mem = @fieldParentPtr(RegMem, field_name, flag);
                    const func = @fieldParentPtr(Func, "reg_mem", reg_mem);
                    try func.addInst(set_inst, .{ .none = {} });
                    flag.state = .set;
                },
            }
        }
        pub fn clear(flag: *Self) !void {
            switch (flag.state) {
                .clear => {},
                .set, .unknown => {
                    const reg_mem = @fieldParentPtr(RegMem, field_name, flag);
                    const func = @fieldParentPtr(Func, "reg_mem", reg_mem);
                    try func.addInst(clear_inst, .{ .none = {} });
                    flag.state = .clear;
                },
            }
        }
    };
}

/// Spills the register if not free and associates it with the new instruction.
// TODO: merge with spillReg?
pub fn freeReg(
    reg_mem: *RegMem,
    reg: Reg,
    /// If null is given, it means the register is temporarily needed and
    /// will not have an owner thereafter.
    maybe_owner: ?Air.Inst.Index,
) !void {
    const func = @fieldParentPtr(Func, "reg_mem", reg_mem);

    log.debug("freeReg: making %{?} the owner of {}", .{ maybe_owner, reg });

    switch (reg) {
        .a => {
            if (reg_mem.reg_a_owner) |old_owner| {
                if (maybe_owner) |owner| {
                    if (owner == old_owner)
                        return;
                }
                // TODO: use PHA and PLA to spill using the hardware stack?
                //       for that we very likely have to introduce a new MemoryValue that says
                //       the value has to be PLA'd first (MemoryValue.stack).
                //       also, can we TXA and TYA below and do it there too? measure total cycles of each solution
                try func.spillReg(reg, old_owner);
            }
            reg_mem.reg_a_owner = maybe_owner;
        },
        .x => {
            if (reg_mem.reg_x_owner) |old_owner| {
                if (maybe_owner) |owner| {
                    if (owner == old_owner)
                        return;
                }
                try func.spillReg(reg, old_owner);
            }
            reg_mem.reg_x_owner = maybe_owner;
        },
        .y => {
            if (reg_mem.reg_y_owner) |old_owner| {
                if (maybe_owner) |owner| {
                    if (owner == old_owner)
                        return;
                }
                try func.spillReg(reg, old_owner);
            }
            reg_mem.reg_y_owner = maybe_owner;
        },
    }
}
/// Assumes the given register is now owned by the new instruction and associates it with the new instruction.
pub fn takeReg(
    reg_mem: *RegMem,
    reg: Reg,
    /// If null is given, it means the register is temporarily needed and
    /// will not have an owner thereafter.
    maybe_owner: ?Air.Inst.Index,
) void {
    switch (reg) {
        .a => reg_mem.reg_a_owner = maybe_owner,
        .x => reg_mem.reg_x_owner = maybe_owner,
        .y => reg_mem.reg_y_owner = maybe_owner,
    }
}

/// Saves a register and makes it free for usage until it is restored.
pub fn saveReg(reg_mem: *RegMem, reg: Reg) !RegSave {
    const func = @fieldParentPtr(Func, "reg_mem", reg_mem);
    // TODO: simply spilling instead of using the hardware stack
    //       would sometimes save cycles here
    //       (at the cost of code size; use the optimize mode (getOptimizeMode)
    //       to determine what's better)
    var reg_owner: ?Air.Inst.Index = undefined;
    switch (reg) {
        .a => {
            // NOTE: for this we are looking at 3 cycles (PHA) + 4 cycles (PLA) = 7
            if (reg_mem.reg_a_owner != null) {
                try func.addInst(.pha_impl, .{ .none = {} });
                try func.mir_instructions.setCapacity(func.getAllocator(), func.mir_instructions.capacity + 1);
            }
            reg_owner = reg_mem.reg_a_owner;
            reg_mem.reg_a_owner = null;
        },
        .x => {
            if (reg_mem.reg_x_owner != null) {
                try func.addInst(.txa_impl, .{ .none = {} });
                try func.addInst(.pha_impl, .{ .none = {} });
                try func.mir_instructions.setCapacity(func.getAllocator(), func.mir_instructions.capacity + 2);
            }
            reg_owner = reg_mem.reg_x_owner;
            reg_mem.reg_x_owner = null;
        },
        .y => {
            if (reg_mem.reg_x_owner != null) {
                try func.addInst(.tya_impl, .{ .none = {} });
                try func.addInst(.pha_impl, .{ .none = {} });
                try func.mir_instructions.setCapacity(func.getAllocator(), func.mir_instructions.capacity + 2);
            }
            reg_owner = reg_mem.reg_y_owner;
            reg_mem.reg_y_owner = null;
        },
    }
    return RegSave{ .func = func, .reg = reg, .reg_owner = reg_owner };
}
pub const RegSave = struct {
    func: *Func,
    reg: Reg,
    reg_owner: ?Air.Inst.Index,

    pub fn restore(reg_save: RegSave) void {
        // No OOM is possible because we preallocated in saveReg.
        switch (reg_save.reg) {
            .a => {
                if (reg_save.reg_owner != null) {
                    reg_save.func.addInst(.pla_impl, .{ .none = {} }) catch unreachable;
                }
            },
            .x => {
                if (reg_save.reg_owner != null) {
                    reg_save.func.addInst(.pla_impl, .{ .none = {} }) catch unreachable;
                    reg_save.func.addInst(.tax_impl, .{ .none = {} }) catch unreachable;
                }
            },
            .y => {
                if (reg_save.reg_owner != null) {
                    reg_save.func.addInst(.pla_impl, .{ .none = {} }) catch unreachable;
                    reg_save.func.addInst(.tay_impl, .{ .none = {} }) catch unreachable;
                }
            },
        }
    }
};

/// Returns a register free for allocation.
/// Returns null if all registers are occupied.
pub fn alloc(reg_mem: *RegMem, owner: Air.Inst.Index) ?Reg {
    if (reg_mem.reg_a_owner == null) {
        reg_mem.reg_a_owner = owner;
        return .a;
    }
    if (reg_mem.reg_x_owner == null) {
        reg_mem.reg_x_owner = owner;
        return .x;
    }
    if (reg_mem.reg_y_owner == null) {
        reg_mem.reg_y_owner = owner;
        return .y;
    }
    return null;
}

pub fn checkInst(reg_mem: *RegMem, inst: Mir.Inst) void {
    const func = @fieldParentPtr(Func, "reg_mem", reg_mem);
    if (inst.tag.getAffectedReg()) |affected_reg| {
        switch (affected_reg) {
            .a => if (reg_mem.reg_a_owner) |reg_a_owner| assert(reg_a_owner == func.air_current_inst), // Failure: register clobbered.
            .x => if (reg_mem.reg_x_owner) |reg_x_owner| assert(reg_x_owner == func.air_current_inst), // Failure: register clobbered.
            .y => if (reg_mem.reg_y_owner) |reg_y_owner| assert(reg_y_owner == func.air_current_inst), // Failure: register clobbered.
        }
    }
}
