//! Machine Intermediate Representation.
//! This step is useful to have because it allows us to lower our code into multiple formats,
//! such as actual 6502 machine code or its assembly code textual representation.

const std = @import("std");
const mem = std.mem;
const log = std.log;
const debug = std.debug;
const assert = debug.assert;
const testing = std.testing;
const bits = @import("bits.zig");
const Register = bits.Register;

const Mir = @This();

instructions: std.MultiArrayList(Inst).Slice,
//extra: []const u32,

/// An opcode and an operand with an addressing mode.
pub const Inst = struct {
    // Design decision note:
    // We could have not used an additional `Data` field and gone with a single `Tag` field:
    // ```
    // pub const Tag = union(enum(u8)) {
    //     lda_zp:  u8 =  0xA5, // 8 bits for zero page address
    //     lda_abs: u16 = 0xA5, // 16 bits for absolute address
    //     // ...
    // };
    // ```
    // This would be convenient but less efficient; it would take 4 bytes for each instruction:
    // * 1 byte for the opcode,
    // * the at most 2 bytes for an absolute address, and
    // * 1 byte for padding (to be a power of two).
    // So if we have for example three PHA (PusH A) instructions, it would take 4 * 3 = 12 bytes for the MIR.
    // With our current design it takes only 3 bytes for each instruction because PHA
    // has an implied operand and thus takes no additional data (`Data.none`).
    // This is all possible thanks to `std.MultiArrayList`.
    /// The instruction's opcode.
    tag: Tag,
    /// The instruction's operand with an addressing mode.
    data: Data,

    /// The position of an MIR instruction within the `Mir` instructions array.
    pub const Index = u32;

    /// The 6502 has 151 unique instructions. Organized into addressing modes, it has 56 instructions.
    ///
    /// We recognize the following addressing modes:
    /// * Implied (impl):
    ///   The operand, if any, is implicitly stated in the operation code of the instruction itself.
    /// * Immediate (imm):
    ///   Value is the one-byte operand.
    /// * Absolute (abs):
    ///   Value is read from absolute address as the two-byte operand.
    /// * X-indexed absolute (x_abs):
    ///   Value is read from absolute address formed by adding the X register to the two-byte operand.
    /// * Y-indexed absolute (y_abs):
    ///   Value is read from absolute address formed by adding the Y register to the two-byte operand.
    /// * Indirect absolute (ind_abs):
    ///   Value is read from absolute address read from the given address.
    /// * Zero page (zp):
    ///   Value is read from zero page address.
    /// * X-indexed zero page (x_zp):
    ///   Value is read from zero page address formed by adding X to the one-byte operand.
    /// * Y-indexed zero page (y_zp):
    ///   Value is read from zero page address formed by adding Y to the one-byte operand.
    /// * X-indexed indirect zero page (x_ind_zp):
    ///   The X register's value is added to the one-byte operand resulting in a zero page address
    ///   from which a two-byte address is read from which the value is read.
    ///   This allows dereferencing pointers.
    ///   The zero page boundary is never crossed.
    /// * Indirect Y-indexed zero page (ind_y_zp):
    ///   A 2-byte address is from the given zero page address, the Y register's value is added to the 2-byte address
    ///   and then the value is read from that address.
    ///   This allows dereferencing pointers.
    ///   The zero page boundary is never crossed.
    /// * Relative (rel):
    ///   The operand is a signed byte that is added to the program counter, which allows jumping relatively +-128 bytes from the instruction following this one.
    ///
    // Design decision note:
    // Because the layout of an instruction byte is supposedly in the form of
    // AAABBBCC, where AA and CC define the operation and BBB the addressing mode,
    // we could assemble the byte according to that format but the problem with that is that
    // we would deal with possible ambiguities and exceptions to that rule.
    // It seems simpler to define only the opcodes we actually need. This way it is trivial to extend, too.
    //
    // Design decision note:
    // What if we had separated mnemonic and addressing mode into two enums?
    // In that case `checkCombo` would need a gigantic `switch` checking each combo manually,
    // or we would simply not have that kind of safety.
    pub const Tag = enum(u8) {
        // each tag name must be the opcode's mnemonic plus the addressing mode as a suffix.
        // instructions are sorted by opcode.
        // zig fmt: off
        brk_impl     = 0x00, // BRK          ; BReaK
        clc_impl     = 0x18, // CLC          ; CLear Carry
        jsr_abs      = 0x20, // JSR $XXXX    ; Jump to SubRoutine
        sec_impl     = 0x38, // SEC          ; SEt Carry
        pha_impl     = 0x48, // PHA          ; PusH A
        jmp_abs      = 0x4C, // JMP $XXXX    ; JuMP
        rts_impl     = 0x60, // RTS          ; ReTurn from Subroutine
        adc_zp       = 0x65, // ADC $XX      ; ADd with Carry
        adc_imm      = 0x69, // ADC #$XX     ; ADd with Carry
        pla_impl     = 0x68, // PLA          ; PuLl A
        adc_abs      = 0x6D, // ADC $XXXX    ; ADd with Carry
        adc_ind_y_zp = 0x71, // ADC ($XX), Y ; ADd with Carry
        adc_x_ind_zp = 0x75, // ADC ($XX, X) ; ADd with Carry
        adc_x_abs    = 0x7D, // ADC $XXXX, X ; ADd with Carry
        sty_zp       = 0x84, // STY $XX      ; STore Y
        sta_zp       = 0x85, // STA $XX      ; STore A
        stx_zp       = 0x86, // STX $XX      ; STore X
        txa_impl     = 0x8A, // TXA          ; Transfer X to A
        sty_abs      = 0x8C, // STY $XXXX    ; STore Y
        sta_abs      = 0x8D, // STA $XXXX    ; STore A
        stx_abs      = 0x8E, // STX $XXXX    ; STore X
        tya_impl     = 0x98, // TYA          ; Transfer Y to A
        ldy_imm      = 0xA0, // LDY #$XX     ; LoaD Y
        lda_x_ind_zp = 0xA1, // LDX ($XX, X) ; LoaD X
        ldx_imm      = 0xA2, // LDX #$XX     ; LoaD X
        lda_zp       = 0xA5, // LDA $XX      ; LoaD A
        ldy_zp       = 0xA4, // LDY $XX      ; LoaD Y
        ldx_zp       = 0xA6, // LDX $XX      ; LoaD X
        tay_impl     = 0xA8, // TAY          ; Transfer A to Y
        lda_imm      = 0xA9, // LDA #$XX     ; LoaD A
        tax_impl     = 0xAA, // TAX          ; Transfer A to X
        lda_abs      = 0xAD, // LDA $XXXX    ; LoaD A
        lda_x_abs    = 0xBD, // STA $XXXX, X ; STore A
        cld_impl     = 0xD8, // CLD          ; CLear Decimal
        sbc_zp       = 0xE5, // SBC $XX      ; SuBtract with Carry
        sbc_imm      = 0xE9, // SBC #$XX     ; SuBtract with Carry
        nop_impl     = 0xEA, // NOP          ; No OPeration
        sbc_abs      = 0xED, // SBC $XXXX    ; SuBtract with Carry
        sed_impl     = 0xF8, // SED          ; SEt Decimal
        // zig fmt: on

        fn getOpcodeMnemonic(tag: Tag) []const u8 {
            var parts = mem.split(u8, @tagName(tag), "_");
            return parts.first();
        }

        pub fn getAddressingMode(tag: Tag) []const u8 {
            var parts = mem.split(u8, @tagName(tag), "_");
            _ = parts.first();
            const two = parts.next().?;
            if (parts.next()) |three| {
                if (parts.next()) |four| {
                    return four;
                } else {
                    return three;
                }
            } else {
                return two;
            }
        }

        /// Returns the register that could be affected by the execution of this opcode,
        /// excluding the status register.
        /// "Could" because a register's value might stay the same after execution of this opcode.
        pub fn getAffectedRegister(tag: Tag) ?Register {
            return switch (tag) {
                .brk_impl => null,
                .clc_impl => null,
                .jsr_abs => null,
                .sec_impl => null,
                .pha_impl => null,
                .jmp_abs => null,
                .rts_impl => null,
                .adc_zp => .a,
                .adc_imm => .a,
                .pla_impl => .a,
                .adc_abs => .a,
                .adc_ind_y_zp => .a,
                .adc_x_ind_zp => .a,
                .adc_x_abs => .a,
                .sty_zp => null,
                .sta_zp => null,
                .stx_zp => null,
                .txa_impl => .a,
                .sty_abs => null,
                .sta_abs => null,
                .stx_abs => null,
                .tya_impl => .a,
                .ldy_imm => .y,
                .lda_x_ind_zp => .a,
                .ldx_imm => .x,
                .lda_zp => .a,
                .ldy_zp => .y,
                .ldx_zp => .x,
                .tay_impl => .y,
                .lda_imm => .a,
                .tax_impl => .x,
                .lda_abs => .a,
                .lda_x_abs => .a,
                .cld_impl => null,
                .sbc_zp => .a,
                .sbc_imm => .a,
                .nop_impl => null,
                .sbc_abs => .a,
                .sed_impl => null,
            };
        }

        test getOpcodeMnemonic {
            try testing.expectEqualStrings("nop", getOpcodeMnemonic(.nop_impl));
            try testing.expectEqualStrings("lda", getOpcodeMnemonic(.lda_x_abs));
            try testing.expectEqualStrings("sta", getOpcodeMnemonic(.sta_zp));
        }

        test getAddressingMode {
            try testing.expectEqualStrings("impl", getAddressingMode(.clc_impl));
            try testing.expectEqualStrings("abs", getAddressingMode(.lda_abs));
            try testing.expectEqualStrings("zp", getAddressingMode(.adc_x_ind_zp));
        }
    };

    /// The payload. This is the operand of the instruction.
    /// The meaning of this data is determined by `tag`.
    pub const Data = union {
        /// The operand, if any, is implied.
        none: void,
        /// The value is immediately available.
        imm: u8,
        /// A zero page memory address.
        zp: u8,
        /// An absolute memory address.
        abs: AbsoluteAddress,
    };

    // TODO: use `extra` for this?
    pub const AbsoluteAddress = union(enum) {
        /// The address is known and immediately available.
        imm: u16,
        /// The absolute memory address is unknown and yet to be resolved by the linker.
        unresolved: struct {
            blk_i: u16,
            offset: u16 = 0,
        },
        /// The current address of this word subtracted by the given offset.
        current: struct {
            decl_index: @import("../../Module.zig").Decl.Index,
            offset: u16,
        },
    };

    /// Checks this combination of tag (opcode) and data (addressing mode) and makes sure it evaluates to a valid instruction.
    /// The reason it is `data: anytype` instead of `data: Data` is that while the argument is comptime-known,
    /// (TODO: try making it comptime along with `data: anytype` in `Func.addInst`)
    /// `Data` is a union and the language does not allow us to check which field is active even if
    /// the union value is comptime-known, so instead we use anonymous structs and check
    /// that the field name matches the addressing mode suffix of the opcode tag.
    pub fn checkCombo(tag: Tag, data: anytype) void {
        const addr_mode = tag.getAddressingMode();
        const data_ty = @TypeOf(data);
        if (@hasField(data_ty, "none")) {
            assert(mem.eql(u8, addr_mode, "impl"));
        } else if (@hasField(data_ty, "imm")) {
            assert(mem.eql(u8, addr_mode, "imm"));
        } else if (@hasField(data_ty, "zp")) {
            assert(mem.eql(u8, addr_mode, "zp"));
        } else if (@hasField(data_ty, "abs")) {
            assert(mem.eql(u8, addr_mode, "abs"));
        } else unreachable;
    }

    /// Returns the size of this instruction, including opcode and operand.
    pub fn getByteSize(inst: Inst) u2 {
        const addr_mode = inst.tag.getAddressingMode();
        if (mem.eql(u8, addr_mode, "impl")) {
            return 1;
        } else if (mem.eql(u8, addr_mode, "imm")) {
            return 2;
        } else if (mem.eql(u8, addr_mode, "zp")) {
            return 2;
        } else if (mem.eql(u8, addr_mode, "abs")) {
            return 3;
        } else unreachable;
    }

    // TODO: function for calculating a program's total exact cycles? and execution time using hertz

    test checkCombo {
        Inst.checkCombo(.rts_impl, .{ .none = {} });
        Inst.checkCombo(.lda_imm, .{ .imm = .{ .imm = 0x10 } });
        Inst.checkCombo(.lda_zp, .{ .zp = .{ .zp = 0x00 } });
        Inst.checkCombo(.jsr_abs, .{ .abs = .{ .imm = 0xFFD2 } });
    }

    test getByteSize {
        try testing.expectEqual(@as(u2, 1), getByteSize(.{ .tag = .brk_impl, .data = .{ .none = {} } }));
        try testing.expectEqual(@as(u2, 2), getByteSize(.{ .tag = .adc_imm, .data = .{ .imm = 0x20 } }));
        try testing.expectEqual(@as(u2, 2), getByteSize(.{ .tag = .adc_x_ind_zp, .data = .{ .zp = 0x40 } }));
        try testing.expectEqual(@as(u2, 2), getByteSize(.{ .tag = .sty_zp, .data = .{ .zp = 0xFF } }));
    }
};

// TODO: are these tests run as part of `zig build test`, too?
comptime {
    _ = Inst;
}
