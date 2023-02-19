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
const Module = @import("../../Module.zig");
const Reg = bits.Reg;

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
    // TODO: rename these fields to opcode and operand? as well as the structs/unions
    /// The instruction's opcode.
    tag: Tag,
    /// The instruction's operand with an addressing mode.
    data: Data,

    /// The position of an MIR instruction within the `Mir` instructions array.
    pub const Index = u32;

    /// The 6502 has 151 unique instructions. Organized into addressing modes, it has 56 instructions.
    ///
    /// The 6502 has these addressing modes:
    /// * Implied (impl) (e.g. `BRK`):
    ///   The operand, if any, is implicitly stated in the operation code of the instruction itself.
    /// * Immediate (imm) (e.g. `LDA #4`):
    ///   Value is the one-byte operand.
    /// * Absolute (abs) (e.g. `LDA $6502`):
    ///   Value is read from absolute address as the two-byte operand.
    /// * X-indexed absolute (x_abs) (e.g. `LDA $6502, X`):
    ///   Value is read from absolute address formed by adding the X register to the two-byte operand.
    /// * Y-indexed absolute (y_abs) (e.g. `LDA $6502, Y`):
    ///   Value is read from absolute address formed by adding the Y register to the two-byte operand.
    /// * Indirect absolute (ind_abs) (e.g. `JMP ($6502)`):
    ///   Value is read from absolute address read from the given address.
    /// * Zero page (zp) (e.g. `LDA $65`):
    ///   Value is read from zero page address.
    /// * X-indexed zero page (x_zp) (e.g. `LDA $65, X`):
    ///   Value is read from zero page address formed by adding X to the one-byte operand.
    /// * Y-indexed zero page (y_zp) (e.g. `LDA $65, Y`):
    ///   Value is read from zero page address formed by adding Y to the one-byte operand.
    /// * X-indexed indirect zero page (x_ind_zp) (e.g. `LDA ($65, X)`):
    ///   The X register's value is added to the one-byte operand resulting in a zero page address
    ///   from which a two-byte address is read from which the value is read.
    ///   This allows dereferencing pointers.
    ///   The zero page boundary is never crossed.
    /// * Indirect Y-indexed zero page (ind_y_zp) (e.g. `LDA ($65), Y`):
    ///   A 2-byte address is from the given zero page address, the Y register's value is added to the 2-byte address
    ///   and then the value is read from that address.
    ///   This allows dereferencing pointers.
    ///   The zero page boundary is never crossed.
    /// * Relative (rel) (e.g. `BEQ $40`):
    ///   The operand is a signed byte that is added to the program counter, which allows jumping relatively +-128 bytes from the instruction following this one.
    // Design decision note:
    // Because the layout of an instruction byte is supposedly in the form of
    // AAABBBCC, where AA and CC define the operation and BBB the addressing mode,
    // we could assemble the byte according to that format but the problem with that is that
    // we would deal with possible ambiguities and exceptions to that rule.
    // It seems simpler to define only the opcodes we actually need. This way it is trivial to extend, too.
    pub const Tag = enum(u8) {
        // The tag name must be the opcode's mnemonic followed by the addressing mode as a suffix.
        // Instructions are sorted by opcode.
        // zig fmt: off
        brk_impl     = 0x00, // BRK          ; BReaK
        ora_zp       = 0x05, // ORA $XX      ; OR A
        ora_imm      = 0x09, // ORA          ; OR A
        ora_abs      = 0x0D, // ORA $XXXX    ; OR A
        ora_x_zp     = 0x15, // ORA $XX, X   ; OR A
        ora_x_abs    = 0x1D, // ORA $XXXX, X ; OR A
        clc_impl     = 0x18, // CLC          ; CLear Carry
        jsr_abs      = 0x20, // JSR $XXXX    ; Jump to SubRoutine
        and_zp       = 0x25, // AND $XX      ; AND
        and_imm      = 0x29, // AND #$XX     ; AND
        and_abs      = 0x2D, // AND $XXXX    ; AND
        and_x_zp     = 0x35, // AND $XX, X   ; AND
        sec_impl     = 0x38, // SEC          ; SEt Carry
        and_x_abs    = 0x3D, // AND $XXXX, X ; AND
        eor_zp       = 0x45, // EOR $XX      ; Exclusive OR
        pha_impl     = 0x48, // PHA          ; PusH A
        eor_imm      = 0x49, // EOR #$XX     ; Exclusive OR
        jmp_abs      = 0x4C, // JMP $XXXX    ; JuMP
        eor_abs      = 0x4D, // EOR $XXXX    ; Exclusive OR
        eor_x_zp     = 0x55, // EOR $XX, X   ; Exclusive OR
        eor_x_abs    = 0x5D, // EOR $XXXX, X ; Exclusive OR
        rts_impl     = 0x60, // RTS          ; ReTurn from Subroutine
        adc_zp       = 0x65, // ADC $XX      ; ADd with Carry
        adc_imm      = 0x69, // ADC #$XX     ; ADd with Carry
        pla_impl     = 0x68, // PLA          ; PuLl A
        adc_abs      = 0x6D, // ADC $XXXX    ; ADd with Carry
        adc_ind_y_zp = 0x71, // ADC ($XX), Y ; ADd with Carry
        adc_x_ind_zp = 0x75, // ADC ($XX, X) ; ADd with Carry
        adc_x_abs    = 0x7D, // ADC $XXXX, X ; ADd with Carry
        sta_x_ind_zp = 0x81, // STA ($XX, X) ; STore A
        sty_zp       = 0x84, // STY $XX      ; STore Y
        sta_zp       = 0x85, // STA $XX      ; STore A
        stx_zp       = 0x86, // STX $XX      ; STore X
        dey_impl     = 0x88, // DEY          ; DEcrement Y
        txa_impl     = 0x8A, // TXA          ; Transfer X to A
        sty_abs      = 0x8C, // STY $XXXX    ; STore Y
        sta_abs      = 0x8D, // STA $XXXX    ; STore A
        stx_abs      = 0x8E, // STX $XXXX    ; STore X
        bcc_rel      = 0x90, // BCC $XX      ; Branch on Carry Clear
        sta_ind_y_zp = 0x91, // STA ($XX), Y ; STore A
        sta_x_zp     = 0x95, // STA $XX, X   ; STore A
        tya_impl     = 0x98, // TYA          ; Transfer Y to A
        sta_x_abs    = 0x9D, // STA $XXXX, X ; STore A
        ldy_imm      = 0xA0, // LDY #$XX     ; LoaD Y
        lda_x_ind_zp = 0xA1, // LDX ($XX, X) ; LoaD X
        ldx_imm      = 0xA2, // LDX #$XX     ; LoaD X
        lda_zp       = 0xA5, // LDA $XX      ; LoaD A
        ldy_zp       = 0xA4, // LDY $XX      ; LoaD Y
        ldx_zp       = 0xA6, // LDX $XX      ; LoaD X
        tay_impl     = 0xA8, // TAY          ; Transfer A to Y
        lda_imm      = 0xA9, // LDA #$XX     ; LoaD A
        tax_impl     = 0xAA, // TAX          ; Transfer A to X
        ldy_abs      = 0xAC, // LDY $XXXX    ; LoaD Y
        lda_abs      = 0xAD, // LDA $XXXX    ; LoaD A
        ldx_abs      = 0xAE, // LDX $XXXX    ; LoaD X
        lda_ind_y_zp = 0xB1, // LDA ($XX), Y ; LoaD A
        lda_x_zp     = 0xB5, // LDA $XX, X   ; LoaD A
        lda_x_abs    = 0xBD, // STA $XXXX, X ; STore A
        cpy_imm      = 0xC0, // CPY #$XX     ; ComPare Y
        cmp_zp       = 0xC5, // CMP $XX      ; CoMPare
        dec_zp       = 0xC6, // DEC $XX      ; DECrement
        iny_impl     = 0xC8, // INY          ; INcrement Y
        dex_impl     = 0xCA, // DEX          ; DEcrement X
        cmp_abs      = 0xCD, // CMP $XXXX    ; CoMPare
        dec_abs      = 0xCE, // DEC $XXXX    ; DECrement
        bne_rel      = 0xD0, // BNE $XX      ; Branch on Not Equal
        cld_impl     = 0xD8, // CLD          ; CLear Decimal
        cpx_imm      = 0xE0, // CPX #$XX     ; ComPare X
        sbc_zp       = 0xE5, // SBC $XX      ; SuBtract with Carry
        inc_zp       = 0xE6, // INC $XX      ; INCrement
        inx_impl     = 0xE8, // INX          ; INcrement X
        sbc_imm      = 0xE9, // SBC #$XX     ; SuBtract with Carry
        nop_impl     = 0xEA, // NOP          ; No OPeration
        sbc_abs      = 0xED, // SBC $XXXX    ; SuBtract with Carry
        inc_abs      = 0xEE, // INC $XXXX    ; INCrement
        sed_impl     = 0xF8, // SED          ; SEt Decimal
        // zig fmt: on

        /// Returns the opcode's mnemonic in lowercase.
        fn getOpcodeMnemonic(tag: Tag) []const u8 {
            var parts = mem.split(u8, @tagName(tag), "_");
            return parts.first();
        }

        const AddrMode = enum { impl, imm, abs, x_abs, y_abs, ind_abs, zp, x_zp, y_zp, x_ind_zp, ind_y_zp, rel };
        /// Returns the opcode's addressing mode.
        pub fn getAddrMode(tag: Tag) AddrMode {
            var parts = mem.split(u8, @tagName(tag), "_");
            _ = parts.first();
            return std.meta.stringToEnum(AddrMode, parts.rest()).?;
        }

        /// Returns the register affected by execution of the given opcode.
        pub fn getAffectedReg(tag: Tag) ?Reg {
            return switch (tag) {
                .brk_impl => null,
                .ora_zp => .a,
                .ora_imm => .a,
                .ora_abs => .a,
                .ora_x_zp => .a,
                .ora_x_abs => .a,
                .clc_impl => null,
                .jsr_abs => null,
                .and_zp => .a,
                .and_imm => .a,
                .and_abs => .a,
                .and_x_zp => .a,
                .sec_impl => null,
                .and_x_abs => .a,
                .eor_zp => .a,
                .pha_impl => null,
                .eor_imm => .a,
                .jmp_abs => null,
                .eor_abs => .a,
                .eor_x_zp => .a,
                .eor_x_abs => .a,
                .rts_impl => null,
                .adc_zp => .a,
                .adc_imm => .a,
                .pla_impl => .a,
                .adc_abs => .a,
                .adc_ind_y_zp => .a,
                .adc_x_ind_zp => .a,
                .adc_x_abs => .a,
                .sta_x_ind_zp => null,
                .sty_zp => null,
                .sta_zp => null,
                .stx_zp => null,
                .dey_impl => .y,
                .txa_impl => .a,
                .sty_abs => null,
                .sta_abs => null,
                .stx_abs => null,
                .bcc_rel => null,
                .sta_ind_y_zp => null,
                .sta_x_zp => null,
                .tya_impl => .a,
                .sta_x_abs => null,
                .ldy_imm => .y,
                .lda_x_ind_zp => .a,
                .ldx_imm => .x,
                .lda_zp => .a,
                .ldy_zp => .y,
                .ldx_zp => .x,
                .tay_impl => .y,
                .lda_imm => .a,
                .tax_impl => .x,
                .ldy_abs => .y,
                .lda_abs => .a,
                .ldx_abs => .x,
                .lda_ind_y_zp => .a,
                .lda_x_zp => .a,
                .lda_x_abs => .a,
                .cpy_imm => null,
                .cmp_zp => null,
                .dec_zp => null,
                .iny_impl => .y,
                .dex_impl => .x,
                .cmp_abs => null,
                .dec_abs => null,
                .bne_rel => null,
                .cld_impl => null,
                .cpx_imm => null,
                .sbc_zp => .a,
                .inc_zp => null,
                .inx_impl => .x,
                .sbc_imm => .a,
                .nop_impl => null,
                .sbc_abs => .a,
                .inc_abs => null,
                .sed_impl => null,
            };
        }

        test getOpcodeMnemonic {
            try testing.expectEqualStrings("nop", getOpcodeMnemonic(.nop_impl));
            try testing.expectEqualStrings("lda", getOpcodeMnemonic(.lda_x_abs));
            try testing.expectEqualStrings("sta", getOpcodeMnemonic(.sta_zp));
            try testing.expectEqualStrings("jmp", getOpcodeMnemonic(.jmp_abs));
        }

        test getAddrMode {
            try testing.expectEqual(AddrMode.impl, getAddrMode(.clc_impl));
            try testing.expectEqual(AddrMode.abs, getAddrMode(.lda_abs));
            try testing.expectEqual(AddrMode.x_ind_zp, getAddrMode(.adc_x_ind_zp));
            try testing.expectEqual(AddrMode.x_abs, getAddrMode(.lda_x_abs));
        }
    };

    /// The payload. This is the operand of the opcode.
    /// The meaning of this data is determined by `tag`.
    pub const Data = union {
        /// The operand, if any, is implied.
        none: void,
        /// An immediately available value.
        imm: Imm,
        /// A zero page address.
        zp: u8,
        /// An absolute address.
        abs: Abs,
        /// A relative offset.
        rel: i8,
    };

    pub const Imm = union(enum) {
        val: u8,
        /// The half of an unresolved address.
        unres_addr_half: UnresAddrHalf,
    };

    /// An unknown memory address that is yet to be resolved by the linker.
    pub const UnresAddr = struct {
        /// The index of the block that we want to know the memory address of.
        // TODO: rename to `index` to be more compatible with the terminology of other linkers/output formats?
        //       and then "index of the block" -> "index of the symbol"? maybe rename `unres` to `sym`
        block_index: u16,
        /// This is to be added to the memory address once it is resolved.
        /// This might represent an index or an offset into the data.
        addend: u16 = 0,

        pub fn index(unres_addr: UnresAddr, offset: u16) UnresAddr {
            assert(unres_addr.addend == 0); // Unresolved addresses should be indexed only once.
            return .{ .block_index = unres_addr.block_index, .addend = offset };
        }

        pub fn takeHalf(unres_addr: UnresAddr, half: enum { low, high }) UnresAddrHalf {
            assert(unres_addr.addend == 0); // We won't need the addend.
            return .{ .block_index = unres_addr.block_index, .half = @enumToInt(half) };
        }
    };
    /// An unknown low or high byte half of an memory address that is yet to be resolved by the linker.
    pub const UnresAddrHalf = struct {
        /// The index of the block that we want to know the memory address of.
        // TODO: rename to `index` to be more compatible with the terminology of other linkers/output formats?
        block_index: u16,
        /// The half this resolves to: either the low byte or the high byte half of a word.
        half: u1,
    };
    const DeclAddr = struct {
        /// The index of the declaration to resolve the address of.
        index: Module.Decl.Index,
        /// This is to be added to the address once it is resolved.
        /// This might represent an index or an offset.
        addend: u16,
    };
    /// An absolute memory address.
    pub const Abs = union(enum) {
        /// The address is known and fixed and does not change.
        // Design decision note: "const" is a keyword and "static" is one letter longer.
        //                       "resolved" is a bad because it was never unresolved to begin with.
        fixed: u16,
        /// The memory address is unknown and yet to be resolved by the linker.
        unres: UnresAddr,
        /// The address of a declaration.
        decl: DeclAddr,
    };

    /// Returns the size of this instruction in bytes, including opcode and operand.
    pub fn getSize(inst: Inst) u2 {
        const addr_mode = inst.tag.getAddrMode();
        const operand_size: u2 = switch (addr_mode) {
            .impl => 0,
            .imm => 1,
            .zp, .x_zp, .y_zp, .x_ind_zp, .ind_y_zp => 1,
            .abs, .x_abs, .y_abs, .ind_abs => 2,
            .rel => 1,
        };
        const opcode_size = @sizeOf(Tag);
        return opcode_size + operand_size;
    }

    /// Returns an all-uppercase assembly text code representation of this instruction.
    // TODO: use this for -femit-asm once we can. also support '%' binary.
    pub fn getTextRepr(inst: Inst, buf: *[20]u8) []const u8 {
        const addr_mode = inst.tag.getAddrMode();
        var opcode_mnemonic_buf: [3]u8 = undefined;
        const opcode_mnemonic = std.ascii.upperString(&opcode_mnemonic_buf, inst.tag.getOpcodeMnemonic());
        const prefix = switch (addr_mode) {
            .zp, .abs => "",
            .x_zp, .y_zp, .x_abs, .y_abs => "",
            .x_ind_zp, .ind_y_zp => "(",
            .ind_abs => "(",
            else => undefined,
        };
        const suffix = switch (addr_mode) {
            .zp, .abs => "",
            .x_zp, .x_abs => ", X",
            .y_zp, .y_abs => ", Y",
            .x_ind_zp => ", X)",
            .ind_y_zp => "), Y",
            .ind_abs => "",
            else => undefined,
        };
        const slice = switch (addr_mode) {
            .impl => std.fmt.bufPrint(buf, "{s}", .{opcode_mnemonic}),
            .imm => switch (inst.data.imm) {
                .val => |val| std.fmt.bufPrint(buf, "{s} #${X:0>2}", .{ opcode_mnemonic, val }),
                // TODO: this shouldn't be emitted on -femit-asm; resolve this beforehand to `imm`
                .unres_addr_half => |unres_addr_half| std.fmt.bufPrint(buf, "{s} #${s}", .{
                    opcode_mnemonic, switch (unres_addr_half.half) {
                        0 => "LO",
                        1 => "HI",
                    },
                }),
            },
            .zp, .x_zp, .y_zp, .x_ind_zp, .ind_y_zp => std.fmt.bufPrint(buf, "{s} {s}${X:0>2}{s}", .{ opcode_mnemonic, prefix, inst.data.zp, suffix }),
            .abs, .x_abs, .y_abs, .ind_abs => switch (inst.data.abs) {
                .fixed => |addr| std.fmt.bufPrint(buf, "{s} {s}${X:0>4}{s}", .{ opcode_mnemonic, prefix, addr, suffix }),
                // TODO: this shouldn't be emitted on -femit-asm; resolve this beforehand to `fixed`
                inline .unres, .decl => |unres| std.fmt.bufPrint(buf, "{s} {s}$???? + {d}{s}", .{ opcode_mnemonic, prefix, unres.addend, suffix }),
            },
            .rel => std.fmt.bufPrint(buf, "{s} {d:1}", .{ opcode_mnemonic, inst.data.rel }),
        };
        return slice catch unreachable;
    }

    // TODO: function for calculating a program's total exact cycles? and execution time using hertz

    test getSize {
        try testing.expectEqual(@as(u2, 1), getSize(.{ .tag = .nop_impl, .data = .{ .none = {} } }));
        try testing.expectEqual(@as(u2, 1), getSize(.{ .tag = .brk_impl, .data = .{ .none = {} } }));
        try testing.expectEqual(@as(u2, 2), getSize(.{ .tag = .adc_imm, .data = .{ .imm = .{ .val = 0x20 } } }));
        try testing.expectEqual(@as(u2, 2), getSize(.{ .tag = .adc_x_ind_zp, .data = .{ .zp = 0x40 } }));
        try testing.expectEqual(@as(u2, 2), getSize(.{ .tag = .sty_zp, .data = .{ .zp = 0xFF } }));
        try testing.expectEqual(@as(u2, 3), getSize(.{ .tag = .lda_x_abs, .data = .{ .abs = .{ .fixed = 0xABCD } } }));
        try testing.expectEqual(@as(u2, 3), getSize(.{ .tag = .lda_abs, .data = .{ .abs = .{ .fixed = 0x0801 } } }));
    }

    test getTextRepr {
        var buf: [20]u8 = undefined;
        try testing.expectEqualStrings("NOP", getTextRepr(.{ .tag = .nop_impl, .data = .{ .none = {} } }, &buf));
        try testing.expectEqualStrings("LDA $EE", getTextRepr(.{ .tag = .lda_zp, .data = .{ .zp = 0xEE } }, &buf));
        try testing.expectEqualStrings("STA $02", getTextRepr(.{ .tag = .sta_zp, .data = .{ .zp = 0x02 } }, &buf));
        try testing.expectEqualStrings("BCC +16", getTextRepr(.{ .tag = .bcc_rel, .data = .{ .rel = 16 } }, &buf));
        try testing.expectEqualStrings("BCC -32", getTextRepr(.{ .tag = .bcc_rel, .data = .{ .rel = -32 } }, &buf));
        try testing.expectEqualStrings("STA ($20, X)", getTextRepr(.{ .tag = .sta_x_ind_zp, .data = .{ .zp = 0x20 } }, &buf));
        try testing.expectEqualStrings("ADC ($AD), Y", getTextRepr(.{ .tag = .adc_ind_y_zp, .data = .{ .zp = 0xAD } }, &buf));
        try testing.expectEqualStrings("LDA #$04", getTextRepr(.{ .tag = .lda_imm, .data = .{ .imm = .{ .val = 4 } } }, &buf));
        try testing.expectEqualStrings("LDA #$0F", getTextRepr(.{ .tag = .lda_imm, .data = .{ .imm = .{ .val = 0x0F } } }, &buf));
        try testing.expectEqualStrings("STX $0100", getTextRepr(.{ .tag = .stx_abs, .data = .{ .abs = .{ .fixed = 0x0100 } } }, &buf));
        try testing.expectEqualStrings("STY $ABCD", getTextRepr(.{ .tag = .sty_abs, .data = .{ .abs = .{ .fixed = 0xABCD } } }, &buf));
        try testing.expectEqualStrings("ADC $FFFF, X", getTextRepr(.{ .tag = .adc_x_abs, .data = .{ .abs = .{ .fixed = 0xFFFF } } }, &buf));
        try testing.expectEqualStrings("LDA $???? + 65535, X", getTextRepr(.{ .tag = .lda_x_abs, .data = .{ .abs = .{ .unres = .{ .block_index = undefined, .addend = 0xFFFF } } } }, &buf));
    }
};

// TODO(meeting): are these tests run as part of `zig build test`, too?
comptime {
    _ = Inst;
}
