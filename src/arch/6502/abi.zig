const std = @import("std");
const bits = @import("bits.zig");
const Register = bits.Register;

/// Returns operating system-specific zero page addresses that we are free to use for any purpose.
pub fn getZeroPageAddresses(target: std.Target) std.BoundedArray(u8, 256) {
    // memory maps are used to find this data

    var addrs = std.BoundedArray(u8, 256){};
    // TODO: this is how they do it in LLVM-MOS:
    //       https://github.com/llvm-mos/llvm-mos-sdk/blob/da558fcbfc729f702b24e4bf6f1a30da94e5cbf2/mos-platform/commodore/commodore.ld#L3
    //       why is it ok to overwrite BASIC memory regions in the zp and can we do that too?
    switch (target.os.tag) {
        .c64 => {
            // TODO: find all the other available bytes. here's a memory map:
            // https://sta.c64.org/cbm64mem.html
            addrs.appendSliceAssumeCapacity(
                &[_]u8{ 0x02, 0x2A, 0x52, 0xFB, 0xFC, 0xFD, 0xFE },
            );
        },
        .freestanding => {
            // assume everything is free
            var i: u8 = 0;
            while (i < 256) : (i += 1)
                addrs.appendAssumeCapacity(i);
        },
        else => unreachable,
    }
    return addrs;
}

pub fn getAbsoluteMemoryOffset(target: std.Target) u16 {
    return switch (target.os.tag) {
        // end of the BASIC area
        .c64 => 0x9FFF,
        // this is before the three vectors at the top of the address space (NMI, RESET, IRQ/BRK),
        // each taking up 2 bytes: https://www.pagetable.com/?p=410
        .freestanding => 0xFFFF - 3 * 2,
        else => unreachable,
    };
}
