const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .right_shift_overflow) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    const x = shr(0b0010111111111111, 3);
    if (x == 0) return error.Whatever;
    return error.TestFailed;
}
fn shr(a: u16, b: u4) u16 {
    return @shrExact(a, b);
}
// run
// backend=llvm
// target=native
