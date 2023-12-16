const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .incorrect_alignment) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    var x: usize = 5;
    _ = &x;
    const y: [*]align(4) u8 = @ptrFromInt(x);
    _ = y;
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
