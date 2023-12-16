const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .unreach) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    unreachable;
}
// run
// backend=llvm
// target=native
