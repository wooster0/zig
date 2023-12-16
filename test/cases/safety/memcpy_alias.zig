const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .memcpy_args_alias) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    var buffer = [2]u8{ 1, 2 } ** 5;
    var len: usize = 5;
    _ = &len;
    @memcpy(buffer[0..len], buffer[4 .. 4 + len]);
}
// run
// backend=llvm
// target=native
