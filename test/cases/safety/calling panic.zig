const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .msg and std.mem.eql(u8, cause.msg, "oh no")) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    if (true) @panic("oh no");
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
