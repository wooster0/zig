const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .unwrap_null) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    var ptr: ?*i32 = null;
    _ = &ptr;
    const b = ptr.?;
    _ = b;
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
