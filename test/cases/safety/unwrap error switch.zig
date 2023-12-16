const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .unwrap_error and cause.unwrap_error == error.Whatever) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    bar() catch |err| switch (err) {
        error.Whatever => unreachable,
    };
    return error.TestFailed;
}
fn bar() !void {
    return error.Whatever;
}
// run
// backend=llvm
// target=native
