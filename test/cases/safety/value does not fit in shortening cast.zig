const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .intcast_truncated_bits) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    const x = shorten_cast(200);
    if (x == 0) return error.Whatever;
    return error.TestFailed;
}
fn shorten_cast(x: i32) i8 {
    return @intCast(x);
}
// run
// backend=llvm
// target=native
