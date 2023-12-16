const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .cast_of_negative_value_to_unsigned) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var x: @Vector(4, i32) = @splat(-2147483647);
    _ = &x;
    const y: @Vector(4, u32) = @intCast(x);
    _ = y;
    return error.TestFailed;
}

// run
// backend=llvm
// target=native
