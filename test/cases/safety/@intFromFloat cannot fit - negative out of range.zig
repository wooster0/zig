const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .int_part_of_float_out_of_bounds) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    baz(bar(-129.1));
    return error.TestFailed;
}
fn bar(a: f32) i8 {
    return @intFromFloat(a);
}
fn baz(_: i8) void {}
// run
// backend=llvm
// target=native
