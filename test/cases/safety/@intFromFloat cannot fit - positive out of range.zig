const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .int_part_of_float_out_of_bounds) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    baz(bar(256.2));
    return error.TestFailed;
}
fn bar(a: f32) u8 {
    return @intFromFloat(a);
}
fn baz(_: u8) void {}
// run
// backend=llvm
// target=native
