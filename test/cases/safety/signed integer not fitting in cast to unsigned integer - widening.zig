const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .cast_of_negative_value_to_unsigned) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    var value: c_short = -1;
    _ = &value;
    const casted: u32 = @intCast(value);
    _ = casted;
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
