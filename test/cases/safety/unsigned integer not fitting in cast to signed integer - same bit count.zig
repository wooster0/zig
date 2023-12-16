const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .intcast_truncated_bits) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    var value: u8 = 245;
    _ = &value;
    const casted: i8 = @intCast(value);
    _ = casted;
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
