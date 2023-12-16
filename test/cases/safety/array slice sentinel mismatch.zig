const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .sentinel_mismatch_usize and cause.sentinel_mismatch_usize.expected == 0 and cause.sentinel_mismatch_usize.actual == 4) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var buf: [4]u8 = .{ 1, 2, 3, 4 };
    const slice = buf[0..3 :0];
    _ = slice;
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
