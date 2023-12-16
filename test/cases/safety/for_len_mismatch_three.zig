const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .for_loop_objects_lengths_mismatch) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var slice: []const u8 = "hello";
    _ = &slice;
    for (10..20, slice, 20..30) |a, b, c| {
        _ = a;
        _ = b;
        _ = c;
        return error.TestFailed;
    }
    return error.TestFailed;
}
// run
// backend=llvm
// target=native
