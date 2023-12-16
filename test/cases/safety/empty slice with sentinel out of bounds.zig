const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .index_out_of_bounds and cause.index_out_of_bounds.index == 1 and cause.index_out_of_bounds.len == 0) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var buf_zero = [0]u8{};
    const input: []u8 = &buf_zero;
    const slice = input[0..0 :0];
    _ = slice;
    return error.TestFailed;
}

// run
// backend=llvm
// target=native
