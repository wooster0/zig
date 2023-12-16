const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .index_out_of_bounds and cause.index_out_of_bounds.index == 5 and cause.index_out_of_bounds.len == 4) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var buf = [4]u8{ 'a', 'b', 'c', 0 };
    const input: []u8 = &buf;
    const slice = input[0..4 :0];
    _ = slice;
    return error.TestFailed;
}

// run
// backend=llvm
// target=native
