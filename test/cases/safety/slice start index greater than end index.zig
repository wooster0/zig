const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .start_index_greater_than_end and cause.start_index_greater_than_end.start == 10 and cause.start_index_greater_than_end.end == 1) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

pub fn main() !void {
    var a: usize = 1;
    var b: usize = 10;
    _ = .{ &a, &b };
    var buf: [16]u8 = undefined;

    const slice = buf[b..a];
    _ = slice;
    return error.TestFailed;
}

// run
// backend=llvm
// target=native
