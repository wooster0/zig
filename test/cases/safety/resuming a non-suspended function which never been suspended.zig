const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = message;

    std.process.exit(0);
}
fn foo() void {
    var f = async bar(@frame());
    _ = &f;
    std.os.exit(1);
}

fn bar(frame: anyframe) void {
    suspend {
        resume frame;
    }
    std.os.exit(1);
}

pub fn main() !void {
    _ = async foo();
    return error.TestFailed;
}
// run
// backend=stage1
// target=native
