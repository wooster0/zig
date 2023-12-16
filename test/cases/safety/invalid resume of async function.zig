const std = @import("std");

pub fn panic(_: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    std.process.exit(0);
}
pub fn main() !void {
    var p = async suspendOnce();
    resume p; //ok
    resume p; //bad
    return error.TestFailed;
}
fn suspendOnce() void {
    suspend {}
}
// run
// backend=stage1
// target=native
