const std = @import("std");

pub fn panic(_: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    std.process.exit(0);
}
pub fn main() !void {
    var frame = async first();
    resume frame;
    return error.TestFailed;
}
fn first() void {
    var frame = async other();
    await frame;
}
fn other() void {
    suspend {}
}
// run
// backend=stage1
// target=native
