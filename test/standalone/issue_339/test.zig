const std = @import("std");

pub fn panic(_: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    @breakpoint();
    while (true) {}
}

fn bar() anyerror!void {}

export fn foo() void {
    bar() catch unreachable;
}
