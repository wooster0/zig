const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (std.mem.eql(u8, message, "'noreturn' function returned")) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
const T = struct {
    export fn bar() void {
        // ...
    }
};

extern fn bar() noreturn;
pub fn main() void {
    _ = T.bar;
    bar();
}
// run
// backend=llvm
// target=native
