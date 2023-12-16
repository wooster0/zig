const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .invalid_error_code) {
        std.process.exit(0);
    }
    std.process.exit(1);
}
pub fn main() !void {
    const bar: error{Foo}!i32 = @errorCast(foo());
    _ = &bar;
    return error.TestFailed;
}
fn foo() anyerror!i32 {
    return error.Bar;
}
// run
// backend=llvm
// target=native
