export fn main(r0: u32, r1: u32, atags: u32) callconv(.C) noreturn {
    _ = r0;
    _ = r1;
    _ = atags;
    unreachable; // never gets run so it doesn't matter
}
pub fn panic(_: std.builtin.PanicCause, _: ?*@import("std").builtin.StackTrace, _: ?usize) noreturn {
    while (true) {}
}
