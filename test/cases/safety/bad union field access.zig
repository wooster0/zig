const std = @import("std");

pub fn panic(cause: std.builtin.PanicCause, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    if (cause == .inactive_union_field and std.mem.eql(u8, cause.inactive_union_field.active, "int") and std.mem.eql(u8, cause.inactive_union_field.accessed, "float")) {
        std.process.exit(0);
    }
    std.process.exit(1);
}

const Foo = union {
    float: f32,
    int: u32,
};

pub fn main() !void {
    var f = Foo{ .int = 42 };
    bar(&f);
    return error.TestFailed;
}

fn bar(f: *Foo) void {
    f.float = 12.34;
}
// run
// backend=llvm
// target=native
