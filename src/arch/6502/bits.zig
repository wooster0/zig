const std = @import("std");

// TODO: move into RegMem.zig?
pub const Reg = enum(u2) {
    a,
    x,
    y,

    pub fn parse(name: []const u8) ?Reg {
        return std.meta.stringToEnum(Reg, name);
    }
};
