const std = @import("std");

pub const Register = enum(u2) {
    a,
    x,
    y,

    pub fn parse(name: []const u8) ?Register {
        return std.meta.stringToEnum(Register, name);
    }
};
