pub const Register = enum(u2) {
    a,
    x,
    y,

    pub fn id(reg: Register) u2 {
        return @enumToInt(reg);
    }
};
