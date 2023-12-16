export fn a() void {
    const x: [50:.{1}][1]u8 = undefined;
    _ = x;
}
export fn b() void {
    const x: [*:.{ 1, 2 }]const [2]u8 = undefined;
    _ = x;
}
export fn c() void {
    const x: [:.{1}][1]u8 = undefined;
    _ = x;
}

// error
// backend=stage2
// target=native
//
// :2:19: error: non-scalar sentinel '[1]u8' not allowed
// :6:18: error: non-scalar sentinel '[2]u8' not allowed
// :10:17: error: non-scalar sentinel '[1]u8' not allowed
