pub fn main() void {
    // pointer loads, stores, basic arithmetic...
    var str = "@ETO".*;
    str[0] += 8; // '@' -> 'H'
    str[2] -= 8; // 'T' -> 'L'
    printChar(str[0]);

    // different ways of accessing the array...
    var ptr = @ptrCast([*]u8, &str);
    // pointer arithmetic...
    ptr += 2;
    ptr -= 1;
    printChar(ptr[0]);
    // adding and subtracting big integers...
    var b: u64 = 0;
    b += str[2];
    b -%= 123456789;
    b +%= 123456789;
    printChar(@intCast(u8, b));
    printChar(str[2]);
    printChar((&str)[3]);

    // wild pointer indirection stuff with big types
    var x: u256 = 'X';
    x = ' ';
    printChar(@intCast(u8, x));
    const y: *u256 = &x;
    y.* = 'W';
    printChar(@intCast(u8, x));
    x = 'O';
    printChar(@intCast(u8, y.*));
    var z: *const *u256 = &y;
    y.* = 'R';
    printChar(@intCast(u8, z.*.*));
    const w = &x;
    z = &w;
    z.*.* = 'L';
    printChar(@intCast(u8, z.*.*));
    var a = &z;
    a.*.*.* = 'D';
    printChar(@intCast(u8, w.*));
}

/// Writes the byte in the A register to the screen - "CHaR OUT".
const CHROUT = @intToPtr(*const fn () void, 0xFFD2);

/// Prints the character to the screen.
fn printChar(char: u8) void {
    // Load the character into the 6502's A register.
    asm volatile ("" // no assembly code
        : // outputs (none)
        // inputs (1):
        : [a] // unused in the assembly code string but mandatory syntax
          "{a}" // A register
          (char), // loadee
    );
    CHROUT();
}
