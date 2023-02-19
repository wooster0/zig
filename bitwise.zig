pub fn main() void {
    var x: u128 = 'z';
    x &= 0b11011111;
    printChar(@intCast(u8, x));
    x = x | 0b00100000;
    x = x & 0b11011111;
    printChar(@intCast(u8, x));
    x ^= 0b1010_1010;
    x ^= 0b0101_0101;
    printChar(@intCast(u8, x));
    printChar(165);
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
          (@intCast(u8, char)), // loadee
    );
    CHROUT();
}
