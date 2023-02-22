export fn entry() void {
    printChar('H');
    printChar('E');
    printChar('L');
    printChar('L');
    printChar('O');
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
