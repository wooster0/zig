//! The 7-bit [ASCII](https://en.wikipedia.org/wiki/ASCII) character encoding standard.
//!
//! This is not to be confused with the 8-bit [Extended ASCII](https://en.wikipedia.org/wiki/Extended_ASCII).
//!
//! Even though this module concerns itself with 7-bit ASCII,
//! functions use `u8` as the type instead of `u7` for convenience and compatibility.
//! Characters outside of the 7-bit range are gracefully handled (e.g. by returning `false`).
//!
//! See also: https://en.wikipedia.org/wiki/ASCII#Character_set

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

/// Contains constants for the C0 control codes of the ASCII encoding.
///
/// See also: https://en.wikipedia.org/wiki/C0_and_C1_control_codes and `is_control`
pub const control = struct {
    pub const NUL = 0x00;
    pub const SOH = 0x01;
    pub const STX = 0x02;
    pub const ETX = 0x03;
    pub const EOT = 0x04;
    pub const ENQ = 0x05;
    pub const ACK = 0x06;
    pub const BEL = 0x07;
    pub const BS = 0x08;
    pub const TAB = 0x09;
    pub const LF = 0x0A;
    pub const VT = 0x0B;
    pub const FF = 0x0C;
    pub const CR = 0x0D;
    pub const SO = 0x0E;
    pub const SI = 0x0F;
    pub const DLE = 0x10;
    pub const DC1 = 0x11;
    pub const DC2 = 0x12;
    pub const DC3 = 0x13;
    pub const DC4 = 0x14;
    pub const NAK = 0x15;
    pub const SYN = 0x16;
    pub const ETB = 0x17;
    pub const CAN = 0x18;
    pub const EM = 0x19;
    pub const SUB = 0x1A;
    pub const ESC = 0x1B;
    pub const FS = 0x1C;
    pub const GS = 0x1D;
    pub const RS = 0x1E;
    pub const US = 0x1F;

    pub const DEL = 0x7F;

    /// An alias to `DC1`.
    pub const XON = 0x11;
    /// An alias to `DC3`.
    pub const XOFF = 0x13;
};

// These naive functions are used to generate the lookup table
// and as fallbacks for if the lookup table isn't available.
//
// Note that some functions like for example `isDigit` don't use a table because it's slower.
// Using a table is generally only useful if not all `true` values in the table would be in one row.

fn isControlNaive(char: u8) bool {
    return char <= control.US or char == control.DEL;
}
fn isAlphabeticNaive(char: u8) bool {
    return isLower(char) or isUpper(char);
}
fn isHexadecimalNaive(char: u8) bool {
    return isDigit(char) or
        (char >= 'a' and char <= 'f') or
        (char >= 'A' and char <= 'F');
}
fn isAlphanumericNaive(char: u8) bool {
    return isDigit(char) or isAlphabeticNaive(char);
}
fn isWhitespaceNaive(char: u8) bool {
    @setEvalBranchQuota(5000);
    return mem.indexOfScalar(u8, &whitespace, char) != null;
}

/// A lookup table.
const CombinedTable = struct {
    table: [256]u8,

    const Index = enum {
        control,
        alphabetic,
        hexadecimal,
        alphanumeric,
        whitespace,
    };

    /// Generates a table which is filled with the results of the given function for all characters.
    fn getBoolTable(comptime condition: fn (u8) bool) [128]bool {
        @setEvalBranchQuota(2000);
        comptime var table: [128]bool = undefined;
        comptime var index = 0;
        while (index < 128) : (index += 1) {
            table[index] = condition(index);
        }
        return table;
    }

    fn init() CombinedTable {
        comptime var table: [256]u8 = undefined;

        const control_table = comptime getBoolTable(isControlNaive);
        const alpha_table = comptime getBoolTable(isAlphabeticNaive);
        const hex_table = comptime getBoolTable(isHexadecimalNaive);
        const alphanumeric_table = comptime getBoolTable(isAlphanumericNaive);
        const whitespace_table = comptime getBoolTable(isWhitespaceNaive);

        comptime var i = 0;
        inline while (i < 128) : (i += 1) {
            table[i] =
                @boolToInt(control_table[i]) << @enumToInt(Index.control) |
                @boolToInt(alpha_table[i]) << @enumToInt(Index.alphabetic) |
                @boolToInt(hex_table[i]) << @enumToInt(Index.hexadecimal) |
                @boolToInt(alphanumeric_table[i]) << @enumToInt(Index.alphanumeric) |
                @boolToInt(whitespace_table[i]) << @enumToInt(Index.whitespace);
        }

        mem.set(u8, table[128..256], 0);

        return .{ .table = table };
    }

    fn contains(self: CombinedTable, char: u8, index: Index) bool {
        return (self.table[char] & (@as(u8, 1) << @enumToInt(index))) != 0;
    }
};

/// The combined table for fast lookup.
///
/// This is not used in `ReleaseSmall` to save 256 bytes at the cost of
/// a small decrease in performance.
const combined_table: ?CombinedTable = if (@import("builtin").mode == .ReleaseSmall)
    null
else
    CombinedTable.init();

/// Returns whether the character is a control character.
///
/// See also: `control`
pub fn isControl(char: u8) bool {
    if (combined_table) |table|
        return table.contains(char, .control)
    else
        return isControlNaive(char);
}

/// Returns whether the character is alphanumeric. This is case-insensitive.
pub fn isAlphanumeric(char: u8) bool {
    if (combined_table) |table|
        return table.contains(char, .alphanumeric)
    else
        return isAlphanumericNaive(char);
}

/// Returns whether the character is alphabetic. This is case-insensitive.
pub fn isAlphabetic(char: u8) bool {
    if (combined_table) |table|
        return table.contains(char, .alphabetic)
    else
        return isAlphabeticNaive(char);
}

pub fn isDigit(char: u8) bool {
    return char >= '0' and char <= '9';
}

/// Returns whether the character has some graphical representation and can be printed.
pub fn isPrintable(char: u8) bool {
    return char >= ' ' and char <= '~';
}

pub fn isLower(char: u8) bool {
    return char >= 'a' and char <= 'z';
}

pub fn isUpper(char: u8) bool {
    return char >= 'A' and char <= 'Z';
}

pub fn isWhitespace(char: u8) bool {
    if (combined_table) |table|
        return table.contains(char, .whitespace)
    else
        return isWhitespaceNaive(char);
}

/// All the values for which `isWhitespace()` returns `true`.
/// This may be used with e.g. `std.mem.trim()` to trim whitespace.
pub const whitespace = [_]u8{ ' ', '\t', '\n', '\r', control.VT, control.FF };

test "whitespace" {
    for (whitespace) |char| try testing.expect(isWhitespace(char));

    var i: u8 = 0;
    while (isASCII(i)) : (i += 1) {
        if (isWhitespace(i)) try testing.expect(mem.indexOfScalar(u8, &whitespace, i) != null);
    }
}

/// Returns whether the character is a hexadecimal digit. This is case-insensitive.
pub fn isHexadecimal(char: u8) bool {
    if (combined_table) |table|
        return table.contains(char, .hexadecimal)
    else
        return isHexadecimalNaive(char);
}

pub fn isASCII(char: u8) bool {
    return char < 128;
}

pub fn toUpper(char: u8) u8 {
    if (isLower(char)) {
        return char & 0b11011111;
    } else {
        return char;
    }
}

pub fn toLower(char: u8) u8 {
    if (isUpper(char)) {
        return char | 0b00100000;
    } else {
        return char;
    }
}

test "ascii character classes" {
    try testing.expect(!isControl('a'));
    try testing.expect(!isControl('z'));
    try testing.expect(isControl(control.NUL));
    try testing.expect(isControl(control.FF));
    try testing.expect(isControl(control.US));

    try testing.expect('C' == toUpper('c'));
    try testing.expect(':' == toUpper(':'));
    try testing.expect('\xab' == toUpper('\xab'));
    try testing.expect(!isUpper('z'));

    try testing.expect('c' == toLower('C'));
    try testing.expect(':' == toLower(':'));
    try testing.expect('\xab' == toLower('\xab'));
    try testing.expect(!isLower('Z'));

    try testing.expect(isAlphanumeric('Z'));
    try testing.expect(isAlphanumeric('z'));
    try testing.expect(isAlphanumeric('5'));
    try testing.expect(isAlphanumeric('5'));
    try testing.expect(!isAlphanumeric('!'));

    try testing.expect(!isAlphabetic('5'));
    try testing.expect(isAlphabetic('c'));
    try testing.expect(!isAlphabetic('5'));

    try testing.expect(isWhitespace(' '));
    try testing.expect(isWhitespace('\t'));
    try testing.expect(isWhitespace('\r'));
    try testing.expect(isWhitespace('\n'));
    try testing.expect(!isWhitespace('.'));

    try testing.expect(!isHexadecimal('g'));
    try testing.expect(isHexadecimal('b'));
    try testing.expect(isHexadecimal('9'));

    try testing.expect(!isDigit('~'));
    try testing.expect(isDigit('0'));
    try testing.expect(isDigit('9'));

    try testing.expect(isPrintable(' '));
    try testing.expect(isPrintable('@'));
    try testing.expect(isPrintable('~'));
    try testing.expect(!isPrintable(control.ESC));
}

/// Writes a lower case copy of `ascii_string` to `output`.
/// Asserts `output.len >= ascii_string.len`.
pub fn lowerString(output: []u8, ascii_string: []const u8) []u8 {
    std.debug.assert(output.len >= ascii_string.len);
    for (ascii_string) |char, i| {
        output[i] = toLower(char);
    }
    return output[0..ascii_string.len];
}

test "lowerString" {
    var buf: [1024]u8 = undefined;
    const result = lowerString(&buf, "aBcDeFgHiJkLmNOPqrst0234+💩!");
    try testing.expectEqualStrings("abcdefghijklmnopqrst0234+💩!", result);
}

/// Allocates a lower case copy of `ascii_string`.
/// Caller owns returned string and must free with `allocator`.
pub fn allocLowerString(allocator: mem.Allocator, ascii_string: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, ascii_string.len);
    return lowerString(result, ascii_string);
}

test "allocLowerString" {
    const result = try allocLowerString(testing.allocator, "aBcDeFgHiJkLmNOPqrst0234+💩!");
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("abcdefghijklmnopqrst0234+💩!", result);
}

/// Writes an upper case copy of `ascii_string` to `output`.
/// Asserts `output.len >= ascii_string.len`.
pub fn upperString(output: []u8, ascii_string: []const u8) []u8 {
    std.debug.assert(output.len >= ascii_string.len);
    for (ascii_string) |char, i| {
        output[i] = toUpper(char);
    }
    return output[0..ascii_string.len];
}

test "upperString" {
    var buf: [1024]u8 = undefined;
    const result = upperString(&buf, "aBcDeFgHiJkLmNOPqrst0234+💩!");
    try testing.expectEqualStrings("ABCDEFGHIJKLMNOPQRST0234+💩!", result);
}

/// Allocates an upper case copy of `ascii_string`.
/// Caller owns returned string and must free with `allocator`.
pub fn allocUpperString(allocator: mem.Allocator, ascii_string: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, ascii_string.len);
    return upperString(result, ascii_string);
}

test "allocUpperString" {
    const result = try allocUpperString(testing.allocator, "aBcDeFgHiJkLmNOPqrst0234+💩!");
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("ABCDEFGHIJKLMNOPQRST0234+💩!", result);
}

/// Compares strings `a` and `b` case-insensitively and returns whether they are equal.
pub fn eqlInsensitive(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a) |a_char, i| {
        if (toLower(a_char) != toLower(b[i])) return false;
    }
    return true;
}

test "eqlInsensitive" {
    try testing.expect(eqlInsensitive("HEl💩Lo!", "hel💩lo!"));
    try testing.expect(!eqlInsensitive("hElLo!", "hello! "));
    try testing.expect(!eqlInsensitive("hElLo!", "helro!"));
}

pub fn startsWithInsensitive(haystack: []const u8, needle: []const u8) bool {
    return if (needle.len > haystack.len) false else eqlInsensitive(haystack[0..needle.len], needle);
}

test "ascii.startsWithInsensitive" {
    try testing.expect(startsWithInsensitive("boB", "Bo"));
    try testing.expect(!startsWithInsensitive("Needle in hAyStAcK", "haystack"));
}

pub fn endsWithInsensitive(haystack: []const u8, needle: []const u8) bool {
    return if (needle.len > haystack.len) false else eqlInsensitive(haystack[haystack.len - needle.len ..], needle);
}

test "ascii.endsWithInsensitive" {
    try testing.expect(endsWithInsensitive("Needle in HaYsTaCk", "haystack"));
    try testing.expect(!endsWithInsensitive("BoB", "Bo"));
}

/// Finds `substr` in `container`, ignoring case, starting at `start_index`.
/// TODO boyer-moore algorithm
pub fn indexOfInsensitivePos(container: []const u8, start_index: usize, substr: []const u8) ?usize {
    if (substr.len > container.len) return null;

    var i: usize = start_index;
    const end = container.len - substr.len;
    while (i <= end) : (i += 1) {
        if (eqlInsensitive(container[i .. i + substr.len], substr)) return i;
    }
    return null;
}

/// Finds `substr` in `container`, ignoring case, starting at index 0.
pub fn indexOfInsensitive(container: []const u8, substr: []const u8) ?usize {
    return indexOfInsensitivePos(container, 0, substr);
}

test "indexOfInsensitive" {
    try testing.expect(indexOfInsensitive("one Two Three Four", "foUr").? == 14);
    try testing.expect(indexOfInsensitive("one two three FouR", "gOur") == null);
    try testing.expect(indexOfInsensitive("foO", "Foo").? == 0);
    try testing.expect(indexOfInsensitive("foo", "fool") == null);
    try testing.expect(indexOfInsensitive("FOO foo", "fOo").? == 0);
}

/// Compares two slices of numbers lexicographically. O(n).
pub fn orderInsensitive(lhs: []const u8, rhs: []const u8) std.math.Order {
    const n = std.math.min(lhs.len, rhs.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        switch (std.math.order(toLower(lhs[i]), toLower(rhs[i]))) {
            .eq => continue,
            .lt => return .lt,
            .gt => return .gt,
        }
    }
    return std.math.order(lhs.len, rhs.len);
}

/// Returns whether lhs < rhs.
pub fn lessThanInsensitive(lhs: []const u8, rhs: []const u8) bool {
    return orderInsensitive(lhs, rhs) == .lt;
}

// TODO: remove everything below this line after 0.10.0

/// DEPRECATED: use `isAlphanumeric`
pub const isAlNum = isAlphanumeric;
/// DEPRECATED: use `eqlInsensitive`
pub const eqlIgnoreCase = eqlInsensitive;
/// DEPRECATED: use `isPrintable`
pub const isPrint = isPrintable;
/// DEPRECATED: use `startsWithInsensitive`
pub const startsWithIgnoreCase = startsWithInsensitive;
/// DEPRECATED: use `lessThanInsensitive`
pub const lessThanIgnoreCase = lessThanInsensitive;
/// DEPRECATED: use `orderInsensitive`
pub const orderIgnoreCase = orderInsensitive;
/// DEPRECATED: use `indexOfInsensitive`
pub const indexOfIgnoreCase = indexOfInsensitive;
/// DEPRECATED: use `indexOfInsensitivePos`
pub const indexOfIgnoreCasePos = indexOfInsensitivePos;
/// DEPRECATED: use `endsWithInsensitive`
pub const endsWithIgnoreCase = endsWithInsensitive;
/// DEPRECATED: use `control`
pub const control_code = control;
/// DEPRECATED: use `char == ' ' or char == '\t'`
fn isBlank(char: u8) bool {
    return char == ' ' or char == '\t';
}
/// DEPRECATED: use `isHexadecimal`
pub const isXDigit = isHexadecimal;
/// DEPRECATED: use `isControl`
pub const isCntrl = isControl;
/// DEPRECATED: use `isAlphabetic`
pub const isAlpha = isAlphabetic;
/// DEPRECATED: use `isWhitespace`
pub const isSpace = isWhitespace;
/// DEPRECATED: use `whitespace`
pub const spaces = whitespace;
pub const isPunct = @compileError("removed: write your own function suited to your particular case");
/// DEPRECATED: use `isPrintable(char) and char != ' '`
fn isGraph(char: u8) bool {
    return isPrintable(char) and char != ' ';
}
