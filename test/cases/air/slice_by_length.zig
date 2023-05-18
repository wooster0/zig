fn foo(s: []const i32, start: usize, len: usize) []const i32 {
    return s[start..][0..len];
}

export fn entry() void {
    _ = foo;
}

// air
// backend=stage2
// target=native
//
// foo:
// ```
// %0 = const_ty([]const i32)
// %5 = const_ty([]const i32)
// %7 = const_ty([]const i32)
// %9 = const_ty(*const []const i32)
// %11 = const_ty([]const i32)
// %13 = const_ty([*]const i32)
// %15 = const_ty([*]const i32)
// %18 = const_ty([]const i32)
// 
// %1 = arg([]const i32, 0)
// %2 = arg(usize, 1)
// %3 = arg(usize, 2)
// %6 = alloc(*[]const i32)
// %8!= store(%6, %1!)
// %10 = bitcast(*const []const i32, %6!)
// %12 = load([]const i32, %10!)
// %14 = slice_ptr([*]const i32, %12!)
// %16 = ptr_add([*]const i32, %14!, %2!)
// %17!= add(%2, %3)
// %19 = slice([]const i32, %16!, %3!)
// %20!= ret(%19!)
// ```
