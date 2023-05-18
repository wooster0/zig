export fn foo(x: [*]u8, l: usize) void {
    @memset(x[0..l], undefined);
}

// air
// backend=stage2
// target=native
//
// foo:
// ```
// %0 = const_ty([*]u8)
// %5 = const_ty([*]u8)
// %7 = const_ty(*const [*]u8)
// %9 = const_ty([*]u8)
// %11 = constant(usize, 0)
// %12 = const_ty([*]u8)
// %14 = const_ty([]u8)
// %16 = constant(u8, undefined)
//
// %1 = arg([*]u8, 0)
// %2 = arg(usize, 1)
// %4 = alloc(*[*]u8)
// %6!= store(%4, %1!)
// %8 = bitcast(*const [*]u8, %4!)
// %10 = load([*]u8, %8!)
// %13 = ptr_add([*]u8, %10!, %11)
// %15 = slice([]u8, %13!, %2!)
// %17!= memset(%15!, %16)
// %19!= ret(@src.Zir.Inst.Ref.void_value)
// ```
