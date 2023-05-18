fn hello() void {
    for ("hello") |c| {
        _ = c;
    }
}

fn counter() void {
    for (0..5) |i| {
        // TODO: maybe at some point this will need @declareSideEffect()
        _ = i;
    }
}

export fn entry() void {
    _ = hello;
    _ = counter;
}

// air
// backend=stage2
// target=native
//
// hello:
// ```
// %3 = const_ty(*const [5:0]u8)
// %4 = constant(*const [5:0]u8, "hello")
// %5 = constant(usize, 5)
// %10 = const_ty(u64)
// %12 = const_ty(u64)
// %13 = constant(u64, 5)
// %15 = const_ty(*const u8)
//
// %1 = alloc(*usize)
// %2!= store(%1, @src.Zir.Inst.Ref.zero_usize)
// %6!= block(void, {
//   %7!= loop(noreturn, {
//     %8 = load(usize, %1)
//     %9!= block(void, {
//       %11 = bitcast(u64, %8)
//       %14 = cmp_lt(%11!, %13)
//       %22!= cond_br(%14!, {
//         %16!= ptr_elem_ptr(*const u8, %4, %8)
//         %17!= load(u8, %16)
//         %20!= br(%9, @src.Zir.Inst.Ref.void_value)
//       }, {
//         %1! %8!
//         %21!= br(%6, @src.Zir.Inst.Ref.void_value)
//       })
//     })
//     %23 = add(%8!, @src.Zir.Inst.Ref.one_usize)
//     %24!= store(%1, %23!)
//   })
// } %1!)
// %26!= ret(@src.Zir.Inst.Ref.void_value)
// ```
// counter:
// ```
// %3 = constant(comptime_int, 5)
// %4 = constant(usize, 5)
// %9 = const_ty(u64)
// %11 = const_ty(u64)
// %12 = constant(u64, 5)
//
// %1 = alloc(*usize)
// %2!= store(%1, @src.Zir.Inst.Ref.zero_usize)
// %5!= block(void, {
//   %6!= loop(noreturn, {
//     %7 = load(usize, %1)
//     %8!= block(void, {
//       %10 = bitcast(u64, %7)
//       %13 = cmp_lt(%10!, %12)
//       %18!= cond_br(%13!, {
//         %16!= br(%8, @src.Zir.Inst.Ref.void_value)
//       }, {
//         %7! %1!
//         %17!= br(%5, @src.Zir.Inst.Ref.void_value)
//       })
//     })
//     %19 = add(%7!, @src.Zir.Inst.Ref.one_usize)
//     %20!= store(%1, %19!)
//   })
// } %1!)
// %22!= ret(@src.Zir.Inst.Ref.void_value)
// ```
