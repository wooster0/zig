export fn x(a: bool, b: bool) bool {
    return a and b;
}

export fn y(a: bool, b: bool) bool {
    return a or b;
}

// air
// backend=stage2
// target=native
//
// x:
// ```
// %0 = arg(bool, 0)
// %1 = arg(bool, 1)
// %2 = bool_and(%0!, %1!)
// %3 = ret(%2)
// ```
// y:
// ```
// %0 = arg(bool, 0)
// %1 = arg(bool, 1)
// %2 = bool_or(%0!, %1!)
// %3 = ret(%2)
// ```
