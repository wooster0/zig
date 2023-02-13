#!/bin/sh

file_name=hello

stage3/bin/zig build-exe $file_name.zig \
    -ofmt=prg \
    -target 6502-c64 \
    --zig-lib-dir lib \
    -fno-LLVM \
    --verbose-air \
    --debug-log codegen \
    --debug-log emit \
    --debug-log link \
    --debug-log default \
    --verbose-link \
    `# TODO: make sure it works well without ReleaseSmall` \
    -O ReleaseSmall \
    2>&1 | less &&
hexdump $file_name.prg -C -v &&
da65 $file_name.prg
