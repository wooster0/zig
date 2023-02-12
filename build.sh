#!/bin/sh

zig build -p stage3 \
    --zig-lib-dir lib \
    -Denable-llvm=false \
    -Dlog -freference-trace 
