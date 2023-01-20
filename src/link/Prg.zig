//! The PRG (PRoGram) file format.
//!
//! Only supports executable output mode and only the static link mode.
//! This format is not relocatable.
//!
//! File structure:
//! 1. Little-endian 2-byte load address
//! 2. BASIC bootstrap code that makes the RUN command work
//! 3. Entry point function
//! 4. All other functions, and symbols
//!
//! This format is used by the Commodore 64, Commodore 16, Commodore 128, and others.
//!
//! In the Commodore retrocomputing space there are a few file formats in use:
//! * .T64: most .T64 files are .PRG files with an added header.
//! * .TAP: tape image for making emulator pretend there is a tape. Contains .PRG data.
//! * .D64: disk image for making emulator pretend there is a disk in a Commodore 1541. Contains .PRG data.
//! * .PRG: base file format.
//!
//! PRG is generally the most useful one.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const mem = std.mem;
const debug = std.debug;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.link);
const assert = std.debug.assert;
const link = @import("../link.zig");
const Module = @import("../Module.zig");
const Compilation = @import("../Compilation.zig");
const codegen = @import("../codegen.zig");
const trace = @import("../tracy.zig").trace;
const File = link.File;
const Air = @import("../Air.zig");
const Liveness = @import("../Liveness.zig");
const TypedValue = @import("../TypedValue.zig");

const Prg = @This();

base: link.File,
error_flags: File.ErrorFlags = File.ErrorFlags{},

header: []const u8,

blocks: std.AutoArrayHashMapUnmanaged(Module.Decl.Index, Block) = .{},
free_block_indices: std.ArrayListUnmanaged(u16) = .{},
block_index: u16 = 0,

/// One declaration can have multiple unnamed constants associated with it.
unnamed_consts_blocks: std.AutoHashMapUnmanaged(Module.Decl.Index, std.ArrayListUnmanaged(Block)) = .{},

abs_offset: ?u16 = null,
zp_free: ?std.BoundedArray(u8, 256) = null,

pub const base_tag: File.Tag = .prg;

/// A function or a symbol.
const Block = struct {
    /// This is nullable because we may allocate this block in
    /// `allocateDeclIndexes` without any code and free it in `freeDecl`
    /// without ever generating code for it.
    code: ?[]const u8,
    /// All blocks written to the final binary must be written sorted by this index.
    index: u16,
    /// This can technically be true for a symbol, in which case execution would start at the symbol's code.
    entry_point: bool,
    /// Whether this block is actually used in the final binary and should be included (to avoid bloat).
    /// TODO: this works around something that the Zig compiler should be able to do; see:
    /// * https://github.com/ziglang/zig/issues/6256
    /// * https://github.com/ziglang/zig/issues/13111
    /// * https://github.com/ziglang/zig/issues/14003
    used: bool = false, // TODO: actually use this field

    const Type = enum { function, symbol };

    fn deinit(block: *Block, allocator: Allocator, prg: *Prg) void {
        if (block.code) |code|
            allocator.free(code);
        prg.free_block_indices.append(allocator, block.index) catch {
            // TODO
            @panic("oom probably");
        };
    }
};

pub fn createEmpty(allocator: Allocator, options: link.Options) !*Prg {
    log.debug("createEmpty...", .{});

    assert(options.target.ofmt == .prg);

    if (options.use_llvm) return error.LLVMHasNoPrgBackend;
    if (options.use_lld) return error.LLDHasNoPrgBackend;

    const prg = try allocator.create(Prg);
    errdefer allocator.destroy(prg);
    prg.* = .{
        .base = .{
            .tag = .prg,
            .options = options,
            .allocator = allocator,
            .file = null,
        },
        .header = undefined,
    };
    prg.header = try prg.writeHeader();

    return prg;
}

pub fn openPath(allocator: Allocator, sub_path: []const u8, options: link.Options) !*Prg {
    log.debug("openPath...", .{});

    const prg = try createEmpty(allocator, options);

    const file = try options.emit.?.directory.handle.createFile(sub_path, .{
        .read = true,
        .mode = link.determineMode(options),
    });
    errdefer file.close();
    prg.base.file = file;

    return prg;
}

pub fn deinit(prg: *Prg) void {
    for (prg.blocks.values()) |*block| {
        block.deinit(prg.base.allocator, prg);
    }
    prg.blocks.deinit(prg.base.allocator);
    prg.base.allocator.free(prg.header);
    var it = prg.unnamed_consts_blocks.iterator();
    while (it.next()) |blocks| {
        for (blocks.value_ptr.items) |*block|
            block.deinit(prg.base.allocator, prg);
        blocks.value_ptr.deinit(prg.base.allocator);
    }
    prg.unnamed_consts_blocks.deinit(prg.base.allocator);
    prg.free_block_indices.deinit(prg.base.allocator);
}

/// Returns the block index of the lowered unnamed constant, as a `u32` but should be casted to `u16`.
/// `decl_index` is the function that this unnamed constant belongs to.
pub fn lowerUnnamedConst(prg: *Prg, tv: TypedValue, decl_index: Module.Decl.Index) !u32 {
    const module = prg.base.options.module.?;
    const decl = module.declPtr(decl_index);

    var buf = std.ArrayList(u8).init(prg.base.allocator);
    const res = try codegen.generateSymbol(
        &prg.base,
        decl.srcLoc(),
        tv,
        &buf,
        .{ .none = {} },
        .{ .parent_atom_index = @enumToInt(decl_index) },
    );
    const code = switch (res) {
        .externally_managed => |external_value| external_value,
        .appended => buf.items,
        .fail => |error_message| {
            decl.analysis = .codegen_failure;
            try module.failed_decls.putNoClobber(module.gpa, decl_index, error_message);
            return error.AnalysisFail;
        },
    };

    const blk_i = prg.getNewBlockIndex();
    const unnamed_const = Block{
        .code = code,
        .index = blk_i,
        .entry_point = false,
    };

    const gop = try prg.unnamed_consts_blocks.getOrPut(prg.base.allocator, decl_index);
    if (!gop.found_existing) {
        gop.value_ptr.* = .{};
    }
    const unnamed_consts = gop.value_ptr;
    try unnamed_consts.append(prg.base.allocator, unnamed_const);

    return blk_i;
}

fn getNewBlockIndex(prg: *Prg) u16 {
    if (prg.free_block_indices.popOrNull()) |blk_i|
        return blk_i;
    const blk_i = prg.block_index;
    prg.block_index += 1;
    return blk_i;
}

pub fn flush(prg: *Prg, comp: *Compilation, prog_node: *std.Progress.Node) !void {
    return prg.flushModule(comp, prog_node);
}

fn verifyBuildOptions(prg: Prg) !void {
    switch (prg.base.options.link_mode) {
        .Static => {},
        .Dynamic => return error.InvalidLinkMode,
    }
    switch (prg.base.options.effectiveOutputMode()) {
        .Exe => {},
        .Obj, .Lib => return error.InvalidOutputMode,
    }
}

/// Facilitates efficient writing of multiple buffers to a file.
const FileFlush = struct {
    buffers: std.ArrayListUnmanaged(std.os.iovec_const) = .{},
    file_size: u64 = 0,

    fn ensureUnusedCapacity(file_flush: *FileFlush, allocator: Allocator, additional_count: usize) !void {
        try file_flush.buffers.ensureUnusedCapacity(allocator, additional_count);
    }

    fn appendBufAssumeCapacity(file_flush: *FileFlush, buf: []const u8) void {
        file_flush.buffers.appendAssumeCapacity(.{ .iov_base = buf.ptr, .iov_len = buf.len });
    }

    fn flush(file_flush: FileFlush, file: std.fs.File) !void {
        try file.setEndPos(file_flush.file_size);
        try file.pwritevAll(file_flush.buffers.items, 0);
    }

    fn deinit(file_flush: *FileFlush, allocator: Allocator) void {
        file_flush.buffers.deinit(allocator);
    }
};

pub fn recordDecl(prg: *Prg, decl_index: Module.Decl.Index) !u16 {
    const gop = try prg.blocks.getOrPut(prg.base.allocator, decl_index);
    if (gop.found_existing) {
        return gop.value_ptr.index;
    } else {
        const blk_i = prg.getNewBlockIndex();
        gop.value_ptr.* = .{
            .code = null,
            .entry_point = false,
            .index = blk_i,
        };
        return blk_i;
    }
}

/// Returns the lowest load address tailored to the target operating system,
/// specifying where the program will be loaded to.
pub fn getLoadAddress(prg: Prg) u16 {
    return switch (prg.base.options.target.os.tag) {
        .c64 => 0x0801,
        .freestanding => 0x0200, // after the zero page and the stack
        else => unreachable,
    };
}

fn writeHeader(prg: Prg) ![]const u8 {
    // ref: https://github.com/llvm-mos/llvm-mos/blob/ca75934d2b4c29cc9cea38629d50b6c2f42d9016/lld/test/ELF/Inputs/mos-c64.inc
    // ref: https://github.com/cc65/cc65/blob/60f56c43c769f39089f5005f736a06aacd393224/libsrc/cbm/exehdr.s
    // ref: https://retrogramming.blogspot.com/2022/02/c64-create-prg-file-with-simple-machine.html
    var bytes = std.ArrayListUnmanaged(u8){};
    const writer = bytes.writer(prg.base.allocator);
    switch (prg.base.options.target.os.tag) {
        .c64 => {
            const load_address = prg.getLoadAddress();
            try writer.writeIntLittle(u16, load_address);

            // TODO: -fno-basic-bootstrap to allow omitting this code
            // TODO: allow embedding custom BASIC commands? with `asm`?
            // the following is minimal BASIC bootstrap code that allows us to run our program
            // using the RUN command, right after loading it, making our program distributable

            // pointer to the next BASIC line
            try writer.writeIntLittle(
                u16,
                load_address +
                    10, // size of what we write after this
            );

            // the following encodes the BASIC line "0 SYS 2061"
            {
                // line marker
                try writer.writeIntLittle(u16, 0x0000);

                // SYS command token code
                try writer.writeByte(0x9E);

                // no space required

                // execution address
                try writer.writeAll("2061\x00");
            }

            // in our case this marks the end of our linked list
            try writer.writeIntLittle(u16, 0x0000);
        },
        // NOTE: this is like -ofmt=raw...
        .freestanding => {},
        else => unreachable,
    }
    return bytes.toOwnedSlice(prg.base.allocator);
}

const DeclIndexAndBlock = struct { decl_index: Module.Decl.Index, block: Block };

pub fn getAllBlocks(prg: *Prg, allocator: Allocator) ![]const DeclIndexAndBlock {
    var decl_index_and_blocks = std.ArrayList(DeclIndexAndBlock).init(allocator);
    var it1 = prg.blocks.iterator();
    while (it1.next()) |decl_index_and_block| {
        try decl_index_and_blocks.append(.{ .decl_index = decl_index_and_block.key_ptr.*, .block = decl_index_and_block.value_ptr.* });
    }
    var it2 = prg.unnamed_consts_blocks.iterator();
    while (it2.next()) |blocks| {
        for (blocks.value_ptr.items) |block| {
            try decl_index_and_blocks.append(.{ .decl_index = blocks.key_ptr.*, .block = block });
        }
    }
    const slice = try decl_index_and_blocks.toOwnedSlice();
    // we sort to make sure we output and iterate the blocks in the same order every time
    std.sort.sort(DeclIndexAndBlock, slice, {}, struct {
        fn comparator(context: void, lhs: DeclIndexAndBlock, rhs: DeclIndexAndBlock) bool {
            _ = context;
            return lhs.block.index < rhs.block.index;
        }
    }.comparator);
    return slice;
}

pub fn flushModule(prg: *Prg, comp: *Compilation, prog_node: *std.Progress.Node) !void {
    _ = comp;
    if (build_options.skip_non_native and builtin.object_format != .prg) {
        @panic("attempted to compile for object format that was disabled by build configuration");
    }

    try prg.verifyBuildOptions();

    var sub_prog_node = prog_node.start("Flush Module", 0);
    sub_prog_node.activate();
    defer sub_prog_node.end();

    const file = prg.base.file.?;

    var it = prg.blocks.iterator();
    while (it.next()) |entry| {
        log.debug("block: {} -> {}", .{ entry.key_ptr.*, entry.value_ptr.* });
    }

    var file_flush = FileFlush{};
    defer file_flush.deinit(prg.base.allocator);

    try file_flush.ensureUnusedCapacity(prg.base.allocator, 1);
    file_flush.appendBufAssumeCapacity(prg.header);

    // prg.blocks does not have all blocks that need to be written.
    // we need to combine prg.blocks and prg.unnamed_consts_blocks into one but also sort them.
    const decl_index_and_blocks = try prg.getAllBlocks(prg.base.allocator);
    defer prg.base.allocator.free(decl_index_and_blocks);

    // put the entry point function code first so that we start execution there
    const entry_point_function = for (decl_index_and_blocks) |decl_index_and_block| {
        if (decl_index_and_block.block.entry_point)
            break decl_index_and_block.block;
    } else {
        prg.error_flags.no_entry_point_found = true;
        return;
    };
    const entry_point_code = entry_point_function.code.?;
    assert(entry_point_code.len != 0);
    file_flush.appendBufAssumeCapacity(entry_point_code);

    // now come all the other functions and symbols
    try file_flush.ensureUnusedCapacity(prg.base.allocator, decl_index_and_blocks.len);
    for (decl_index_and_blocks) |decl_index_and_block| {
        if (decl_index_and_block.block.entry_point) continue;
        const code = decl_index_and_block.block.code.?;
        assert(code.len != 0);
        file_flush.appendBufAssumeCapacity(code);
    }

    try file_flush.flush(file);
}

pub fn freeDecl(prg: *Prg, decl_index: Module.Decl.Index) void {
    log.debug("freeDecl for {}...", .{decl_index});
    if (prg.blocks.fetchSwapRemove(decl_index)) |kv| {
        var block = kv.value;
        block.deinit(prg.base.allocator, prg);
    }
}

/// Generates a declaration's code.
/// Called after `allocateDeclIndexes`.
pub fn updateDecl(prg: *Prg, module: *Module, decl_index: Module.Decl.Index) !void {
    const decl = module.declPtr(decl_index);

    log.debug("codegen decl {*} ({s}) ({d})", .{ decl, decl.name, decl_index });

    var buf = std.ArrayList(u8).init(prg.base.allocator);
    const decl_val = if (decl.val.castTag(.variable)) |payload|
        payload.data.init
    else
        decl.val;
    const res = try codegen.generateSymbol(
        &prg.base,
        decl.srcLoc(),
        .{ .ty = decl.ty, .val = decl_val },
        &buf,
        .{ .none = {} },
        .{ .parent_atom_index = @enumToInt(decl_index) },
    );
    const code = switch (res) {
        .externally_managed => |external_value| external_value,
        .appended => buf.items,
        .fail => |error_message| {
            decl.analysis = .codegen_failure;
            try module.failed_decls.putNoClobber(module.gpa, decl_index, error_message);
            return error.AnalysisFail;
        },
    };

    defer log.debug("generated symbol code: {any} (block index {})", .{ code, prg.blocks.get(decl_index).?.index });
    const gop = try prg.blocks.getOrPut(prg.base.allocator, decl_index);
    if (gop.found_existing) {
        const block = gop.value_ptr;
        if (block.code) |old_code|
            prg.base.allocator.free(old_code);
        block.code = code;
    } else {
        const block = .{
            .code = code,
            .index = prg.getNewBlockIndex(),
            .entry_point = false,
        };
        gop.value_ptr.* = block;
    }
}

/// Generates a function's code.
/// Called after `allocateDeclIndexes`.
pub fn updateFunc(prg: *Prg, module: *Module, func: *Module.Fn, air: Air, liveness: Liveness) !void {
    if (build_options.skip_non_native and builtin.object_format != .prg) {
        @panic("attempted to compile for object format that was disabled by build configuration");
    }

    const decl_index = func.owner_decl;
    const decl = module.declPtr(decl_index);

    // this function is being updated so we need to redo this as well
    if (prg.unnamed_consts_blocks.getPtr(decl_index)) |unnamed_consts| {
        for (unnamed_consts.items) |*unnamed_const|
            unnamed_const.deinit(prg.base.allocator, prg);
        unnamed_consts.clearAndFree(prg.base.allocator);
    }

    log.debug("updateFunc for {} ({s})...", .{ decl_index, decl.name });

    var buf = std.ArrayList(u8).init(prg.base.allocator);
    const res = try codegen.generateFunction(
        &prg.base,
        decl.srcLoc(),
        func,
        air,
        liveness,
        &buf,
        .{ .none = {} },
    );
    const code = switch (res) {
        .appended => buf.items,
        .fail => |error_message| {
            decl.analysis = .codegen_failure;
            try module.failed_decls.putNoClobber(prg.base.allocator, decl_index, error_message);
            return;
        },
    };

    const gop = try prg.blocks.getOrPut(prg.base.allocator, decl_index);
    if (gop.found_existing) {
        const block = gop.value_ptr;
        if (block.code) |old_code|
            prg.base.allocator.free(old_code);
        block.code = code;
    } else {
        const blk_i = prg.getNewBlockIndex();
        const block = .{
            .code = code,
            .index = blk_i,
            .entry_point = false,
        };
        gop.value_ptr.* = block;
    }
}

pub fn getDeclVAddr(prg: *Prg, decl_index: Module.Decl.Index, reloc_info: link.File.RelocInfo) !u64 {
    _ = prg;
    debug.panic("do we need `getDeclVAddr`? {}, {}", .{ decl_index, reloc_info });
}

/// Always called before `updateDecl` or `updateDeclExports`.
pub fn allocateDeclIndexes(prg: *Prg, decl_index: Module.Decl.Index) !void {
    _ = prg;
    log.debug("allocateDeclIndexes for {}", .{decl_index});
}

/// Called after `allocateDeclIndexes`.
pub fn updateDeclExports(
    prg: *Prg,
    module: *Module,
    decl_index: Module.Decl.Index,
    exports: []const *Module.Export,
) !void {
    // TODO: why "_start"? we could use a prettier name more conventional in the Commodore community
    //       (entry, main, start?). we are free to choose any name we like.
    const entry_name = prg.base.options.entry orelse "_start";

    log.debug("updateDeclExports... for {}, exports: {any}", .{ decl_index, exports });
    for (exports) |@"export"| {
        log.debug("export: {} ({s})", .{ @"export", @"export".options.name });
        const options_are_default =
            @"export".options.visibility == .default and
            @"export".options.section == null and
            @"export".options.linkage == .Strong;

        if (mem.eql(u8, @"export".options.name, entry_name)) {
            if (!options_are_default) {
                try module.failed_exports.putNoClobber(
                    module.gpa,
                    @"export",
                    try Module.ErrorMsg.create(
                        prg.base.allocator,
                        module.declPtr(decl_index).srcLoc(),
                        "\"{s}\" must have default export options",
                        .{entry_name},
                    ),
                );
                return error.AnalysisFail;
            }
            const gop = try prg.blocks.getOrPut(prg.base.allocator, decl_index);
            if (gop.found_existing) {
                log.debug("updateDeclExports: found existing", .{});
                const block = gop.value_ptr;
                block.entry_point = true;
            } else {
                const block = .{
                    .code = null,
                    .index = prg.getNewBlockIndex(),
                    .entry_point = true,
                };
                gop.value_ptr.* = block;
            }
        } else {
            try module.failed_exports.putNoClobber(
                module.gpa,
                @"export",
                try Module.ErrorMsg.create(
                    prg.base.allocator,
                    module.declPtr(decl_index).srcLoc(),
                    "PRG does not support exports other than \"{s}\"",
                    .{entry_name},
                ),
            );
            return error.AnalysisFail;
        }
    }
}

pub fn updateDeclLineNumber(prg: *Prg, mod: *Module, decl_index: Module.Decl.Index) !void {
    // we do not provide debug info
    _ = prg;
    _ = mod;
    _ = decl_index;
}
