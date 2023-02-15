//! The PRG (PRoGram) file format.
//!
//! Only supports executable output mode and only the static link mode.
//! This format is not relocatable.
//!
//! Binary content:
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
/// A pool of free block indices that can be reused for new blocks.
free_block_indices: std.ArrayListUnmanaged(u16) = .{},
/// Holds the next block's index.
/// Block index 0 is reserved for the entry point and must not be freed.
block_index: u16 = 1,

/// One declaration can have multiple unnamed constants associated with it.
unnamed_consts_blocks: std.AutoHashMapUnmanaged(Module.Decl.Index, std.ArrayListUnmanaged(Block)) = .{},

/// This holds a list of absolute addresses that we are yet to resolve.
/// These will be resolved only after we know the code of the entire program.
unresolved_addresses: std.ArrayListUnmanaged(UnresolvedAddress) = .{},

// This is memory state we need to preserve across function codegens. (TODO: encapsulate)
abs_offset: ?u16 = null,
zp_free: ?std.BoundedArray(u8, 256) = null,
// TODO: allocate or reserve stack space
/// This provides storage for codgen to store state across function codegens.
//codegen_state: anyopaque, //[512]u8,

pub const base_tag: File.Tag = .prg;

/// A function or a symbol.
const Block = struct {
    /// This is mutable so that we can fix up unresolved addresses and resolve them.
    /// This is nullable because we might record a block's existence but do not have its code yet.
    code: ?[]u8,
    /// All blocks written to the final binary must be written sorted by this index.
    /// If this is 0, it means execution starts at this block's code,
    /// which can be either a function as well as a symbol.
    index: u16,
    /// Whether this block is actually used in the final binary and should be included (to avoid bloat).
    /// TODO: this works around something that the Zig compiler should be able to do; see:
    /// * https://github.com/ziglang/zig/issues/6256
    /// * https://github.com/ziglang/zig/issues/13111
    /// * https://github.com/ziglang/zig/issues/14003
    //used: bool = false, // TODO: actually use this field

    fn deinit(block: *Block, allocator: Allocator) void {
        if (block.code) |code|
            allocator.free(code);
    }
};

/// An unresolved absolute memory address.
pub const UnresolvedAddress = struct {
    /// The declaration this unresolved address is within.
    decl_index: Module.Decl.Index,
    /// Where in this block's code the resolved address is to be written.
    code_offset: u16,
    /// The index of the block whose address should be resolved.
    block_index: u16,
    /// This is to be added to the absolute memory address once it is resolved.
    /// This might represent an index or an offset into the data.
    addend: u16 = 0,
};

pub fn createEmpty(allocator: Allocator, options: link.Options) !*Prg {
    log.debug("createEmpty...", .{});

    assert(options.target.ofmt == .prg);

    if (options.use_llvm) return error.LLVMHasNoPrgBackend;
    if (options.use_lld) return error.LLDHasNoPrgBackend;
    switch (options.target.cpu.arch) {
        .@"6502" => {},
        else => return error.UnsupportedCpuArchitecture,
    }

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
        // If this was true, the file would be truncated even if compilation failed,
        // possibly truncating a previous compilation result,
        // so we want to truncate only when writing the binary, which happens in FileFlush.flush().
        .truncate = false,
        .mode = link.determineMode(options),
    });
    errdefer file.close();
    prg.base.file = file;

    return prg;
}

pub fn deinit(prg: *Prg) void {
    for (prg.blocks.values()) |*block|
        block.deinit(prg.base.allocator);
    prg.blocks.deinit(prg.base.allocator);
    prg.base.allocator.free(prg.header);
    var it = prg.unnamed_consts_blocks.iterator();
    while (it.next()) |blocks| {
        for (blocks.value_ptr.items) |*block|
            block.deinit(prg.base.allocator);
        blocks.value_ptr.deinit(prg.base.allocator);
    }
    prg.unnamed_consts_blocks.deinit(prg.base.allocator);
    prg.free_block_indices.deinit(prg.base.allocator);
    prg.unresolved_addresses.deinit(prg.base.allocator);
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
    var code = switch (res) {
        .ok => try buf.toOwnedSlice(),
        .fail => |error_message| {
            decl.analysis = .codegen_failure;
            try module.failed_decls.putNoClobber(module.gpa, decl_index, error_message);
            return error.AnalysisFail;
        },
    };

    const blk_i = try prg.allocBlockIndex();
    const unnamed_const = Block{
        .code = code,
        .index = blk_i,
    };

    const gop = try prg.unnamed_consts_blocks.getOrPut(prg.base.allocator, decl_index);
    if (!gop.found_existing) {
        gop.value_ptr.* = .{};
    }
    const unnamed_consts = gop.value_ptr;
    try unnamed_consts.append(prg.base.allocator, unnamed_const);

    return blk_i;
}

/// Allocates a new block index assignable to a new block.
fn allocBlockIndex(prg: *Prg) !u16 {
    if (prg.free_block_indices.popOrNull()) |blk_i|
        return blk_i;
    const blk_i = prg.block_index;
    prg.block_index += 1;
    return blk_i;
}

/// Frees a block index and allows it to be reused.
fn freeBlockIndex(prg: *Prg, index: u16) !void {
    assert(index != 0); // This block index existed initially and must not be freed.
    try prg.free_block_indices.append(prg.base.allocator, index);
}

pub fn flush(prg: *Prg, comp: *Compilation, prog_node: *std.Progress.Node) !void {
    log.debug("flush...", .{});
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

// TODO: like the ELF linker and others, instead of storing the binary's code in buffers
//       and then writing them all at the end using writev,
//       consider always immediately writing the buffer to the binary file at a certain offset using pwrite
//       and extend the file as we go
/// Facilitates efficient writing of multiple buffers to a file.
const FileFlush = struct {
    buffers: std.ArrayListUnmanaged(std.os.iovec_const) = .{},
    /// The exact amount of bytes that will be written to the file upon flush().
    file_size: u64 = 0,

    fn ensureUnusedCapacity(file_flush: *FileFlush, allocator: Allocator, additional_count: usize) !void {
        try file_flush.buffers.ensureUnusedCapacity(allocator, additional_count);
    }

    fn appendBufAssumeCapacity(file_flush: *FileFlush, buf: []const u8) void {
        file_flush.buffers.appendAssumeCapacity(.{ .iov_base = buf.ptr, .iov_len = buf.len });
        file_flush.file_size += buf.len;
    }

    fn flush(file_flush: FileFlush, file: std.fs.File) !void {
        // This truncates the file's size to exactly the size that we will write.
        // This basically serves as a preallocation hint for the file system
        // for the following write. TODO: confirm this and confirm doing this is actually faster.
        // This does not alter the file offset.
        try file.setEndPos(file_flush.file_size);
        // Now write all buffers to the file at the start.
        // Under normal conditions the file offset should still be 0,
        // so we can use writev instead of pwritev.
        assert((file.getPos() catch unreachable) == 0);
        try file.writevAll(file_flush.buffers.items);
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
        const blk_i = try prg.allocBlockIndex();
        gop.value_ptr.* = .{
            .code = null,
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
        // TODO: actually test the 6502-freestanding target
        .freestanding => 0x0200, // The address after the zero page and the stack.
        else => unreachable,
    };
}

fn writeHeader(prg: Prg) ![]const u8 {
    // Reference: https://github.com/llvm-mos/llvm-mos/blob/ca75934d2b4c29cc9cea38629d50b6c2f42d9016/lld/test/ELF/Inputs/mos-c64.inc
    // Reference: https://github.com/cc65/cc65/blob/60f56c43c769f39089f5005f736a06aacd393224/libsrc/cbm/exehdr.s
    // Reference: https://retrogramming.blogspot.com/2022/02/c64-create-prg-file-with-simple-machine.html
    var bytes = std.ArrayListUnmanaged(u8){};
    const writer = bytes.writer(prg.base.allocator);
    switch (prg.base.options.target.os.tag) {
        .c64 => {
            const load_address = prg.getLoadAddress();
            try writer.writeIntLittle(u16, load_address);

            // TODO: -fno-basic-bootstrap or `-fbasic-bootstrap=false` etc. to allow omitting this entirely optional code
            // TODO: allow embedding custom BASIC commands? with `asm`?
            //       shouldn't it be possible if I use `fno-basic-bootstrap` and then
            //       write the bootstrap code in comptime top-level asm at the top of the file?
            // The following is minimal BASIC bootstrap code that allows us to run our program
            // using the RUN command, right after loading it, making our program distributable.

            // Pointer to the next BASIC line.
            try writer.writeIntLittle(
                u16,
                load_address +
                    10, // Size of what we write after this.
            );

            // The following encodes the BASIC line "0 SYS 2061".
            {
                // Line marker.
                try writer.writeIntLittle(u16, 0x0000);

                // SYS command token code.
                try writer.writeByte(0x9E);

                // No space required.

                // Execution address.
                try writer.writeAll("2061\x00");
            }

            // In our case this marks the end of our linked list.
            try writer.writeIntLittle(u16, 0x0000);
        },
        // NOTE: this is like -ofmt=raw...
        .freestanding => {},
        else => unreachable,
    }
    return try bytes.toOwnedSlice(prg.base.allocator);
}

const DeclIndexAndBlock = struct { decl_index: Module.Decl.Index, block: Block };

/// Returns all recorded blocks in block index order, with the entry point block first.
/// This is necessary in order to know where exactly a block will end up in the final binary.
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
    // We sort by block index to make sure we output and iterate the blocks in the same order every time.
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

    log.debug("flush module...", .{});

    try prg.verifyBuildOptions();

    var sub_prog_node = prog_node.start("Flush Module", 0);
    sub_prog_node.activate();
    defer sub_prog_node.end();

    const file = prg.base.file.?;

    var file_flush = FileFlush{};
    defer file_flush.deinit(prg.base.allocator);

    try file_flush.ensureUnusedCapacity(prg.base.allocator, 1);
    file_flush.appendBufAssumeCapacity(prg.header);

    // prg.blocks does not have all blocks that need to be written so
    // we need to combine prg.blocks and prg.unnamed_consts_blocks into one but also sort them.
    const decl_index_and_blocks = try prg.getAllBlocks(prg.base.allocator);
    defer prg.base.allocator.free(decl_index_and_blocks);

    // Resolve all unresolved absolute memory addresses.
    for (prg.unresolved_addresses.items) |unresolved_address| {
        log.debug("resolving {}...", .{unresolved_address});

        // Figure out the address of `unresolved_address.block_index`.
        var offset: u16 = 0;
        const blk_addr = (for (decl_index_and_blocks) |decl_index_and_block| {
            const block = decl_index_and_block.block;
            const code = block.code.?;
            assert(code.len != 0);
            if (unresolved_address.block_index == block.index) {
                const load_address = prg.getLoadAddress();
                break load_address + @intCast(u16, prg.header.len) - @sizeOf(@TypeOf(load_address)) + offset;
            }
            offset += @intCast(u16, code.len);
        } else unreachable) + unresolved_address.addend;

        var owner_block = for (decl_index_and_blocks) |decl_index_and_block| {
            if (decl_index_and_block.decl_index == unresolved_address.decl_index)
                break decl_index_and_block.block;
        } else unreachable;

        log.debug("resolved address {} as 0x{X:0>4}", .{ unresolved_address, blk_addr });

        mem.writeIntLittle(u16, @ptrCast(*[2]u8, owner_block.code.?[unresolved_address.code_offset..][0..2]), blk_addr);
    }

    // Write all functions and symbols.
    // The entry point block's code will be first because its block index is 0 so we will start execution there.
    var offset: u16 = 0;
    try file_flush.ensureUnusedCapacity(prg.base.allocator, decl_index_and_blocks.len);
    for (decl_index_and_blocks) |decl_index_and_block| {
        const block = decl_index_and_block.block;
        const code = block.code.?;
        assert(code.len != 0);
        const load_address = prg.getLoadAddress();
        const addr = load_address + @intCast(u16, prg.header.len) - @sizeOf(@TypeOf(load_address)) + offset;
        log.debug("writing block {} of address 0x{X:0>4}", .{ decl_index_and_block.block, addr });
        file_flush.appendBufAssumeCapacity(code);
        offset += @intCast(u16, code.len);
    }

    try file_flush.flush(file);
}

pub fn freeDecl(prg: *Prg, decl_index: Module.Decl.Index) void {
    log.debug("freeDecl for {}...", .{decl_index});
    if (prg.blocks.fetchSwapRemove(decl_index)) |kv| {
        var block = kv.value;
        block.deinit(prg.base.allocator);
        prg.freeBlockIndex(block.index) catch {
            // TODO: propose making freeDecl errorable. Plan9 is facing this exact problem in their freeDecl too
            @panic("OOM");
        };
    }
}

/// Generates a declaration's code.
pub fn updateDecl(prg: *Prg, module: *Module, decl_index: Module.Decl.Index) !void {
    const decl = module.declPtr(decl_index);

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
    var code = switch (res) {
        .ok => try buf.toOwnedSlice(),
        .fail => |error_message| {
            decl.analysis = .codegen_failure;
            try module.failed_decls.putNoClobber(module.gpa, decl_index, error_message);
            return error.AnalysisFail;
        },
    };

    const gop = try prg.blocks.getOrPut(prg.base.allocator, decl_index);
    if (gop.found_existing) {
        const block = gop.value_ptr;
        if (block.code) |old_code|
            prg.base.allocator.free(old_code);
        block.code = code;
    } else {
        const block = .{
            .code = code,
            .index = try prg.allocBlockIndex(),
        };
        gop.value_ptr.* = block;
    }
}

/// Generates a function's code.
pub fn updateFunc(prg: *Prg, module: *Module, func: *Module.Fn, air: Air, liveness: Liveness) !void {
    if (build_options.skip_non_native and builtin.object_format != .prg) {
        @panic("attempted to compile for object format that was disabled by build configuration");
    }

    const decl_index = func.owner_decl;
    const decl = module.declPtr(decl_index);

    // This function is being updated so we need to redo this as well.
    if (prg.unnamed_consts_blocks.getPtr(decl_index)) |unnamed_consts| {
        for (unnamed_consts.items) |*unnamed_const|
            unnamed_const.deinit(prg.base.allocator);
        unnamed_consts.clearAndFree(prg.base.allocator);
    }

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
        .ok => try buf.toOwnedSlice(),
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
        const blk_i = try prg.allocBlockIndex();
        const block = .{
            .code = code,
            .index = blk_i,
        };
        gop.value_ptr.* = block;
    }
}

pub fn getDeclVAddr(prg: *Prg, decl_index: Module.Decl.Index, reloc_info: link.File.RelocInfo) !u64 {
    // TODO
    _ = prg;
    debug.panic("do we need `getDeclVAddr`? {}, {}", .{ decl_index, reloc_info });
}

pub fn updateDeclExports(
    prg: *Prg,
    module: *Module,
    decl_index: Module.Decl.Index,
    exports: []const *Module.Export,
) !void {
    // TODO: do we use prg.base.options.entry_addr for anything?

    // TODO: why "_start"? we could use a prettier name more conventional in the Commodore community
    //       (entry, main, start?). we are free to choose any name we like.
    const entry_name = prg.base.options.entry orelse "_start";

    log.debug("updateDeclExports for {}...", .{decl_index});
    for (exports) |@"export"| {
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
                const block = gop.value_ptr;
                try prg.freeBlockIndex(block.index);
                block.index = 0;
            } else {
                const block = .{
                    .code = null,
                    .index = 0,
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
    // We do not provide debug info.
    _ = prg;
    _ = mod;
    _ = decl_index;
}
