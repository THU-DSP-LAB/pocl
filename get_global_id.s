        .text
        .attribute      4, 16
        .attribute      5, "rv32i2p0_m2p0_a2p0_zfinx1p0_zdinx1p0_zve32f1p0_zve32x1p0_zvl32b1p0_zhinx1p0"
        .file   "get_global_id.cl"
        .globl  _Z13get_global_idj
        .p2align        2
        .type   _Z13get_global_idj,@function
_Z13get_global_idj:
        csrr t0,0x800
        vid.v v0
        vadd.vx v0,v0,t0
        ret
.Lfunc_end0:
        .size   _Z13get_global_idj, .Lfunc_end0-_Z13get_global_idj

        .ident  "clang version 16.0.0 (git@git.tpt.com:/git/ventus-llvm.git 3fdda4cd8e5725c61febae6fa448e4ba549486f9)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig