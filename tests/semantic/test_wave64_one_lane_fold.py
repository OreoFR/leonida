"""--wave64-emulate: a region that at most one lane of the wave runs (EXEC narrowed by `v_cmpx_eq_u32 c, v0`, v0
being the thread id) folds lane-locally with several stores, and its v_readfirstlane_b32 takes the lane's own value.

Hand-written programs, 128 threads (two waves), on the workgroup interpreter of test_wave64_emulate_opt.py in both
invocation orders, compared with the plain round emulation (no fold); the constant picks a lane of wave 0, of wave 1,
or none. The negatives must not take the one-lane path and must still match.

Run from the repository root:  python -I tests/semantic/test_wave64_one_lane_fold.py
"""
import contextlib
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_wave64_emulate_opt as base  # noqa: E402

opt = base.wave64_emulate_opt
PUSH = base.VSHARP + [0] * 28
OUT = 0x30000
BIG = ("buffer_", "ds_")

# One lane clears LDS (three writes, the last two to one address), every wave meets at a barrier, the same lane
# reads LDS back, takes first-lane copies of what it read (one into vcc_lo) and stores twice to one address.
PROGRAM = f"""
    s_mov_b64 exec, -1
    v_lshlrev_b32_e32 v9, 2, v0
    v_mov_b32_e32 v4, 7
    v_add_nc_u32_e32 v4, v0, v4
    s_mov_b64 s[40:41], exec
    {{pick}}
    s_mov_b64 vcc, exec
    s_cbranch_execz @A
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v5, 11
    ds_write_b32 v2, v5
    v_mov_b32_e32 v5, 12
    ds_write_b32 v2, v5 offset:4
    v_mov_b32_e32 v5, 13
    ds_write_b32 v2, v5
A:
    s_mov_b64 exec, s[40:41]
    s_barrier
    s_and_b64 exec, vcc, exec
    s_cbranch_execz @B
    v_mov_b32_e32 v2, 0
    ds_read_b32 v6, v2
    ds_read_b32 v7, v2 offset:4
    s_waitcnt lgkmcnt(0)
    v_readfirstlane_b32 s12, v6
    v_readfirstlane_b32 vcc_lo, v7
    v_add_nc_u32_e32 v8, s12, v0
    v_add_nc_u32_e32 v8, vcc_lo, v8
    v_add_nc_u32_e32 v3, {OUT + 512:#x}, v9
    buffer_store_dword v4, v3, s[0:3], 0 offen
    buffer_store_dword v8, v3, s[0:3], 0 offen
B:
    s_mov_b64 exec, -1
    v_add_nc_u32_e32 v9, {OUT:#x}, v9
    buffer_store_dword v4, v9, s[0:3], 0 offen
    s_endpgm
"""


def assemble(text):
    """`(text, nbytes)` list of a listing with `name:` labels and `s_cbranch_x @name` branches."""
    items, labels, pc = [], {}, 0
    for raw in text.strip().splitlines():
        line = raw.strip()
        if line.endswith(":"):
            labels[line[:-1]] = pc
            continue
        n = 8 if line.startswith(BIG) or "0x" in line else 4
        items.append((pc, line, n))
        pc += n
    return [(re.sub(r"@(\w+)", lambda m: str((labels[m.group(1)] - p - n) // 4 & 0xFFFF), line), n)
            for p, line, n in items]


@contextlib.contextmanager
def rounds_only():
    saved = opt.find_local_exec_folds, opt.plan_structured_emulation
    opt.find_local_exec_folds = lambda *a, **k: ({}, {}, set())
    opt.plan_structured_emulation = lambda *a, **k: None
    try:
        yield
    finally:
        opt.find_local_exec_folds, opt.plan_structured_emulation = saved


class OneLaneTests(unittest.TestCase):
    def check(self, pick, size=(128, 1, 1)):
        """Translate and run the program; compare with the round emulation. Returns (folds, memory)."""
        insns = assemble(PROGRAM.replace("{pick}", pick))
        seen = []
        fold = opt.find_local_exec_folds

        def spy(*a, **k):
            seen.append(fold(*a, **k))
            return seen[-1]

        opt.find_local_exec_folds = spy
        try:
            spv, _ = base._translate_encoded("cs_one_lane", insns, size, rsrc2=0x20088)
        finally:
            opt.find_local_exec_folds = fold
        with rounds_only():
            ref, _ = base._translate_encoded("cs_one_lane_ref", insns, size, rsrc2=0x20088)
        want, _ = base.run_emulated_workgroup(ref, size, push_constants=PUSH)
        want = {k: v for k, v in want.items() if k >= OUT}
        for rev in (False, True):
            got, _ = base.run_emulated_workgroup(spv, size, push_constants=PUSH, reverse_order=rev)
            self.assertEqual({k: v for k, v in got.items() if k >= OUT}, want, f"reverse={rev}")
        return seen[-1][0], want

    def test_one_lane_folds_and_matches(self):
        for c in (0, 65, 200):      # a lane of wave 0, of wave 1, no lane
            with self.subTest(c=c):
                folds, mem = self.check(f"v_cmpx_eq_u32_e32 {c}, v0")
                self.assertEqual(sum(fd["one_lane"] for fd in folds.values()), 2)
                self.assertEqual(mem.get(OUT + 512 + 4 * c), 13 + 12 + c if c < 128 else None)
                self.assertEqual(mem[OUT + 4 * 127], 7 + 127)

    def test_one_lane_from_a_compare_mask(self):
        folds, mem = self.check("v_cmp_eq_u32_e64 s[42:43], 3, v0\ns_and_saveexec_b64 s[44:45], s[42:43]")
        self.assertEqual(sum(fd["one_lane"] for fd in folds.values()), 2)
        self.assertEqual(mem[OUT + 512 + 12], 13 + 12 + 3)

    def test_not_one_lane_keeps_the_rounds(self):
        variants = {
            "not the thread id": ("v_cmpx_eq_u32_e32 0, v4", (128, 1, 1)),
            "thread id overwritten": ("v_mov_b32_e32 v0, 0\nv_cmpx_eq_u32_e32 0, v0", (128, 1, 1)),
            "not an equality": ("v_cmpx_gt_u32_e32 2, v0", (128, 1, 1)),
            "rows of 32 threads": ("v_cmpx_eq_u32_e32 0, v0", (32, 4, 1)),
        }
        for name, (pick, size) in variants.items():
            with self.subTest(name):
                folds, _ = self.check(pick, size)
                self.assertFalse(any(fd["one_lane"] for fd in folds.values()))


if __name__ == "__main__":
    unittest.main()
