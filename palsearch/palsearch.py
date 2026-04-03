#!/usr/bin/env python3
"""
palsearch.py  —  Python reimplementation of palsearch.ml

Converts a PNG image to BBC Master Mode 1 screen data with per-2-scanline
palette changes, using a greedy palette solver with numpy-accelerated fallback.

Usage:
    python palsearch.py input.png -o output.bin [-d ordered|fs] [-q]

Output binary format (compatible with showimage.s):
    Bytes 0-15        : Initial palette (16 entries)
    Bytes 16-255      : Padding (zeros) — aligns stream table to a 256-byte boundary
    Bytes 256+        : Per-section delta streams:
                          output_palettes[256 + slot*num_sections + (section-1)]
                          for slot in 0..changes_per_row-1
                          for section in 1..num_sections
    Bytes pal_size+   : Screen data (20480 bytes, BBC non-linear interleaved layout)

    The 256-byte alignment ensures that LDA stream_base,X (X=0..num_sections-1)
    never crosses a page boundary on the 6502, avoiding the +1 cycle penalty.

Palette byte encoding: (palette_index << 4) | (bbc_colour ^ 7)
    palette_index : which of the 16 ULA palette entries to update
    bbc_colour    : 0-7 (bit0=R, bit1=G, bit2=B), XOR'd with 7 for ULA negative logic

Mode 1 ULA bit-extraction (lookup_cols):
    For each screen byte, 4 pixel colours are extracted by progressively
    shifting left and ORing 1.  Pixel 0 uses bits {b7,b5,b3,b1} as palette
    index; pixel 2 index is (pixel0_index & 7)*2+1 (forced odd); similarly
    pixel 1 is free, pixel 3 index is (pixel1_index & 7)*2+1.

Dependencies:
    pip install Pillow numpy
"""

import os
import sys
import argparse
import random
from collections import Counter

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed.  Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import z3 as _z3_mod
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNKSIZE      = 2    # scanlines per section
CHANGE_PER_ROW = 9    # max palette entries changed between sections
NUM_SECTIONS   = 128  # 256 / CHUNKSIZE
SCREEN_W       = 320  # Mode 1 pixels wide
SCREEN_H       = 256  # scanlines
BYTES_PER_ROW  = 80   # SCREEN_W // 4 pixels-per-byte

# ── BBC colour utilities ──────────────────────────────────────────────────────

def col_to_rgb(col: int):
    """BBC colour (0–7) → (R, G, B) each 0 or 255."""
    return (255 if col & 1 else 0,
            255 if col & 2 else 0,
            255 if col & 4 else 0)


def closest_colour(r: int, g: int, b: int) -> int:
    """Quantise 0-255 RGB to nearest BBC colour 0-7."""
    return (1 if r >= 128 else 0) | (2 if g >= 128 else 0) | (4 if b >= 128 else 0)


def lookup_cols(palette, byte_val: int):
    """
    ULA bit-extraction: return (c0, c1, c2, c3) BBC colours for a byte+palette.

    Matches OCaml lookup_cols: tap bits {b7,b5,b3,b1} for the palette index,
    then shift left and OR 1 to advance to the next pixel.
    """
    b = byte_val & 0xFF
    out = []
    for _ in range(4):
        idx = (((b & 0x80) >> 4) |
               ((b & 0x20) >> 3) |
               ((b & 0x08) >> 2) |
               ((b & 0x02) >> 1))
        out.append(palette[idx])
        b = ((b << 1) | 1) & 0xFF
    return tuple(out)


def find_byte_for_quad(palette, quad) -> int | None:
    """Brute-force: find a byte value 0-255 that produces quad with palette."""
    for bv in range(256):
        if lookup_cols(palette, bv) == quad:
            return bv
    return None


def quad_rgb_distance(q1, q2) -> float:
    """Weighted perceptual distance between two colour quads (matches OCaml calc_distance)."""
    total = 0.0
    for a, b in zip(q1, q2):
        r1, g1, b1 = col_to_rgb(a)
        r2, g2, b2 = col_to_rgb(b)
        total += (0.2126 * abs(r1 - r2) +
                  0.7152 * abs(g1 - g2) +
                  0.0722 * abs(b1 - b2))
    return total


# ── Fast quad achievability check (used by 2-step look-ahead) ────────────────

def _quad_achievable(palette, c0, c1, c2, c3):
    """
    Return True if quad (c0,c1,c2,c3) is achievable with the given palette.

    Uses the group structure that mirrors the Z3 constraint encoding:
      group g (0–7) has odd slot g*2+1 and free slots g and g+8.
    A quad is achievable iff:
      ∃ group g1: palette[g1*2+1]=c2 AND (palette[g1]=c0 OR palette[g1+8]=c0)
      ∃ group g2: palette[g2*2+1]=c3 AND (palette[g2]=c1 OR palette[g2+8]=c1)
    O(8) — much faster than scanning all 256 byte values.
    """
    found1 = any(palette[g*2+1] == c2 and (palette[g] == c0 or palette[g+8] == c0)
                 for g in range(8))
    if not found1:
        return False
    return any(palette[g*2+1] == c3 and (palette[g] == c1 or palette[g+8] == c1)
               for g in range(8))


# ── Precomputed slot→bytes map (used by hill-climbing greedy) ─────────────────

def _compute_slot_bytes():
    """For each palette slot s, which byte values b reference palette[s]?"""
    slot_bytes = [set() for _ in range(16)]
    for b in range(256):
        byte = b
        for _ in range(4):
            idx = (((byte & 0x80) >> 4) |
                   ((byte & 0x20) >> 3) |
                   ((byte & 0x08) >> 2) |
                   ((byte & 0x02) >> 1))
            slot_bytes[idx].add(b)
            byte = ((byte << 1) | 1) & 0xFF
    return [sorted(s) for s in slot_bytes]

_SLOT_BYTES = _compute_slot_bytes()


# ── Numpy colour table (used by best-effort vectorised fallback) ───────────────

_BBC_RGB_NP = np.array([[255 if c & 1 else 0,
                          255 if c & 2 else 0,
                          255 if c & 4 else 0] for c in range(8)], dtype=np.float32)
_PERC_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _best_effort_numpy(palette, unmatched_quads):
    """
    Option B: numpy-vectorised best-effort fallback.

    For each quad in unmatched_quads, find the byte value 0-255 whose
    lookup_cols result has the smallest perceptual distance to that quad.
    Replaces the O(256 × n_unmatched) Python loop with numpy operations.
    """
    if not unmatched_quads:
        return {}
    # (256, 4) — BBC colour index for each byte value × 4 pixels
    all_arr = np.array([lookup_cols(palette, b) for b in range(256)], dtype=np.int32)
    # (256, 4, 3) — RGB for each (byte, pixel)
    all_rgb = _BBC_RGB_NP[all_arr]
    besteffort = {}
    for quad in unmatched_quads:
        quad_rgb = _BBC_RGB_NP[np.array(quad, dtype=np.int32)]   # (4, 3)
        diffs = np.abs(all_rgb - quad_rgb[np.newaxis, :, :])      # (256, 4, 3)
        dists = (diffs * _PERC_WEIGHTS).sum(axis=(1, 2))           # (256,)
        besteffort[quad] = int(np.argmin(dists))
    return besteffort


def _all_bytes_rgb(palette):
    """
    Return (all_quads, all_rgb) for a palette.

    all_quads : (256, 4) int32  — BBC colour index per byte × pixel
    all_rgb   : (256, 4, 3) float32 — RGB values
    """
    all_quads = np.array([lookup_cols(palette, v) for v in range(256)], dtype=np.int32)
    return all_quads, _BBC_RGB_NP[all_quads]






# ── Mixes table (matches OCaml mixes()) ──────────────────────────────────────

def _rgb01(col):
    """BBC colour 0-7 → (r,g,b) each 0 or 1 (OCaml rgb_of_col)."""
    return col & 1, (col >> 1) & 1, (col >> 2) & 1

def _intensity01(r, g, b):
    """Perceptual intensity for r,g,b ∈ {0,1} (OCaml intensity)."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _build_mixes():
    """
    Build the 125-entry mixes table matching OCaml mixes().

    Each entry corresponds to one of the 125 quantised (r/52, g/52, b/52)
    RGB buckets and holds a list of (variance, [c0,c1,c2,c3]) tuples sorted
    by variance ascending.  The four colours in each tuple are ordered
    brightest-to-darkest.

    Deduplication matches OCaml: entries with the same sorted colour list
    (regardless of original ordering) appear only once per bucket.
    """
    buckets = [dict() for _ in range(125)]   # key: sorted-cols tuple → var

    for c1 in range(8):
        for c2 in range(c1, 8):
            for c3 in range(c2, 8):
                for c4 in range(c3, 8):
                    cols = [c1, c2, c3, c4]
                    intensities = [_intensity01(*_rgb01(c)) for c in cols]
                    rv = sum(_rgb01(c)[0] for c in cols)
                    gv = sum(_rgb01(c)[1] for c in cols)
                    bv = sum(_rgb01(c)[2] for c in cols)
                    idx = bv * 25 + gv * 5 + rv
                    mean = sum(intensities) / 4
                    var  = sum((i - mean) ** 2 for i in intensities) / 4
                    # Sort descending by intensity (OCaml: compare i2 i1)
                    sorted_cols = tuple(
                        c for c, _ in sorted(zip(cols, intensities),
                                             key=lambda p: -p[1]))
                    if sorted_cols not in buckets[idx]:
                        buckets[idx][sorted_cols] = var

    result = []
    for bucket in buckets:
        entries = sorted(((var, list(cols)) for cols, var in bucket.items()),
                         key=lambda e: e[0])
        result.append(entries)
    return result

_MIXES_TABLE = _build_mixes()

# Bayer position (dy*2+dx) → index into the descending-intensity colour list.
# Matches OCaml: let offset = [| 0; 3; 2; 1 |]
_DITHER2_POS = [0, 3, 2, 1]


def ordered_dither_2(x: int, y: int, r: int, g: int, b: int, mixno: int = 0) -> int:
    """
    Colour-aware 2×2 ordered dither (matches OCaml ordered_dither_2).

    Quantises (r,g,b) into a 5-level bucket, selects a 4-BBC-colour
    combination from the precomputed mixes table that averages to that bucket,
    then returns the colour for spatial position (x,y) using the Bayer pattern
    applied to the descending-intensity-sorted colour list.
    """
    r5  = r // 52
    g5  = g // 52
    b5  = b // 52
    idx = b5 * 25 + g5 * 5 + r5
    bucket = _MIXES_TABLE[idx]
    _, col_list = bucket[min(mixno, len(bucket) - 1)]
    return col_list[_DITHER2_POS[(y & 1) * 2 + (x & 1)]]


# ── Blue-noise dither ─────────────────────────────────────────────────────────

_BN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bluenoise.png')
try:
    _bn_img   = Image.open(_BN_PATH).convert('L')
    _BLUE_NOISE = np.array(_bn_img, dtype=np.uint8)
    _BN_H, _BN_W = _BLUE_NOISE.shape
except Exception:
    _BLUE_NOISE = None
    _BN_H = _BN_W = 1


def blue_noise_dither_2(x: int, y: int, r: int, g: int, b: int,
                        mixno: int = 0) -> int:
    """
    Colour-aware blue-noise dither.

    Identical to ordered_dither_2 but replaces the 2×2 Bayer matrix with the
    bundled 128×128 blue-noise texture.  The 0-255 threshold is split into
    four equal bands (>>6 gives 0-3) selecting from the descending-intensity
    colour list.  Produces less regular patterns than Bayer, which reduces
    visible grid artefacts on smooth gradients.
    """
    r5  = r // 52
    g5  = g // 52
    b5  = b // 52
    idx = b5 * 25 + g5 * 5 + r5
    bucket = _MIXES_TABLE[idx]
    _, col_list = bucket[min(mixno, len(bucket) - 1)]
    threshold = int(_BLUE_NOISE[y % _BN_H, x % _BN_W])
    return col_list[threshold >> 6]   # 0-63→0, 64-127→1, 128-191→2, 192-255→3


# ── Dithering ─────────────────────────────────────────────────────────────────

def dither_section_ordered(img: np.ndarray, section: int, randomness: int = 64,
                           chunk_size: int = 2, mixno: int = 0):
    """
    Colour-aware ordered dither for chunk_size rows of section using the mixes table.

    Matches the OCaml default 'ordered' mode (ordered_dither_2, mixno=0).
    Each pixel is mapped to a 5-level RGB bucket; a 4-BBC-colour combination
    is selected from the precomputed mixes table and assigned to spatial
    positions via a Bayer pattern over the sorted colour list.

    randomness (0–255): per-pixel correlated random bias applied before bucket
    assignment, matching OCaml's -random flag (default 64).  A single random
    value is drawn per pixel and scaled by perceptual channel weights
    (R×54/256, G×183/256, B×18/256), then added to the preprocessed colour.
    Set to 0 to disable.

    mixno (0–N): selects which entry from the mixes table to use.  Higher
    values prefer higher-contrast dither combinations (e.g. black+white rather
    than adjacent mid-tones).  Clamped to the number of entries available for
    each bucket.  Default 0 matches the OCaml default.

    Returns a list of (chunk_size × BYTES_PER_ROW) quads in screen order:
    [row0_byte0, …, row0_byte79, row1_byte0, …, row1_byte79].
    """
    r_rand = (randomness * 54) // 256
    g_rand = (randomness * 183) // 256
    b_rand = (randomness * 18) // 256

    quads = []
    for row in range(chunk_size):
        y = section * chunk_size + row
        for bp in range(BYTES_PER_ROW):
            pix = []
            for p in range(4):
                x = bp * 4 + p
                pr, pg, pb = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if randomness:
                    rnd = random.randint(0, 255) - 128
                    pr = max(0, min(255, pr + (rnd * r_rand) // 256))
                    pg = max(0, min(255, pg + (rnd * g_rand) // 256))
                    pb = max(0, min(255, pb + (rnd * b_rand) // 256))
                col = ordered_dither_2(x, y, pr, pg, pb, mixno=mixno)
                pix.append(col)
            quads.append(tuple(pix))
    return quads


def dither_section_bn(img: np.ndarray, section: int,
                      chunk_size: int = 2, mixno: int = 0):
    """
    Blue-noise ordered dither for chunk_size rows of section.

    Like dither_section_ordered but uses blue_noise_dither_2 instead of
    the Bayer matrix.  No randomness parameter — the noise texture already
    provides good spatial distribution.

    Returns a list of (chunk_size × BYTES_PER_ROW) quads in screen order.
    """
    quads = []
    for row in range(chunk_size):
        y = section * chunk_size + row
        for bp in range(BYTES_PER_ROW):
            pix = []
            for p in range(4):
                x = bp * 4 + p
                pr, pg, pb = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                col = blue_noise_dither_2(x, y, pr, pg, pb, mixno=mixno)
                pix.append(col)
            quads.append(tuple(pix))
    return quads


_FS_DAMPEN = 0.875   # reduce diffused error to shorten worm length


def dither_section_fs(img: np.ndarray, section: int, err: np.ndarray,
                      chunk_size: int = 2):
    """
    Floyd-Steinberg dither with serpentine scanning and error dampening.

    err : shape (3, SCREEN_W) float array holding accumulated error for the
          current row.  Updated in-place; errors propagate across section
          boundaries.

    Serpentine scanning alternates row direction (L→R, R→L, …) using the
    global scanline index so the direction is consistent across sections.
    Error dampening (_FS_DAMPEN) reduces diffused error to shorten worm length.

    Returns list of (chunk_size × BYTES_PER_ROW) quads in screen order.
    """
    quads_by_row = []
    for row in range(chunk_size):
        y = section * chunk_size + row
        left_to_right = (y % 2 == 0)
        xs = range(SCREEN_W) if left_to_right else range(SCREEN_W - 1, -1, -1)
        pix_row = [0] * SCREEN_W
        next_err = np.zeros((3, SCREEN_W), dtype=float)
        for x in xs:
            pr, pg, pb = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
            # Don't let dither error corrupt true-black border pixels
            # (letterbox/pillarbox padding added by fit resize).
            if pr == 0 and pg == 0 and pb == 0:
                err[0, x] = err[1, x] = err[2, x] = 0.0
            this_r = max(0, min(255, pr + int(err[0, x])))
            this_g = max(0, min(255, pg + int(err[1, x])))
            this_b = max(0, min(255, pb + int(err[2, x])))
            col = closest_colour(this_r, this_g, this_b)
            sel_r, sel_g, sel_b = col_to_rgb(col)
            er = (this_r - sel_r) * _FS_DAMPEN
            eg = (this_g - sel_g) * _FS_DAMPEN
            eb = (this_b - sel_b) * _FS_DAMPEN
            # Diffuse in scan direction (mirrored for R→L rows)
            fwd = 1 if left_to_right else -1
            xf = x + fwd          # forward neighbour (current row)
            xfl = x - fwd         # back-diagonal neighbour (next row)
            if 0 <= xf < SCREEN_W:
                err[0, xf]      += er * 7/16
                err[1, xf]      += eg * 7/16
                err[2, xf]      += eb * 7/16
                next_err[0, xf] += er * 1/16
                next_err[1, xf] += eg * 1/16
                next_err[2, xf] += eb * 1/16
            if 0 <= xfl < SCREEN_W:
                next_err[0, xfl] += er * 3/16
                next_err[1, xfl] += eg * 3/16
                next_err[2, xfl] += eb * 3/16
            next_err[0, x] += er * 5/16
            next_err[1, x] += eg * 5/16
            next_err[2, x] += eb * 5/16
            pix_row[x] = col
        # Build quads in left-to-right screen order regardless of scan direction
        for bp in range(BYTES_PER_ROW):
            quads_by_row.append(tuple(pix_row[bp * 4 + p] for p in range(4)))
        # Propagate: next row's starting error comes from this row's diffused error
        err[:] = next_err
    return quads_by_row


# ── Option 9: adaptive dither variance helper ─────────────────────────────────

_AUTO_DITHER_THRESHOLD = 600.0   # per-pixel variance threshold for auto dither
                                 # sections above this use FS; below use ordered


def _section_variance(arr: np.ndarray, section: int, chunk_size: int) -> float:
    """Mean per-pixel luminance variance for the source rows of this section."""
    r0 = section * chunk_size
    r1 = min(r0 + chunk_size, arr.shape[0])
    patch = arr[r0:r1].astype(np.float32)
    # Perceptual luminance: 0.299R + 0.587G + 0.114B
    lum = patch[:, :, 0] * 0.299 + patch[:, :, 1] * 0.587 + patch[:, :, 2] * 0.114
    return float(np.var(lum))


# ── Greedy palette solver (Option A) ──────────────────────────────────────────

_LOOK_AHEAD_K = 3       # top-K unmatched quads considered for 2-step look-ahead
_LOOK_AHEAD_FACTOR = 1.0  # weight of look-ahead gain relative to direct gain


def _greedy_palette(sorted_quads_with_counts, previous_palette=None,
                    look_ahead=False, changes_per_row=CHANGE_PER_ROW,
                    init_palette=None, smooth_penalty=0.0):
    """
    Option A: hill-climbing greedy palette solver replacing Z3.

    At each of up to CHANGE_PER_ROW budget steps, evaluates every possible
    single-slot change (16 slots × 7 values = 112 candidates) and applies the
    one that maximises the frequency-weighted coverage of required quads.

    previous_palette : the previous section's palette — defines the budget
                       reference (changes are counted from this).
    smooth_penalty   : Option 10 — cost subtracted from gain per new slot change
                       (in pixel-frequency units).  A positive value discourages
                       gratuitous changes that cause inter-section banding.
                       Values of 5-30 are typical; 0 = off (default).
    init_palette     : optional override for the starting state of the palette
                       before hill-climbing begins.  Slots that already differ
                       from previous_palette consume part of the budget.  Used
                       by random-restart to explore different local optima
                       without violating the inter-section change limit.

    Two-step look-ahead: for the top-K highest-frequency unmatched quads,
    checks whether any follow-up single-slot change (given this step applied)
    would make the quad achievable.  If so, credits the quad's frequency as a
    look-ahead bonus — but only when direct gain is already zero (Bug 1 fix).

    sorted_quads_with_counts : list of (quad, count) sorted by count descending.
    Returns (palette, matched_dict).  Never exceeds CHANGE_PER_ROW budget;
    unmatched quads fall through to best-effort.
    """
    prev = list(previous_palette) if previous_palette else [0] * 16
    if init_palette is not None:
        palette = list(init_palette)
    else:
        palette = list(prev)
    has_budget = previous_palette is not None
    budget = changes_per_row if has_budget else 16

    freq = {q: cnt for q, cnt in sorted_quads_with_counts}
    required_set = set(freq)
    if not required_set:
        return palette, {}

    # Current quad for each byte value, and a coverage counter
    quads = [lookup_cols(palette, b) for b in range(256)]
    achievable_cnt = Counter(quads)   # quad → number of bytes producing it

    # Slots already changed from prev (for budget tracking).
    # If init_palette was supplied, slots that differ from prev already consume budget.
    if init_palette is not None and has_budget:
        changed: set = {i for i in range(16) if palette[i] != prev[i]}
    else:
        changed: set = set()

    for _step in range(budget):
        # Top-K unmatched quads for look-ahead (re-evaluated each step)
        top_unmatched = sorted(
            [(q, freq[q]) for q in required_set if achievable_cnt[q] == 0],
            key=lambda x: -x[1]
        )[:_LOOK_AHEAD_K]

        # Precompute follow-up candidates per unmatched quad: any (slot, val)
        # where val is one of the target colours for that quad.  The
        # achievability check will filter out combinations that don't help.
        quad_follow_ups: dict = {}
        for q, _ in top_unmatched:
            c0, c1, c2, c3 = q
            targets = {c0, c1, c2, c3}
            quad_follow_ups[q] = [(s, v) for s in range(16)
                                  for v in targets if palette[s] != v]

        best_gain  = 0.0
        best_slot  = None
        best_val   = None

        for slot in range(16):
            for val in range(8):
                if val == palette[slot]:
                    continue
                # Budget guard: only count as a change if slot not already changed
                # and new value differs from prev
                if has_budget:
                    if slot not in changed and val != prev[slot]:
                        if len(changed) >= budget:
                            continue   # out of budget

                # Compute coverage delta for this candidate change
                old_slot_val = palette[slot]
                palette[slot] = val

                delta: dict = {}
                for b in _SLOT_BYTES[slot]:
                    old_q = quads[b]
                    new_q = lookup_cols(palette, b)
                    if old_q != new_q:
                        delta[old_q] = delta.get(old_q, 0) - 1
                        delta[new_q] = delta.get(new_q, 0) + 1

                # Direct gain (frequency-weighted)
                gain = 0.0
                for q, d in delta.items():
                    if q in required_set:
                        was = achievable_cnt[q] > 0
                        now = achievable_cnt[q] + d > 0
                        gain += freq[q] * ((1 if now else 0) - (1 if was else 0))

                # Option 10: boundary smoothing — subtract a small cost for
                # each NEW slot change (one not already in `changed`).
                # This discourages gratuitous changes that cause section banding.
                if smooth_penalty > 0 and has_budget:
                    if slot not in changed and val != prev[slot]:
                        gain -= smooth_penalty

                # 2-step look-ahead: credit quads that a follow-up step could
                # unlock given this step is applied first.  Only activate when
                # direct gain is zero — prevents a speculative step 1 choice
                # from overriding a genuinely useful direct gain (Bug 1 fix).
                if look_ahead and top_unmatched and gain == 0:
                    step1_is_new = slot not in changed and val != prev[slot]
                    budget_after = budget - len(changed) - (1 if step1_is_new else 0)

                    for q, fq in top_unmatched:
                        # Skip quads already covered by this step
                        if delta.get(q, 0) > 0:
                            continue
                        c0, c1, c2, c3 = q
                        # Try each follow-up candidate
                        for s2, v2 in quad_follow_ups[q]:
                            if s2 == slot:
                                continue
                            if palette[s2] == v2:
                                continue
                            # Budget guard for the follow-up step
                            if has_budget:
                                if s2 not in changed and v2 != prev[s2]:
                                    if budget_after <= 0:
                                        continue
                            old_v2 = palette[s2]
                            palette[s2] = v2
                            achievable = _quad_achievable(palette, c0, c1, c2, c3)
                            palette[s2] = old_v2
                            if achievable:
                                gain += fq * _LOOK_AHEAD_FACTOR
                                break  # one valid follow-up is enough

                palette[slot] = old_slot_val

                if gain > best_gain:
                    best_gain = gain
                    best_slot = slot
                    best_val  = val

        if best_slot is None or best_gain <= 0:
            break   # no improvement possible within budget

        # Apply the best change
        old_val = palette[best_slot]
        palette[best_slot] = best_val

        # Update achievable_cnt and quads incrementally
        for b in _SLOT_BYTES[best_slot]:
            old_q = quads[b]
            new_q = lookup_cols(palette, b)
            if old_q != new_q:
                achievable_cnt[old_q] -= 1
                achievable_cnt[new_q]  += 1
                quads[b] = new_q

        # Track budget
        if has_budget:
            if best_slot not in changed and best_val != prev[best_slot]:
                changed.add(best_slot)
            elif best_slot in changed and best_val == prev[best_slot]:
                changed.discard(best_slot)   # reverted to prev

    # Build matched dict
    matched = {}
    for quad in required_set:
        bv = find_byte_for_quad(palette, quad)
        if bv is not None:
            matched[quad] = bv

    return palette, matched


# ── Beam search palette solver (Option 3) ─────────────────────────────────────

def _beam_palette(sorted_quads_with_counts, previous_palette=None,
                  changes_per_row=CHANGE_PER_ROW, beam_width=5):
    """
    Option 3: beam search palette solver.

    Keeps the top `beam_width` palette states at each budget step rather than
    committing to one.  Each step expands every beam state with all 112
    single-slot candidates; the top-k unique resulting states (by
    frequency-weighted coverage score) are retained.  Guarantees at least as
    good a result as greedy (beam_width=1 == greedy without look-ahead).

    previous_palette : budget reference — changes counted from this.
    Returns (palette, matched_dict).
    """
    prev = list(previous_palette) if previous_palette else [0] * 16
    has_budget = previous_palette is not None
    budget = changes_per_row if has_budget else 16

    freq = {q: cnt for q, cnt in sorted_quads_with_counts}
    required_set = set(freq)
    if not required_set:
        return list(prev), {}

    def _score(cnt):
        return sum(freq[q] for q in required_set if cnt[q] > 0)

    def _make_state(pal, changed=None):
        qs  = [lookup_cols(pal, b) for b in range(256)]
        cnt = Counter(qs)
        ch  = set(changed) if changed else set()
        return dict(palette=list(pal), quads=qs, cnt=cnt,
                    changed=ch, score=_score(cnt))

    # Initial beam: just the starting palette
    start = list(prev)
    beam = [_make_state(start)]

    for _step in range(budget):
        candidates = []
        for state in beam:
            pal     = state['palette']
            ch      = state['changed']
            qs      = state['quads']
            cnt     = state['cnt']
            for slot in range(16):
                for val in range(8):
                    if val == pal[slot]:
                        continue
                    # Budget guard
                    if has_budget and slot not in ch and val != prev[slot]:
                        if len(ch) >= budget:
                            continue
                    # Compute new state incrementally
                    new_pal = list(pal)
                    new_pal[slot] = val
                    new_ch = set(ch)
                    if slot not in ch and val != prev[slot]:
                        new_ch.add(slot)
                    elif slot in ch and val == prev[slot]:
                        new_ch.discard(slot)
                    new_qs  = list(qs)
                    new_cnt = Counter(cnt)
                    for b in _SLOT_BYTES[slot]:
                        old_q = new_qs[b]
                        nq    = lookup_cols(new_pal, b)
                        if old_q != nq:
                            new_cnt[old_q] -= 1
                            new_cnt[nq]    += 1
                            new_qs[b]       = nq
                    sc = _score(new_cnt)
                    if sc > state['score']:   # only keep improvements
                        candidates.append(dict(palette=new_pal, quads=new_qs,
                                               cnt=new_cnt, changed=new_ch,
                                               score=sc))

        if not candidates:
            break   # no improvement possible

        # Merge current beam + candidates, deduplicate by palette, keep top-k
        candidates.sort(key=lambda s: -s['score'])
        seen: set = set()
        new_beam = []
        for s in candidates + beam:
            key = tuple(s['palette'])
            if key not in seen:
                seen.add(key)
                new_beam.append(s)
                if len(new_beam) >= beam_width:
                    break
        beam = new_beam

    best_palette = beam[0]['palette']
    matched = {}
    for quad in required_set:
        bv = find_byte_for_quad(best_palette, quad)
        if bv is not None:
            matched[quad] = bv

    return best_palette, matched


# ── Simulated annealing palette solver (Option 2) ─────────────────────────────

def _anneal_palette(sorted_quads_with_counts, previous_palette=None,
                    changes_per_row=CHANGE_PER_ROW,
                    anneal_steps=200, t_start=2.0, t_end=0.05):
    """
    Option 2: simulated annealing palette solver.

    Unlike the greedy which accepts only improvements, SA occasionally accepts
    downhill moves early in the schedule.  This lets it escape local optima
    that the greedy cannot escape within the budget.

    anneal_steps : total SA steps (budget guard still applies to hard changes)
    t_start      : initial temperature (fraction of total frequency)
    t_end        : final temperature

    Temperature is scaled by total pixel frequency so the acceptance probability
    is image-independent.  The budget is enforced as a hard constraint: a
    candidate that would exceed `changes_per_row` changes from previous_palette
    is never accepted.

    Returns (palette, matched_dict).
    """
    import math

    prev = list(previous_palette) if previous_palette else [0] * 16
    has_budget = previous_palette is not None
    budget = changes_per_row if has_budget else 16

    freq = {q: cnt for q, cnt in sorted_quads_with_counts}
    required_set = set(freq)
    if not required_set:
        return list(prev), {}

    total_freq = sum(freq.values())
    T0 = t_start * total_freq
    T1 = t_end * total_freq

    palette = list(prev)
    quads = [lookup_cols(palette, b) for b in range(256)]
    achievable_cnt = Counter(quads)
    changed: set = set()

    def current_score():
        return sum(freq[q] for q in required_set if achievable_cnt[q] > 0)

    best_palette = list(palette)
    best_score   = current_score()
    score        = best_score

    for step in range(anneal_steps):
        T = T0 * (T1 / T0) ** (step / max(1, anneal_steps - 1))

        # Pick a random candidate
        slot = random.randint(0, 15)
        val  = random.randint(0, 6)
        if val >= palette[slot]:
            val += 1   # skip current value, ensuring val != palette[slot]

        # Budget guard
        if has_budget and slot not in changed and val != prev[slot]:
            if len(changed) >= budget:
                continue

        # Evaluate delta
        old_val = palette[slot]
        palette[slot] = val
        delta: dict = {}
        for b in _SLOT_BYTES[slot]:
            old_q = quads[b]
            new_q = lookup_cols(palette, b)
            if old_q != new_q:
                delta[old_q] = delta.get(old_q, 0) - 1
                delta[new_q] = delta.get(new_q, 0) + 1

        gain = 0.0
        for q, d in delta.items():
            if q in required_set:
                was = achievable_cnt[q] > 0
                now = achievable_cnt[q] + d > 0
                gain += freq[q] * ((1 if now else 0) - (1 if was else 0))

        # Accept?
        if gain >= 0 or (T > 0 and random.random() < math.exp(gain / T)):
            # Apply
            for b in _SLOT_BYTES[slot]:
                old_q = quads[b]
                new_q = lookup_cols(palette, b)
                if old_q != new_q:
                    achievable_cnt[old_q] -= 1
                    achievable_cnt[new_q]  += 1
                    quads[b] = new_q
            if has_budget:
                if slot not in changed and val != prev[slot]:
                    changed.add(slot)
                elif slot in changed and val == prev[slot]:
                    changed.discard(slot)
            score += gain
            if score > best_score:
                best_score   = score
                best_palette = list(palette)
        else:
            # Revert
            palette[slot] = old_val

    matched = {}
    for quad in required_set:
        bv = find_byte_for_quad(best_palette, quad)
        if bv is not None:
            matched[quad] = bv

    return best_palette, matched


# ── Z3 palette solver (legacy, used when --solver z3) ─────────────────────────

def _add_quad_constraint(solver, pal, quad):
    """Add Z3 constraints asserting that quad (c0,c1,c2,c3) is achievable."""
    z3 = _z3_mod
    c0, c1, c2, c3 = quad
    alts_02, alts_13 = [], []
    for g in range(8):
        idx_odd = g * 2 + 1
        for idx_free in (g, g + 8):
            alts_02.append(z3.And(pal[idx_free] == c0, pal[idx_odd] == c2))
            alts_13.append(z3.And(pal[idx_free] == c1, pal[idx_odd] == c3))
    solver.add(z3.Or(*alts_02))
    solver.add(z3.Or(*alts_13))


def _solve_palette_z3(required_quads, previous_palette=None,
                      changes_per_row=CHANGE_PER_ROW):
    """
    Find a 16-entry BBC palette satisfying all required_quads using Z3.

    Returns (palette_list, matched_dict) on SAT, or None on UNSAT.
    """
    z3 = _z3_mod
    s   = z3.Solver()
    pal = [z3.Int(f'p{i}') for i in range(16)]
    for p in pal:
        s.add(p >= 0, p <= 7)
    for quad in required_quads:
        _add_quad_constraint(s, pal, quad)
    if previous_palette is not None:
        same = [z3.If(pal[i] == int(previous_palette[i]),
                      z3.IntVal(1), z3.IntVal(0)) for i in range(16)]
        s.add(z3.Sum(same) >= 16 - changes_per_row)
    if s.check() != z3.sat:
        return None
    model   = s.model()
    palette = [model[pal[i]].as_long() for i in range(16)]
    matched = {}
    for quad in required_quads:
        bv = find_byte_for_quad(palette, quad)
        if bv is not None:
            matched[quad] = bv
    return palette, matched


# ── Per-section palette search ────────────────────────────────────────────────

def find_palette_for_section(sorted_quads, previous_palette,
                              verbose=True, solver='greedy', look_ahead=False,
                              changes_per_row=CHANGE_PER_ROW, restarts=1,
                              beam=1, anneal=0, smooth_penalty=0.0):
    """
    Find a palette for this section.

    solver   : 'greedy' (default) — hill-climbing greedy with numpy best-effort
               'z3'               — Z3 SMT binary search (requires z3-solver)
    restarts : number of random-restart attempts for the greedy solver (Option 1).
               Each attempt beyond the first starts from a randomly-perturbed
               copy of previous_palette; the best coverage wins.
    beam     : Option 3 beam width (1 = greedy, 5-10 = beam search). When > 1,
               uses _beam_palette instead of _greedy_palette. Ignores restarts.
    anneal        : Option 2 SA step count (0 = off). When > 0, uses _anneal_palette.
                    Typical values: 100-500. Ignores restarts and beam.
    smooth_penalty: Option 10 — per-new-slot-change cost in pixel-frequency units.
                    Passed to _greedy_palette; values 5-30 reduce banding.

    Both paths share Option C (early exit when palette is already sufficient)
    and Option B (numpy-vectorised best-effort for unmatched quads).

    sorted_quads : list of (quad, count) sorted by count descending.
    Returns (palette, matched_dict, besteffort_dict).
    """
    all_quads = [q for q, _ in sorted_quads]
    n = len(all_quads)

    if n == 0:
        palette = list(previous_palette) if previous_palette else [0] * 16
        return palette, {}, {}

    # ── Option C: skip solver if current palette already covers all quads ─────
    if previous_palette is not None:
        achievable = {lookup_cols(previous_palette, b) for b in range(256)}
        if all(q in achievable for q in all_quads):
            matched = {}
            for quad in all_quads:
                bv = find_byte_for_quad(previous_palette, quad)
                if bv is not None:
                    matched[quad] = bv
            if verbose:
                print(f"    All {n} unique quads achievable (palette unchanged)")
            return list(previous_palette), matched, {}

    if solver == 'z3':
        palette, matched, besteffort = _find_palette_z3(
            all_quads, previous_palette, verbose,
            changes_per_row=changes_per_row)
    elif anneal > 0:
        # ── Option 2: simulated annealing ─────────────────────────────────────
        palette, matched = _anneal_palette(sorted_quads, previous_palette,
                                            changes_per_row=changes_per_row,
                                            anneal_steps=anneal)
        unmatched = [q for q in all_quads if q not in matched]
        if verbose:
            if unmatched:
                print(f"    Anneal({anneal}): {n - len(unmatched)}/{n} quads satisfied; "
                      f"{len(unmatched)} best-effort")
            else:
                print(f"    All {n} unique quads satisfied by anneal({anneal})")
        besteffort = _best_effort_numpy(palette, unmatched)
    elif beam > 1:
        # ── Option 3: beam search ─────────────────────────────────────────────
        palette, matched = _beam_palette(sorted_quads, previous_palette,
                                         changes_per_row=changes_per_row,
                                         beam_width=beam)
        unmatched = [q for q in all_quads if q not in matched]
        if verbose:
            if unmatched:
                print(f"    Beam({beam}): {n - len(unmatched)}/{n} quads satisfied; "
                      f"{len(unmatched)} best-effort")
            else:
                print(f"    All {n} unique quads satisfied by beam({beam})")
        besteffort = _best_effort_numpy(palette, unmatched)
    else:
        # ── Option A + Option 1: greedy with random restarts ──────────────────
        freq = {q: c for q, c in sorted_quads}
        best_palette, best_matched = None, {}
        best_score = -1

        for attempt in range(max(1, restarts)):
            if attempt == 0:
                start_pal = previous_palette
            else:
                # Perturb: randomly change 1..changes_per_row slots
                start_pal = list(previous_palette) if previous_palette else [0] * 16
                n_perturb = random.randint(1, max(1, changes_per_row // 2))
                for _ in range(n_perturb):
                    start_pal[random.randint(0, 15)] = random.randint(0, 7)

            p, m = _greedy_palette(sorted_quads, previous_palette,
                                   look_ahead=look_ahead,
                                   changes_per_row=changes_per_row,
                                   init_palette=start_pal if attempt > 0 else None,
                                   smooth_penalty=smooth_penalty)
            score = sum(freq.get(q, 0) for q in m)
            if score > best_score:
                best_score = score
                best_palette, best_matched = p, m

        palette, matched = best_palette, best_matched
        unmatched = [q for q in all_quads if q not in matched]
        if verbose:
            suffix = f" (best of {restarts} restarts)" if restarts > 1 else ""
            if unmatched:
                print(f"    Greedy: {n - len(unmatched)}/{n} quads satisfied; "
                      f"{len(unmatched)} best-effort{suffix}")
            else:
                print(f"    All {n} unique quads satisfied by greedy{suffix}")
        # ── Option B: numpy-vectorised best-effort ────────────────────────────
        besteffort = _best_effort_numpy(palette, unmatched)

    return palette, matched, besteffort


def _find_palette_z3(all_quads, previous_palette, verbose,
                     changes_per_row=CHANGE_PER_ROW):
    """Z3 binary-search solver path (mirrors the original OCaml approach)."""
    if not _Z3_AVAILABLE:
        raise RuntimeError("z3-solver is not installed.  "
                           "Run: pip install z3-solver")
    n = len(all_quads)

    # Try satisfying everything first
    result = _solve_palette_z3(all_quads, previous_palette,
                               changes_per_row=changes_per_row)
    if result:
        palette, matched = result
        if verbose:
            print(f"    Z3: all {n} unique quads satisfied")
        return palette, matched, {}

    # Binary search for maximum satisfiable prefix
    prev_pal = previous_palette
    r1 = _solve_palette_z3(all_quads[:1], prev_pal,
                           changes_per_row=changes_per_row)
    if r1 is None:
        r1 = _solve_palette_z3(all_quads[:1], None,
                               changes_per_row=changes_per_row)
        if r1 is None:
            raise RuntimeError("Single-quad Z3 solve failed even without continuity")
        if verbose:
            print("    Z3 warning: continuity constraint dropped for 1-quad fallback")
        prev_pal = None

    best_result, best_split = r1, 0
    lo, hi = 1, n
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if verbose:
            print(f"    Z3 binary search {lo}–{hi}: trying {mid+1} quads ...",
                  end=' ', flush=True)
        r = _solve_palette_z3(all_quads[:mid + 1], prev_pal,
                              changes_per_row=changes_per_row)
        if r is not None:
            if verbose: print("SAT")
            best_result, best_split = r, mid
            lo = mid
        else:
            if verbose: print("UNSAT")
            hi = mid

    palette, matched = best_result
    if verbose:
        print(f"    Z3: satisfied {best_split + 1}/{n} quads; "
              f"{n - best_split - 1} best-effort")

    besteffort = _best_effort_numpy(palette, all_quads[best_split + 1:])
    return palette, matched, besteffort


# ── Screen byte layout ────────────────────────────────────────────────────────

def screen_offset(section: int, byte_in_section: int, chunk_size: int = 2) -> int:
    """
    Byte offset into the 20480-byte screen buffer for a given section and
    per-section byte index.

    BBC non-linear interleaved layout: each 640-byte block holds 8 rows
    interleaved byte-by-byte.  chunk_size rows form one section; there are
    8 // chunk_size sections per 640-byte block.

        spb       = 8 // chunk_size
        row_start = (section // spb) * 640 + (section % spb) * chunk_size
        offset    = row_start + (idx % 80) * 8 + (idx // 80)
    """
    spb = 8 // chunk_size
    row_start = (section // spb) * 640 + (section % spb) * chunk_size
    return row_start + (byte_in_section % 80) * 8 + (byte_in_section // 80)


# ── Image resize / crop ───────────────────────────────────────────────────────

def _fit_image(img: Image.Image, mode: str) -> Image.Image:
    """
    Resize and/or crop img to exactly SCREEN_W × SCREEN_H.

    mode:
      'fit'          — scale to fit inside 320×256, preserve aspect ratio,
                       centre on a black canvas (letterbox / pillarbox).
      'crop-left'    — scale so height == 256 (may make width > 320),
                       then discard the left-hand excess.
      'crop-right'   — same, discard the right-hand excess.
      'crop-top'     — scale so width == 320 (may make height > 256),
                       then discard the top excess.
      'crop-bottom'  — same, discard the bottom excess.
    """
    src_w, src_h = img.size

    if mode == 'fit':
        scale  = min(SCREEN_W / src_w, SCREEN_H / src_h)
        new_w  = round(src_w * scale)
        new_h  = round(src_h * scale)
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new('RGB', (SCREEN_W, SCREEN_H), (0, 0, 0))
        canvas.paste(scaled, ((SCREEN_W - new_w) // 2, (SCREEN_H - new_h) // 2))
        return canvas

    if mode in ('crop-left', 'crop-right'):
        scale  = SCREEN_H / src_h
        new_w  = round(src_w * scale)
        scaled = img.resize((new_w, SCREEN_H), Image.LANCZOS)
        excess = new_w - SCREEN_W
        x0     = excess if mode == 'crop-left' else 0
        return scaled.crop((x0, 0, x0 + SCREEN_W, SCREEN_H))

    if mode in ('crop-top', 'crop-bottom'):
        scale  = SCREEN_W / src_w
        new_h  = round(src_h * scale)
        scaled = img.resize((SCREEN_W, new_h), Image.LANCZOS)
        excess = new_h - SCREEN_H
        y0     = excess if mode == 'crop-top' else 0
        return scaled.crop((0, y0, SCREEN_W, y0 + SCREEN_H))

    raise ValueError(f"Unknown resize mode: {mode!r}")


# ── Section 0 palette initialisation ─────────────────────────────────────────

def _initial_palette_from_image(arr: np.ndarray) -> list:
    """
    Seed an initial 16-slot palette from the image's dominant BBC colours.

    Preprocesses every pixel (gamma + border clamp), quantises to the nearest
    BBC colour (0–7), and counts occurrences.  The 8 BBC colours ranked by
    frequency are then distributed across all 16 palette slots:

        slots[i] = dominant_colours[i % 8]

    This ensures every dominant colour appears in both even and odd slots,
    giving the greedy solver a much better starting point than all-black.
    """
    counts = Counter()
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            counts[closest_colour(
                int(arr[y, x, 0]), int(arr[y, x, 1]), int(arr[y, x, 2]))] += 1

    # All 8 colours ranked by frequency; fill any absent colours at the end
    ranked = [c for c, _ in counts.most_common()]
    for c in range(8):
        if c not in ranked:
            ranked.append(c)

    return [ranked[i % 8] for i in range(16)]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _write_palette_delta(output_palettes, section, palette, previous_palette,
                         changes_per_row, num_sections, verbose):
    """Write one section's palette delta into the output_palettes bytearray."""
    if section == 0:
        for i in range(16):
            output_palettes[i] = (i << 4) | (palette[i] ^ 7)
    else:
        same_no   = 0
        change_no = 0
        for i in range(16):
            entry_changed = (palette[i] != previous_palette[i])
            force_change  = (same_no >= 16 - changes_per_row)
            if force_change or entry_changed:
                if verbose and entry_changed:
                    print(f"  Entry {i}: {previous_palette[i]} → {palette[i]}")
                if change_no >= changes_per_row:
                    raise AssertionError(
                        f"Section {section}: more than {changes_per_row} "
                        "palette changes required — solver bug")
                output_palettes[256 + change_no * num_sections + (section - 1)] = \
                    (i << 4) | (palette[i] ^ 7)
                change_no += 1
            else:
                same_no += 1


def _write_screen_section(screen_bytes, section, byte_list, chunk_size,
                          palette, preview_arr):
    """Write a flat list of byte values (one per byte-position in section)
    into screen_bytes and optionally into preview_arr."""
    for idx, bv in enumerate(byte_list):
        off = screen_offset(section, idx, chunk_size=chunk_size)
        if off < 20480:
            screen_bytes[off] = bv
        if preview_arr is not None:
            actual_quad = lookup_cols(palette, bv)
            row_in_section = idx // BYTES_PER_ROW
            bp             = idx  % BYTES_PER_ROW
            y = section * chunk_size + row_in_section
            for p, col in enumerate(actual_quad):
                preview_arr[y, bp * 4 + p] = col_to_rgb(col)


def process_image(png_path: str, output_path: str,
                  dither: str = 'ordered', verbose: bool = True,
                  preview_path: str = None, solver: str = 'greedy',
                  resize: str = 'fit', randomness: int = 64,
                  look_ahead: bool = False, chunk_size: int = 2,
                  changes_per_row: int = CHANGE_PER_ROW,
                  restarts: int = 1, mixno: int = 0, sharpen: float = 0.0,
                  auto_threshold: float = _AUTO_DITHER_THRESHOLD,
                  beam: int = 1, anneal: int = 0, smooth: float = 0.0,
                  saturation: float = 1.0, autolevel: bool = False,
                  contrast: float = 1.0, brightness: float = 1.0,
                  input_gamma: float = 1.0, hue: float = 0.0,
                  denoise: bool = False, posterise: int = 0,
                  progress_callback=None):
    """
    Load a PNG, resize to 320×256, run the palsearch algorithm, and write
    the output binary.

    chunk_size      : scanlines per section (1 or 2; default 2)
    changes_per_row : max palette slot changes between sections (default 9)
    restarts        : Option 1 — random-restart count for greedy solver
    mixno           : Option 7 — mixes table entry index (0 = lowest contrast)
    sharpen         : Option 8 — unsharp-mask strength (0 = off, 1.0 = strong)
    auto_threshold  : Option 9 — per-pixel luminance variance threshold used
                      when dither='auto' to choose ordered vs FS per section.
    beam            : Option 3 — beam search width for palette solver (1=greedy,
                      5-10=beam search). Keeps top-k palette states at each
                      budget step; much less likely to commit to a bad choice.
    anneal          : Option 2 — simulated annealing step count (0=off).
                      Accepts occasional downhill moves to escape local optima.
                      Typical values: 100-500. Overrides beam/restarts.
    smooth          : Option 10 — per-new-slot-change cost (pixel-frequency units,
                      0=off). Discourages gratuitous slot changes to reduce
    saturation      : Prep 1 — colour saturation multiplier (1.0=none, >1 boost).
    autolevel       : Prep 2 — stretch per-channel histogram to 0-255.
    contrast        : Prep 3 — contrast multiplier (1.0=none, >1 increase).
    brightness      : Prep 4 — brightness multiplier (1.0=none, >1 brighter).
    input_gamma     : Prep 5 — gamma applied to image pixels (1.0=none, >1 brighter).
    hue             : Prep 6 — hue rotation in degrees (-180 to 180).
    denoise         : Prep 7 — apply median filter to reduce noise.
    posterise       : Prep 8 — reduce each channel to N bits (0=off, 1-7 bits).
                      inter-section banding. Values of 5-30 are typical.

    Output layout:
        Bytes 0–(pal_size-1) : palette data
                               256-byte initial section (16 used + 240 padding)
                               + changes_per_row × num_sections delta bytes
        Bytes pal_size+      : 20480 bytes of BBC screen data

    If preview_path is given, also write a 320×256 PNG showing the actual
    colours that will appear on the BBC (i.e. after palette solve and
    best-effort fallback, not the dithered input).
    """

    num_sections = SCREEN_H // chunk_size
    pal_size     = 256 + changes_per_row * num_sections

    # Load and resize / crop to 320×256
    img = Image.open(png_path).convert('RGB')
    if img.size != (SCREEN_W, SCREEN_H):
        if verbose:
            print(f"Resizing {img.size} -> ({SCREEN_W}x{SCREEN_H})  [{resize}]")
        img = _fit_image(img, resize)

    # ── Image preprocessing pipeline ──────────────────────────────────────────
    from PIL import ImageEnhance, ImageOps, ImageFilter

    # Prep 2: auto-levels (per-channel histogram stretch to 0-255)
    if autolevel:
        img = ImageOps.autocontrast(img)
        if verbose:
            print("Applied auto-levels")

    # Prep 5: input gamma
    if input_gamma != 1.0:
        exp = 1.0 / input_gamma
        lut = [int(min(255, max(0, (i / 255.0) ** exp * 255.0 + 0.5)))
               for i in range(256)]
        img = img.point(lut * 3)   # R, G, B each get the same curve
        if verbose:
            print(f"Applied input gamma {input_gamma:.2f}")

    # Prep 3: contrast
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
        if verbose:
            print(f"Applied contrast {contrast:.2f}")

    # Prep 4: brightness
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
        if verbose:
            print(f"Applied brightness {brightness:.2f}")

    # Prep 1: saturation (most impactful — do after levels/contrast/brightness)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
        if verbose:
            print(f"Applied saturation {saturation:.2f}")

    # Prep 6: hue rotation via numpy RGB→HSV→RGB
    if hue != 0.0:
        h_shift = (hue % 360.0) / 360.0
        rgba = np.array(img, dtype=np.float32) / 255.0
        r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
        cmax  = np.maximum(np.maximum(r, g), b)
        cmin  = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        # Hue (0-1)
        hh = np.zeros_like(r)
        mask = delta > 0
        mr = mask & (cmax == r)
        mg = mask & (cmax == g)
        mb = mask & (cmax == b)
        hh[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
        hh[mg] = ((b[mg] - r[mg]) / delta[mg]) + 2.0
        hh[mb] = ((r[mb] - g[mb]) / delta[mb]) + 4.0
        hh = (hh / 6.0 + h_shift) % 1.0
        ss = np.where(cmax > 0, delta / np.where(cmax > 0, cmax, 1.0), 0.0)
        vv = cmax
        # HSV → RGB
        hh6  = hh * 6.0
        hi   = np.floor(hh6).astype(int) % 6
        ff   = hh6 - np.floor(hh6)
        p    = vv * (1.0 - ss)
        q    = vv * (1.0 - ss * ff)
        t    = vv * (1.0 - ss * (1.0 - ff))
        out  = np.zeros_like(rgba)
        for ch, (r0, g0, b0) in enumerate([(vv, t, p), (q, vv, p), (p, vv, t),
                                             (p, q, vv), (t, p, vv), (vv, p, q)]):
            mask2 = hi == ch
            out[:, :, 0][mask2] = r0[mask2]
            out[:, :, 1][mask2] = g0[mask2]
            out[:, :, 2][mask2] = b0[mask2]
        img = Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), 'RGB')
        if verbose:
            print(f"Applied hue rotation {hue:.1f} deg")

    # Prep 7: denoise (median filter)
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
        if verbose:
            print("Applied denoise (median filter)")

    # Prep 8: posterise (quantise to N bits then rescale to full 0-255 range)
    if posterise > 0:
        levels = (1 << posterise) - 1   # e.g. bits=1 → 1, bits=2 → 3
        lut = [int(((v >> (8 - posterise)) * 255 / levels) + 0.5)
               for v in range(256)]
        img = img.point(lut * 3)
        if verbose:
            print(f"Applied posterise ({posterise} bits, {levels + 1} levels)")

    # Option 8: pre-sharpen before dithering
    if sharpen > 0:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=1.5, percent=int(sharpen * 150), threshold=3))
        if verbose:
            print(f"Applied pre-sharpening (strength {sharpen:.2f})")

    arr = np.array(img, dtype=np.uint8)

    output_palettes = bytearray(pal_size)
    screen_bytes    = bytearray(20480)

    if verbose:
        print("Computing initial palette from image dominant colours...")
    previous_palette = _initial_palette_from_image(arr)
    if verbose:
        print(f"  Initial palette: {previous_palette}")
    fs_err = np.zeros((3, SCREEN_W), dtype=float)   # FS error diffusion state

    # Preview buffer: RGB pixels at native 320×256
    preview_arr = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) if preview_path else None

    # ── Process sections sequentially ─────────────────────────────────────────
    for section in range(num_sections):
        if verbose:
            print(f"\nSection {section}/{num_sections - 1}:")
            sys.stdout.flush()

        # ── Option 9: adaptive dither mode selection ──────────────────────────
        if dither == 'auto':
            var = _section_variance(arr, section, chunk_size)
            section_dither = 'fs' if var > auto_threshold else 'ordered'
            if verbose:
                print(f"  Variance {var:.0f} → {section_dither}")
        else:
            section_dither = dither

        # ── Dither ────────────────────────────────────────────────────────────
        if section_dither == 'ordered':
            quads = dither_section_ordered(arr, section,
                                           randomness=randomness,
                                           chunk_size=chunk_size,
                                           mixno=mixno)
        elif section_dither == 'bn':
            quads = dither_section_bn(arr, section,
                                      chunk_size=chunk_size,
                                      mixno=mixno)
        else:
            quads = dither_section_fs(arr, section, fs_err,
                                      chunk_size=chunk_size)

        counts       = Counter(quads)
        sorted_quads = sorted(counts.items(), key=lambda x: -x[1])

        if verbose:
            print(f"  {len(sorted_quads)} unique quads from "
                  f"{chunk_size * BYTES_PER_ROW} bytes")

        # ── Solve ──────────────────────────────────────────────────────────────
        palette, matched, besteffort = find_palette_for_section(
            sorted_quads, previous_palette, verbose=verbose,
            solver=solver, look_ahead=look_ahead,
            changes_per_row=changes_per_row, restarts=restarts,
            beam=beam, anneal=anneal, smooth_penalty=smooth)

        # ── Write screen bytes ─────────────────────────────────────────────────
        byte_list = [matched.get(quad, besteffort.get(quad, 0))
                     for quad in quads]
        _write_screen_section(screen_bytes, section, byte_list,
                              chunk_size, palette, preview_arr)

        # ── Write palette data ─────────────────────────────────────────────────
        _write_palette_delta(output_palettes, section, palette, previous_palette,
                             changes_per_row, num_sections, verbose)

        previous_palette = list(palette)

        if progress_callback:
            progress_callback((section + 1) / num_sections)

    # ── Write output file ──────────────────────────────────────────────────────
    with open(output_path, 'wb') as f:
        f.write(output_palettes)
        f.write(screen_bytes)

    if verbose:
        total = len(output_palettes) + len(screen_bytes)
        print(f"\nWrote {output_path}: {total} bytes "
              f"({len(output_palettes)} palette + {len(screen_bytes)} screen)")

    # ── Write preview PNG ──────────────────────────────────────────────────────
    if preview_path is not None:
        Image.fromarray(preview_arr, 'RGB').save(preview_path)
        if verbose:
            print(f"Wrote preview {preview_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='BBC Master Mode 1 palette search (Python port of palsearch.ml)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python palsearch.py photo.png -o frog.bin
    python palsearch.py photo.png -o frog.bin -d fs
    python palsearch.py photo.png -o frog.bin -p preview.png
    python palsearch.py photo.png -o frog.bin -s z3
    python palsearch.py photo.png -o frog.bin -q

The output binary is compatible with showimage.s for playback on BBC Master.
        """)
    ap.add_argument('input',
                    help='Input PNG file (auto-resized to 320×256)')
    ap.add_argument('-o', '--output', default='output.bin',
                    help='Output binary file (default: output.bin)')
    ap.add_argument('-d', '--dither', choices=['ordered', 'fs', 'bn', 'auto'],
                    default='ordered',
                    help=('Dithering method: '
                          'ordered = Bayer 2x2 (default), '
                          'fs = Floyd-Steinberg, '
                          'bn = blue-noise ordered dither (128x128 texture), '
                          'auto = Option 9: ordered for low-variance sections, '
                          'FS for high-detail/edge sections'))
    ap.add_argument('-p', '--preview',
                    help='Write a preview PNG of the converted image')
    ap.add_argument('-r', '--resize',
                    choices=['fit', 'crop-left', 'crop-right',
                             'crop-top', 'crop-bottom'],
                    default='fit',
                    help=('How to fit the input to 320×256: '
                          'fit = letterbox/pillarbox centred (default); '
                          'crop-left/right = fill height, crop excess width; '
                          'crop-top/bottom = fill width, crop excess height'))
    ap.add_argument('-s', '--solver', choices=['greedy', 'z3'], default='greedy',
                    help=('Palette solver: '
                          'greedy = fast hill-climbing (default), '
                          'z3 = SMT binary search (requires pip install z3-solver)'))
    ap.add_argument('--randomness', type=int, default=64, metavar='N',
                    help=('Per-pixel random bias for ordered dither, 0–255 '
                          '(default 64, matching OCaml -random 64). '
                          'Higher values break up flat colour regions more. '
                          'Use 0 to disable.'))
    ap.add_argument('--look-ahead', action='store_true', default=False,
                    help=('Enable 2-step look-ahead in the greedy solver. '
                          'Only activates when direct gain is zero, so it '
                          'breaks ties towards setups that enable a follow-up '
                          'change (e.g. white requiring two coordinated slots). '
                          'Slower but helps images with hard-to-reach colours.'))
    ap.add_argument('--chunk-size', type=int, choices=[1, 2], default=2,
                    metavar='N',
                    help=('Scanlines per section (1 or 2; default 2). '
                          'Use 1 for per-scanline palette changes when the '
                          'stable raster loop has sufficient cycles.'))
    ap.add_argument('--changes', type=int, default=CHANGE_PER_ROW, metavar='N',
                    help=f'Max palette slot changes per section (default {CHANGE_PER_ROW})')
    ap.add_argument('--restarts', type=int, default=1, metavar='N',
                    help=('Option 1: number of greedy random-restart attempts '
                          'per section (default 1 = no restarts). Each extra '
                          'attempt starts from a randomly-perturbed copy of '
                          'the previous palette; the best result wins. '
                          'Values of 5–20 trade runtime for quality.'))
    ap.add_argument('--mixno', type=int, default=0, metavar='N',
                    help=('Option 7: mixes table entry index for ordered dither '
                          '(default 0 = lowest contrast / smoothest). Higher '
                          'values select higher-contrast dither combinations '
                          '(e.g. black+white rather than adjacent mid-tones). '
                          'Useful for images with strong highlights or shadows.'))
    ap.add_argument('--sharpen', type=float, default=0.0, metavar='F',
                    help=('Option 8: pre-sharpen the input image before '
                          'dithering (default 0 = off). Applies an unsharp '
                          'mask; 0.5 is mild, 1.0 is strong. Recovers '
                          'perceived detail lost during low-res dithering.'))

    # ── Image preprocessing ────────────────────────────────────────────────────
    ap.add_argument('--saturation', type=float, default=1.0, metavar='F',
                    help=('Prep 1: colour saturation multiplier (default 1.0 = none). '
                          'Values > 1 push colours toward BBC colour corners, '
                          'reducing dither noise. 1.3-1.8 suits most photos.'))
    ap.add_argument('--autolevel', action='store_true', default=False,
                    help=('Prep 2: stretch per-channel histogram to full 0-255 '
                          'range before dithering. Maximises utilisation of the '
                          '8 BBC colours. Recommended for washed-out images.'))
    ap.add_argument('--contrast', type=float, default=1.0, metavar='F',
                    help=('Prep 3: contrast multiplier (default 1.0 = none). '
                          'Values > 1 push midtones toward black/white, '
                          'reducing quantisation error. Apply after --autolevel.'))
    ap.add_argument('--brightness', type=float, default=1.0, metavar='F',
                    help=('Prep 4: brightness multiplier (default 1.0 = none). '
                          'Values > 1 brighten, < 1 darken. Useful for '
                          'systematically under- or overexposed images.'))
    ap.add_argument('--input-gamma', type=float, default=1.0, metavar='F',
                    help=('Prep 5: gamma curve applied to image pixels before '
                          'scoring (default 1.0 = none). Values > 1 brighten '
                          'midtones; < 1 darken them.'))
    ap.add_argument('--hue', type=float, default=0.0, metavar='DEG',
                    help=('Prep 6: hue rotation in degrees (default 0 = none). '
                          'Shifts all hues by DEG; useful when a dominant hue '
                          'falls between BBC primaries (e.g. orange → red).'))
    ap.add_argument('--denoise', action='store_true', default=False,
                    help=('Prep 7: apply a 3x3 median filter to reduce noise '
                          'before dithering. Prevents source noise from being '
                          'encoded into the dither pattern.'))
    ap.add_argument('--posterise', type=int, default=0, metavar='N',
                    help=('Prep 8: reduce each channel to N bits (1-7; '
                          'default 0 = off). Quantises colours to N-bit steps, '
                          'creating a flat-colour graphic-art look. '
                          'Higher N = more colours retained.'))
    ap.add_argument('--smooth', type=float, default=0.0, metavar='F',
                    help=('Option 10: per-new-slot-change cost in pixel-frequency '
                          'units (default 0 = off). Subtracted from the gain when '
                          'the greedy considers changing a slot that has not been '
                          'changed yet this section. Discourages gratuitous slot '
                          'changes and reduces inter-section palette banding. '
                          'Values of 5-30 are typical; start at 10.'))
    ap.add_argument('--anneal', type=int, default=0, metavar='N',
                    help=('Option 2: simulated annealing step count (default 0 = off). '
                          'Replaces the greedy with SA: occasionally accepts '
                          'downhill moves to escape local optima, cooling to '
                          'greedy-like behaviour at the end. '
                          'Typical values: 100 (fast) to 500 (quality). '
                          'Overrides --beam and --restarts.'))
    ap.add_argument('--beam', type=int, default=1, metavar='N',
                    help=('Option 3: beam search width for the palette solver '
                          '(default 1 = greedy). Keeps the top N palette states '
                          'at each budget step instead of committing to one. '
                          'Values of 5-10 give substantially better coverage '
                          'at N× the solver cost per section.'))
    ap.add_argument('--auto-threshold', type=float,
                    default=_AUTO_DITHER_THRESHOLD, metavar='F',
                    help=('Option 9: luminance variance threshold for --dither auto '
                          f'(default {_AUTO_DITHER_THRESHOLD}). Sections with '
                          'per-pixel variance above this use Floyd-Steinberg; '
                          'sections below use ordered dither.'))
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress per-section progress output')
    args = ap.parse_args()

    process_image(args.input, args.output,
                  dither=args.dither, verbose=not args.quiet,
                  preview_path=args.preview, solver=args.solver,
                  resize=args.resize, randomness=args.randomness,
                  look_ahead=args.look_ahead,
                  chunk_size=args.chunk_size,
                  changes_per_row=args.changes,
                  restarts=args.restarts,
                  mixno=args.mixno,
                  sharpen=args.sharpen,
                  auto_threshold=args.auto_threshold,
                  beam=args.beam,
                  anneal=args.anneal,
                  smooth=args.smooth,
                  saturation=args.saturation,
                  autolevel=args.autolevel,
                  contrast=args.contrast,
                  brightness=args.brightness,
                  input_gamma=args.input_gamma,
                  hue=args.hue,
                  denoise=args.denoise,
                  posterise=args.posterise)


if __name__ == '__main__':
    main()
