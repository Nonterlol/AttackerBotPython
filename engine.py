#!/usr/bin/env python3
"""Python UCI engine using Stockfish 17.1's pretrained NNUE.

V17: the evaluation is Stockfish 17.1's main network (nn-1c0000000000.nnue,
HalfKAv2_hm, 2x3072 -> 16 -> 32 -> 1 with 8 PSQT buckets and 8 layer
stacks), replacing the Stockfish 13-era HalfKP net (nn-62ef826d1a6d).  The
network output matches Stockfish's Network::evaluate() bit for bit; the
centipawn score follows SF 17.1's Eval::evaluate() (complexity and material
scaling, rule-50 damping) and its win-rate normalisation (to_cp).
Setup, once:  python3 engine.py --setup
(downloads nn-1c0000000000.nnue, checks its sha256 and converts it to
sf17_ft.npy / sf17_layers.npz; the 139 MB weight file is memory-mapped, so
parallel engine processes share one copy).
V20: identity-preserving speedups (fused eval path, move-list cache,
batched frontier evaluation): same moves/scores/nodes as V19s, ~1% faster.
V19s: single-threaded base = V18.1 + optional Syzygy tablebases (folder
"syzygy" next to this file, or setoption SyzygyPath); no pondering.
V18.1: faster evaluation path (fused eval glue, activations between the
first two layers in plain Python, to_cp table); bit-identical search.
V18: search margins re-tuned for the new net, whose static eval is about
twice as accurate (eval-vs-search spread 76 vs 165 cp): RFP 120->65/ply,
futility 100+100d -> 60+60d, qsearch delta 200->110, aspiration 50->25.
SPRT vs V17 at 10+0.1: +33 Elo (+14..+52), 600 games.
V16: lazy NNUE accumulator updates and a generator-free move loop.
V15: dynamic time management.  V14: own bitboard board for the search.

Runtime requirements: python-chess and NumPy.  No PyTorch, TensorFlow, TFLite,
C++, or handmade material/PST evaluator is used.  The one exception is a small
"mop-up" term for bare-king endgames (KQK, KRK, KBBK, KBNK), where the NNUE
gives no direction toward mate.

Changes vs. the original version (see CHANGES at the bottom of the file for the
full list): eval now feeds the network [side-to-move, opponent] halves, the
accumulator is updated from the move instead of diffing piece maps, the dense
layers run as exact float32 BLAS calls, and the search gained a transposition
table, killer/history ordering, check-aware quiescence, ply-based mate scores,
2-fold repetition detection and proper UCI time handling.
"""
import gc
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # tiny matrices: threads only add overhead
import math
import os
import random
import sys
import time
import chess
import numpy as np
try:
    from numpy._core.umath import clip as _uclip      # NumPy >= 2
except ImportError:                                   # NumPy 1.x
    from numpy.core.umath import clip as _uclip

MATE_SCORE = 30000
MATE_BOUND = MATE_SCORE - 1000
INF = 99999
MAX_PLY = 128
TT_MAX_ENTRIES = 400_000
EVAL_CACHE_MAX = 500_000
MOVE_OVERHEAD = 0.03       # seconds kept in reserve per move
# Optional Syzygy endgame tablebases: put the .rtbw/.rtbz files in a folder
# named "syzygy" next to this script (or "setoption name SyzygyPath value DIR").
# Root: tablebase move choice; search: exact win/draw/loss right after a
# capture or pawn move enters tablebase territory.
TB_WIN = 25000            # below the mate range, far above any eval
# Dynamic time management (clock games only): the soft limit (when to stop
# starting new iterations) is scaled by best-move stability and score drops.
TM_MIN_DEPTH = 4
TM_STABILITY = [1.6, 1.25, 1.0, 0.85, 0.7]   # by iterations with an unchanged best move
TM_DROP_MIN = 25          # cp drop vs previous iteration that triggers extra time
TM_DROP_FULL = 100        # drop at which the full extra is given
TM_DROP_EXTRA = 0.8       # up to +80%
TM_HARD = 2.5             # hard limit = 2.5x target (was 2x), still <= 25% of clock

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = ROOT

WHITE, BLACK = chess.WHITE, chess.BLACK
PAWN, ROOK, KING, QUEEN = chess.PAWN, chess.ROOK, chess.KING, chess.QUEEN
NULL_MOVE = chess.Move.null()
# Null-move pruning: R = NMP_BASE_R + depth // NMP_DEPTH_DIV, tried from
# NMP_MIN_DEPTH plies of remaining depth.
NMP_MIN_DEPTH = 3
NMP_BASE_R = 2
NMP_DEPTH_DIV = 4
# Reverse futility pruning (static null move): at shallow depth in non-PV
# nodes, return if static eval - RFP_MARGIN * depth still beats beta.
RFP_MAX_DEPTH = 4
RFP_MARGIN = 65
# Quiescence: skip captures that cannot raise the score to alpha even if the
# captured piece is won for free (delta pruning) and captures that lose
# material by static exchange evaluation (SEE pruning).
DELTA_MARGIN = 110
# Late-move pruning (move-count pruning): at depth <= LMP_MAX_DEPTH in
# non-PV nodes, once LMP_BASE + depth^2 moves have been considered, the
# remaining quiet moves are skipped entirely.  DISABLED (0): measured on
# game positions it discarded the refuting move in 21-26% of the nodes
# refuted by a quiet move, because quiet-move ordering is not reliable
# enough yet.  A 1000-game test of the enabled version showed no gain.
LMP_MAX_DEPTH = 0
LMP_BASE = 3
# Futility pruning: at depth <= FUTILITY_MAX_DEPTH in non-PV nodes, quiet
# moves that don't give check are skipped when the static eval plus
# FUTILITY_BASE + FUTILITY_PER_PLY * depth still can't reach alpha.
FUTILITY_MAX_DEPTH = 2
FUTILITY_BASE = 60
FUTILITY_PER_PLY = 60
# Quiet-move ordering extras.  Measured before adding them: at depth 1, 44%
# of quiet refutations came after the late-move-pruning limit, typically
# a piece stepping out of an attack.
ESCAPE_BONUS = 1_000_000      # moving a piece out of an enemy pawn attack
PAWN_ATTACKED_MALUS = 1_000_000  # moving a piece onto an enemy pawn attack
COUNTERMOVE_SCORE = 3_900_000    # best quiet reply to the previous move
QUIET_CAP = 3_890_000            # quiets stay below killers (4_000_001)
BAD_CAPTURE_SCORE = 3_895_000   # SEE-losing captures: after countermove, before quiets
# Search refinements (each can be switched off):
QS_CHECKS = True        # first quiescence ply also tries quiet checking moves
ASPIRATION = True       # aspiration windows around the previous score
ASP_WINDOW = 25
ASP_MIN_DEPTH = 4
IMPROVING = True        # "improving" flag: static eval better than 2 plies ago
                        # (smaller RFP margin when improving).
IMPROVING_LMR = True    # one extra LMR ply when not improving.  Costs ~9 WAC
                        # positions at equal time but reaches depth 7 ~2.3x
                        # faster; SPRT at 10+0.1 vs V11: +68 Elo (+26..+113).
IIR = True              # internal iterative reduction: no TT move -> depth - 1
IIR_MIN_DEPTH = 4
# Late-move reductions: log(depth) * log(move number) formula, applied from
# LMR_MIN_DEPTH remaining plies to quiet moves from index LMR_MIN_MOVE on.
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 3
LMR_DIVISOR = 2.25
HISTORY_MAX = 3_000_000
# Batched frontier evaluation: at depth-1 nodes whose first move did not cut
# off, the children that will be searched are evaluated in one batched
# network pass (identical numbers, less NumPy call overhead).
BATCH_EVAL = True
# History with "gravity": h += bonus - h*|bonus|/HIST_LIMIT keeps |h| < HIST_LIMIT
# and lets old information fade.
HIST_LIMIT = 16384
HIST_BONUS = [min(32 * d * d + 64 * d, 4000) for d in range(64)]
LMR_GOOD_HISTORY = 2000
LMR_TABLE = [[0 if d == 0 or m == 0 else int(0.75 + math.log(d) * math.log(m) / LMR_DIVISOR)
              for m in range(64)] for d in range(64)]
BB_RANK_7, BB_RANK_2, BB_BACKRANKS = chess.BB_RANK_7, chess.BB_RANK_2, chess.BB_BACKRANKS
BB_FILE_A, BB_FILE_H, BB_ALL = chess.BB_FILE_A, chess.BB_FILE_H, chess.BB_ALL

# Move-ordering values indexed by piece type (1..6).  The king is treated as
# the cheapest attacker: if a king capture is legal it is always safe.
VICTIM = [0, 100, 320, 330, 500, 900, 0]
ATTACKER = [0, 1, 3, 3, 5, 9, 0]

# --------------------------------------------------------------------------
# HalfKAv2_hm feature indexing (Stockfish 17.1 networks)
#   index = (sq ^ orient) + piece_base + king_bucket
# Our colour codes: white = 1, black = 0 ("pov" = perspective colour).
# The king's side of the board is mirrored to files e-h; black's view is
# flipped vertically.  32 king buckets x 704 piece-square slots = 22528.
# --------------------------------------------------------------------------
NUM_FEATURES = 22528
PS_NB = 11 * 64
L1 = 3072
L1H = L1 // 2
PSQT_HI = L1             # accumulator columns 3072..3079: psqt >> 8 per bucket
PSQT_LO = L1 + 8         # columns 3080..3087: psqt & 255 per bucket
ACC_W = L1 + 16
_KB_ROW_WHITE = [28, 24, 20, 16, 12, 8, 4, 0]     # bucket row by rank (white view)
# KORIENT[pov][ksq], KBUCKET[pov][ksq]
KORIENT = [[(63 if (s & 7) < 4 else 56) for s in range(64)],     # black
           [(7 if (s & 7) < 4 else 0) for s in range(64)]]       # white
KBUCKET = [[((28 - _KB_ROW_WHITE[s >> 3]) + ((s & 7) if (s & 7) < 4 else 7 - (s & 7))) * PS_NB
            for s in range(64)],
           [(_KB_ROW_WHITE[s >> 3] + ((s & 7) if (s & 7) < 4 else 7 - (s & 7))) * PS_NB
            for s in range(64)]]
_PS_OWN = [0, 0, 128, 256, 384, 512, 640]
_PS_ENEMY = [0, 64, 192, 320, 448, 576, 640]
# FBASE[pov][ksq][color][pt] = piece_base + king_bucket
FBASE = [[[[(_PS_OWN[pt] if color == pov else _PS_ENEMY[pt]) + KBUCKET[pov][k]
            for pt in range(7)] for color in (0, 1)] for k in range(64)] for pov in (0, 1)]
# Stockfish's evaluation-side material values (Eval::evaluate, UCI to_cp).
SF_NPM = [0, 0, 781, 825, 1276, 2538, 0]

# --------------------------------------------------------------------------
# Zobrist hashing (incrementally updated in NNUEState.push).
# Python's hash() of the python-chess transposition-key tuple is NOT safe as
# a position key: int hashes are taken modulo 2**61 - 1, so bitboards with a
# piece on f8/g8/h8 collide with the same piece on a1/b1/c1.
# --------------------------------------------------------------------------
_zrng = random.Random(0x5EED)
ZP = [[[_zrng.getrandbits(64) for _ in range(64)] for _ in range(7)] for _ in range(2)]  # [color][pt][sq]
Z_SIDE = _zrng.getrandbits(64)
Z_EP = [_zrng.getrandbits(64) for _ in range(64)]
Z_CASTLE_SQ = [_zrng.getrandbits(64) for _ in range(64)]
_zc_cache = {}


def z_castle(rights):
    v = _zc_cache.get(rights)
    if v is None:
        v = 0
        bb = rights
        while bb:
            low = bb & -bb
            v ^= Z_CASTLE_SQ[low.bit_length() - 1]
            bb ^= low
        _zc_cache[rights] = v
    return v


def zobrist(board):
    """Key from scratch (used by sync() and the tests)."""
    z = 0
    for color in (WHITE, BLACK):
        occ = board.occupied_co[color]
        for pt, bb in ((1, board.pawns), (2, board.knights), (3, board.bishops),
                       (4, board.rooks), (5, board.queens), (6, board.kings)):
            for sq in chess.scan_forward(bb & occ):
                z ^= ZP[color][pt][sq]
    if board.turn == BLACK:
        z ^= Z_SIDE
    z ^= z_castle(board.castling_rights)
    if board.ep_square is not None:
        z ^= Z_EP[board.ep_square]
    return z


BATCH_MAX = 256
NET_FILE = 'nn-1c0000000000.nnue'          # Stockfish 17.1 default (big) network
NET_URLS = ('https://tests.stockfishchess.org/api/nn/' + NET_FILE,
            'https://github.com/official-stockfish/networks/raw/master/' + NET_FILE)
NET_URL = NET_URLS[0]
NET_SHA_PREFIX = '1c0000000000'          # first 12 hex digits of its sha256
NPY_FT = 'sf17_ft.npy'                      # converted feature transformer (+psqt)
NPY_REST = 'sf17_layers.npz'                # converted dense layers


def _leb128(data, pos, count, dtype):
    """Decode one Stockfish LEB128 block (vectorised, in chunks)."""
    magic = b'COMPRESSED_LEB128'
    if data[pos:pos + len(magic)] != magic:
        raise ValueError('bad network file (LEB128 magic missing)')
    pos += len(magic)
    nbytes = int.from_bytes(data[pos:pos + 4], 'little')
    pos += 4
    b = np.frombuffer(data, dtype=np.uint8, count=nbytes, offset=pos)
    out = np.empty(count, dtype=dtype)
    done = 0
    start_byte = 0
    CH = 1 << 22
    while start_byte < nbytes:
        chunk = b[start_byte:start_byte + CH]
        end = np.flatnonzero((chunk & 0x80) == 0)
        if len(end) == 0:
            raise ValueError('bad LEB128 data')
        chunk = chunk[:end[-1] + 1]
        st = np.empty_like(end)
        st[0] = 0
        st[1:] = end[:-1] + 1
        ln = end - st + 1
        p = np.arange(len(chunk), dtype=np.int64) - np.repeat(st, ln)
        vals = np.add.reduceat((chunk & 0x7F).astype(np.int64) << (7 * p), st)
        neg = (chunk[end] & 0x40) != 0
        vals -= np.where(neg, np.int64(1) << (7 * ln), 0)
        out[done:done + len(vals)] = vals
        done += len(vals)
        start_byte += len(chunk)
    if done != count:
        raise ValueError('bad LEB128 count')
    return out, pos + nbytes


def download_network(directory):
    """Download nn-1c0000000000.nnue (75 MB, CC0) and verify its sha256."""
    import hashlib
    import urllib.request
    dest = os.path.join(directory, NET_FILE)
    for url in NET_URLS:
        tmp = dest + '.part'
        try:
            print('Downloading %s ...' % url, flush=True)
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    f.write(chunk)
            if not h.hexdigest().startswith(NET_SHA_PREFIX):
                raise ValueError('checksum mismatch')
            os.replace(tmp, dest)
            return dest
        except Exception as e:                          # try the next mirror
            print('  failed: %s' % e, flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
    raise SystemExit('Could not download %s; download it manually from %s and put it '
                     'next to this script.' % (NET_FILE, NET_URLS[0]))


def convert_network(nnue_path, outdir):
    """One-time conversion of nn-1c0000000000.nnue into NumPy files.
    The psqt weights ride along as 16 extra int16 accumulator columns
    (high and low byte per bucket) so one row add updates everything."""
    data = open(nnue_path, 'rb').read()
    u32 = lambda p: int.from_bytes(data[p:p + 4], 'little')
    if u32(0) != 0x7AF32F20:
        raise ValueError(nnue_path + ' is not a Stockfish 17 network')
    pos = 12 + u32(8) + 4                       # version, hash, description, FT hash
    bias, pos = _leb128(data, pos, L1, np.int16)
    w, pos = _leb128(data, pos, NUM_FEATURES * L1, np.int16)
    psqt, pos = _leb128(data, pos, NUM_FEATURES * 8, np.int32)
    psqt = psqt.reshape(NUM_FEATURES, 8)
    ft = np.empty((NUM_FEATURES + 1, ACC_W), dtype=np.int16)   # last row = biases
    ft[:NUM_FEATURES, :L1] = w.reshape(NUM_FEATURES, L1)
    del w
    ft[:NUM_FEATURES, PSQT_HI:PSQT_HI + 8] = psqt >> 8
    ft[:NUM_FEATURES, PSQT_LO:PSQT_LO + 8] = psqt & 255
    ft[NUM_FEATURES, :L1] = bias
    ft[NUM_FEATURES, L1:] = 0
    rest = {}
    pad0 = (L1 + 31) // 32 * 32
    for k in range(8):
        pos += 4                                  # layer-stack hash
        def arr(dt, n):
            nonlocal pos
            a = np.frombuffer(data, dtype=np.dtype(dt).newbyteorder('<'), count=n, offset=pos)
            pos += n * np.dtype(dt).itemsize
            return a.astype(dt)
        rest['b0_%d' % k] = arr(np.int32, 16)
        rest['w0_%d' % k] = arr(np.int8, 16 * pad0).reshape(16, pad0)[:, :L1]
        rest['b1_%d' % k] = arr(np.int32, 32)
        rest['w1_%d' % k] = arr(np.int8, 32 * 32).reshape(32, 32)[:, :30]
        rest['b2_%d' % k] = arr(np.int32, 1)
        rest['w2_%d' % k] = arr(np.int8, 32)
    if pos != len(data):
        raise ValueError('unexpected network file size')
    tmp = '.%d.tmp' % os.getpid()
    with open(os.path.join(outdir, NPY_FT + tmp), 'wb') as fh:
        np.save(fh, ft)
    with open(os.path.join(outdir, NPY_REST + tmp), 'wb') as fh:
        np.savez(fh, **rest)
    os.replace(os.path.join(outdir, NPY_REST + tmp), os.path.join(outdir, NPY_REST))
    os.replace(os.path.join(outdir, NPY_FT + tmp), os.path.join(outdir, NPY_FT))


def _mmap_npy(path):
    """Plain ndarray backed by a read-only memory map: the 139 MB weight
    file is shared between engine processes through the page cache, and
    (unlike np.memmap) row indexing returns ordinary arrays."""
    import mmap
    with open(path, 'rb') as fh:
        version = np.lib.format.read_magic(fh)
        fmt = np.lib.format
        hdr = fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0
        shape, fortran, dtype = hdr(fh)
        offset = fh.tell()
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    a = np.frombuffer(mm, dtype=dtype, count=int(np.prod(shape)), offset=offset)
    return a.reshape(shape)


class SF17Network:
    """Stockfish 17.1 big network (HalfKAv2_hm, 2x3072 -> 16 -> 32 -> 1,
    8 PSQT buckets and 8 layer stacks).  Output matches Stockfish's
    Network::evaluate() bit for bit (verified against a patched SF 17.1)."""

    def __init__(self, directory=MODEL_DIR):
        ft_path = os.path.join(directory, NPY_FT)
        if not os.path.exists(ft_path) or not os.path.exists(os.path.join(directory, NPY_REST)):
            src = os.path.join(directory, NET_FILE)
            if not os.path.exists(src):
                raise FileNotFoundError(
                    'Network not set up.  Run once:  python3 %s --setup\n'
                    '(downloads %s from %s and converts it)'
                    % (os.path.basename(__file__), NET_FILE, NET_URL))
            convert_network(src, directory)
        self.ft = _mmap_npy(ft_path)
        if self.ft.shape != (NUM_FEATURES + 1, ACC_W):
            raise ValueError('unexpected %s shape %r' % (NPY_FT, self.ft.shape))
        self.bias = np.array(self.ft[NUM_FEATURES])
        r = np.load(os.path.join(directory, NPY_REST))
        # fc_0 in float32 with its bias as an extra input (fixed at 1) and the
        # ">> 6" folded in (dividing by 64 is exact): every partial sum is an
        # integer/64 below 2**24/64 in magnitude (checked below), so the BLAS
        # result is exact.  fc_1 likewise in float64.
        self.w0, self.w1, self.w2, self.b2 = [], [], [], []
        for k in range(8):
            w0 = r['w0_%d' % k].astype(np.float64)
            b0 = r['b0_%d' % k].astype(np.float64)
            if (np.abs(w0).sum(axis=1) * 127 + np.abs(b0)).max() >= 2 ** 24:
                raise ValueError('fc_0 weights too large for exact float32')
            self.w0.append(np.ascontiguousarray(np.hstack((w0, b0[:, None])) / 64.0, dtype=np.float32))
            w1 = r['w1_%d' % k].astype(np.float64)
            b1 = r['b1_%d' % k].astype(np.float64)
            self.w1.append(np.ascontiguousarray(np.hstack((w1, b1[:, None])) / 64.0))
            self.w2.append(r['w2_%d' % k].astype(np.float64))
            self.b2.append(int(r['b2_%d' % k][0]))
        self._t = np.empty(L1, dtype=np.int16)
        self._x = np.ones(L1 + 1, dtype=np.int16)        # last slot = bias input
        self._xs = self._x[:L1]
        self._hl = [1.0] * 31                            # last slot = bias input
        self._hn = np.ones(31, dtype=np.float64)          # same, as an array
        # batched pass (prefetch of a node's children)
        self.w0T = [np.ascontiguousarray(w.T) for w in self.w0]
        self.w1T = [np.ascontiguousarray(w.T) for w in self.w1]
        self._bA = np.empty((BATCH_MAX, 2, L1), dtype=np.int16)
        self._bP = np.empty((BATCH_MAX, 2, L1H), dtype=np.int16)
        self._bX = np.ones((BATCH_MAX, L1 + 1), dtype=np.float32)
        self._bH = np.ones((BATCH_MAX, 31), dtype=np.float64)

    def propagate(self, s, n, bucket):
        """Positional output (internal units, before / OutputScale) for
        accumulators ordered [side to move, other]."""
        t, x = self._t, self._x
        clip, mul = _uclip, np.multiply
        # pairwise clipped product: clamp(a,0,127) * clamp(b,0,127) // 128
        # (int16 is enough: 127 * 127 < 32768)
        clip(s[:L1], 0, 127, out=t)
        mul(t[:L1H], t[L1H:], out=x[:L1H])
        clip(n[:L1], 0, 127, out=t)
        mul(t[:L1H], t[L1H:], out=x[L1H:L1])
        np.right_shift(self._xs, 7, out=self._xs)
        o = np.dot(self.w0[bucket], x).tolist()        # fc_0 / 64 (exact floats)
        # The 15+15 activations in plain Python: cheaper than ~9 tiny NumPy
        # calls.  v*v is exact in double precision (v has <= 24 bits).
        h = self._hl
        for i in range(15):
            v = o[i]
            q = (v * v) // 128.0                          # (fc0^2 >> 19)
            h[i] = q if q < 127.0 else 127.0
            q = v // 1.0                                  # fc0 >> 6
            h[15 + i] = 0.0 if q < 0.0 else (q if q < 127.0 else 127.0)
        o1 = np.dot(self.w1[bucket], h)                   # (fc_1 + bias) / 64
        np.floor(o1, out=o1)
        clip(o1, 0.0, 127.0, out=o1)
        out = int(np.dot(self.w2[bucket], o1)) + self.b2[bucket]
        fwd = int(o[15] * 64) * 9600                      # 600 * OutputScale
        fwd = fwd // 8128 if fwd >= 0 else -((-fwd) // 8128)   # / (127 * 64), truncating
        return out + fwd


    def propagate_batch(self, S, N, bucket):
        """propagate() for K positions at once (S, N: (K, ACC_W) int16 views,
        side to move / other).  Bit-identical to propagate()."""
        K = S.shape[0]
        A = self._bA[:K]
        A[:, 0] = S[:, :L1]
        A[:, 1] = N[:, :L1]
        _uclip(A, 0, 127, out=A)
        P = self._bP[:K]
        np.multiply(A[:, :, :L1H], A[:, :, L1H:], out=P)
        np.right_shift(P, 7, out=P)
        X = self._bX[:K]
        X[:, :L1] = P.reshape(K, L1)
        O = np.dot(X, self.w0T[bucket]).astype(np.float64)    # (K,16) fc_0 / 64
        H = self._bH[:K]
        o15 = O[:, :15]
        h = H[:, :15]
        np.multiply(o15, o15, out=h)
        np.floor_divide(h, 128.0, out=h)
        np.minimum(h, 127.0, out=h)
        h = H[:, 15:30]
        np.floor(o15, out=h)
        _uclip(h, 0.0, 127.0, out=h)
        O1 = np.dot(H, self.w1T[bucket])
        np.floor(O1, out=O1)
        _uclip(O1, 0.0, 127.0, out=O1)
        out = np.dot(O1, self.w2[bucket]).tolist()
        o15 = O[:, 15].tolist()
        b2 = self.b2[bucket]
        res = []
        for j in range(K):
            fwd = int(o15[j] * 64) * 9600
            fwd = fwd // 8128 if fwd >= 0 else -((-fwd) // 8128)
            res.append(int(out[j]) + b2 + fwd)
        return res


def _tdiv(a, b):
    """C++ integer division (truncates toward zero), b > 0."""
    return a // b if a >= 0 else -((-a) // b)


# ==========================================================================
# Fast board (replaces python-chess in the search; python-chess is still used
# for UCI parsing and to build the root position).
#   * Moves are ints: from | to << 6 | promotion << 12 | flags << 15
#     (flags: 1 = castling, 2 = en passant); m & 0x7FFF identifies a move.
#   * State: 8 bitboards [black occ, white occ, P, N, B, R, Q, K] and a
#     64-entry mailbox (piece_type + 8 * color, white = 1).  make() mutates
#     fresh copies of both lists (C-level slices); unmake() restores the old
#     list objects, so there is no incremental undo logic to get wrong.
#   * Zobrist key and NNUE accumulators are updated inside make().
#   * Move generation reproduces python-chess's exact order (incl. evasions);
#     verified by perft and by move-list comparison on 25k random positions.
# ==========================================================================
BB_SQ = chess.BB_SQUARES
KNIGHT_ATT = chess.BB_KNIGHT_ATTACKS
KING_ATT = chess.BB_KING_ATTACKS
PAWN_ATT = chess.BB_PAWN_ATTACKS          # [color][sq]
DIAG_MASK = chess.BB_DIAG_MASKS
DIAG_ATT = chess.BB_DIAG_ATTACKS
FILE_MASK = chess.BB_FILE_MASKS
FILE_ATT = chess.BB_FILE_ATTACKS
RANK_MASK = chess.BB_RANK_MASKS
RANK_ATT = chess.BB_RANK_ATTACKS
RAYS = chess.BB_RAYS
BB_ALL = chess.BB_ALL
BB_LIGHT, BB_DARK = chess.BB_LIGHT_SQUARES, chess.BB_DARK_SQUARES
BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6 = chess.BB_RANK_3, chess.BB_RANK_4, chess.BB_RANK_5, chess.BB_RANK_6
BB_RANKS = chess.BB_RANKS
BB_BACKRANKS = chess.BB_BACKRANKS

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, 0
FLAG_CASTLE = 1 << 15
FLAG_EP = 2 << 15
MOVE_ID = 0x7FFF

# BETWEEN[a][b]: squares strictly between a and b on a common line (else 0).
BETWEEN = [[0] * 64 for _ in range(64)]
for _a in range(64):
    for _b in range(64):
        _bb = RAYS[_a][_b] & ((BB_ALL << _a) ^ (BB_ALL << _b))
        BETWEEN[_a][_b] = _bb & (_bb - 1)

# Castling rights as 4 bits; CASTLE_KEEP[sq] clears the rights touched by a
# move from/to that square.  Matching python-chess's rook-square bitboard.
WK, WQ, BK, BQ = 1, 2, 4, 8
CASTLE_RIGHT_SQ = {WK: 7, WQ: 0, BK: 63, BQ: 56}
CASTLE_KEEP = [15] * 64
CASTLE_KEEP[7] &= ~WK
CASTLE_KEEP[0] &= ~WQ
CASTLE_KEEP[63] &= ~BK
CASTLE_KEEP[56] &= ~BQ
CASTLE_KEEP[4] &= ~(WK | WQ)
CASTLE_KEEP[60] &= ~(BK | BQ)


_MOVES_CACHE = {}     # (from << 64 | target set) -> move list in generation order


def diag(sq, occ):
    return DIAG_ATT[sq][DIAG_MASK[sq] & occ]


def orth(sq, occ):
    return FILE_ATT[sq][FILE_MASK[sq] & occ] | RANK_ATT[sq][RANK_MASK[sq] & occ]


def move_from_chess(m, board):
    """python-chess Move (on python-chess board) -> int move."""
    v = m.from_square | (m.to_square << 6) | ((m.promotion or 0) << 12)
    if board.is_castling(m):
        v |= FLAG_CASTLE
    elif board.is_en_passant(m):
        v |= FLAG_EP
    return v


def move_uci(m):
    s = chess.SQUARE_NAMES[m & 63] + chess.SQUARE_NAMES[(m >> 6) & 63]
    p = (m >> 12) & 7
    return s + 'nbrq'[p - 2] if p else s


def move_to_chess(m):
    return chess.Move(m & 63, (m >> 6) & 63, ((m >> 12) & 7) or None)


class Position:
    """Board + zobrist + NNUE accumulators.  `hooks` provides zobrist tables
    and the NNUE feature indexing so this module has no engine imports."""

    __slots__ = ('bb', 'sq', 'turn', 'castling', 'ep', 'halfmove', 'z', 'keys',
                 'stack', 'accs', 'pargs', 'nn', 'zt')

    def __init__(self, pcboard, nn, zt):
        """Build from a python-chess board, replaying its move stack so that
        repetition history is available."""
        self.nn = nn          # NNUE helper (refresh, finny cache, weights)
        self.zt = zt          # zobrist tables
        root = pcboard.root()
        self._set_from_chess(root)
        replay = root.copy(stack=False)
        for m in pcboard.move_stack:
            if m:
                self.make(move_from_chess(m, replay))
            else:
                self.make_null()
            replay.push(m)

    # -- setup ---------------------------------------------------------------
    def _set_from_chess(self, b):
        ow, ob = b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK]
        self.bb = [ob, ow, b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings]
        sq = [0] * 64
        for s, p in b.piece_map().items():
            sq[s] = p.piece_type + (8 if p.color else 0)
        self.sq = sq
        self.turn = 1 if b.turn else 0
        rights = b.clean_castling_rights()
        c = 0
        for bit, s in CASTLE_RIGHT_SQ.items():
            if rights & BB_SQ[s]:
                c |= bit
        self.castling = c
        self.ep = b.ep_square if b.ep_square is not None else -1
        self.halfmove = b.halfmove_clock
        self.stack = []
        self.z = self.zobrist_full()
        self.keys = [self.z ^ (self.zt.ep[self.ep] if self.ep >= 0 else 0)]
        # Lazy NNUE accumulators: accs[i] = (white, black) after i moves from
        # the start position, or None if not computed yet; pargs[i] holds what
        # move i changed.  make() only records; acc() materialises on demand,
        # so positions that are never evaluated never pay for the update.
        self.accs = [(self.nn.refresh(self, 1), self.nn.refresh(self, 0))]
        self.pargs = []

    def zobrist_full(self):
        zt = self.zt
        z = 0
        for s in range(64):
            p = self.sq[s]
            if p:
                z ^= zt.piece[p >> 3][p & 7][s]
        if not self.turn:
            z ^= zt.side
        z ^= zt.castle[self.castling]
        if self.ep >= 0:
            z ^= zt.ep[self.ep]
        return z

    # -- queries ---------------------------------------------------------------
    def occupied(self):
        return self.bb[0] | self.bb[1]

    def king_sq(self, color):
        return (self.bb[7] & self.bb[color]).bit_length() - 1

    def attackers(self, color, s, occ):
        """python-chess attackers_mask(color, s, occ): pieces of `color` that
        attack s, with sliders seeing through `occ` (pieces themselves are
        taken from the current board, like python-chess).  Slider tables are
        only consulted when a slider of `color` is on one of s's lines."""
        bb = self.bb
        mine = bb[color]
        q = bb[6]
        a = ((KNIGHT_ATT[s] & bb[3]) | (KING_ATT[s] & bb[7]) | (PAWN_ATT[1 - color][s] & bb[2])) & mine
        dq = (bb[4] | q) & mine
        if dq & DIAG_ATT[s][0]:
            a |= DIAG_ATT[s][DIAG_MASK[s] & occ] & dq
        rq = (bb[5] | q) & mine
        if rq & (FILE_ATT[s][0] | RANK_ATT[s][0]):
            a |= (FILE_ATT[s][FILE_MASK[s] & occ] | RANK_ATT[s][RANK_MASK[s] & occ]) & rq
        return a

    def is_check(self):
        # Specialised attackers() for the side to move's king: the slider
        # tables (dict lookups) are only consulted when an enemy slider
        # stands on one of the king's empty-board lines.
        bb = self.bb
        us = self.turn
        them = bb[1 - us]
        k = (bb[7] & bb[us]).bit_length() - 1
        if KNIGHT_ATT[k] & bb[3] & them or PAWN_ATT[us][k] & bb[2] & them:
            return True
        q = bb[6]
        occ = bb[0] | bb[1]
        dq = (bb[4] | q) & them
        if dq & DIAG_ATT[k][0] and DIAG_ATT[k][DIAG_MASK[k] & occ] & dq:
            return True
        rq = (bb[5] | q) & them
        if rq & (FILE_ATT[k][0] | RANK_ATT[k][0]) and \
                (FILE_ATT[k][FILE_MASK[k] & occ] | RANK_ATT[k][RANK_MASK[k] & occ]) & rq:
            return True
        return False

    def slider_blockers(self, king):
        bb = self.bb
        rq = bb[5] | bb[6]
        bq = bb[4] | bb[6]
        snipers = ((RANK_ATT[king][0] & rq) | (FILE_ATT[king][0] & rq) |
                   (DIAG_ATT[king][0] & bq)) & bb[1 - self.turn]
        occ = bb[0] | bb[1]
        blockers = 0
        while snipers:
            s = snipers.bit_length() - 1
            snipers ^= 1 << s
            b = BETWEEN[king][s] & occ
            if b and not (b & (b - 1)):
                blockers |= b
        return blockers & bb[self.turn]

    def is_safe(self, king, blockers, m):
        """python-chess _is_safe (valid when not in check, and for evasions)."""
        f = m & 63
        if f == king:
            if m & FLAG_CASTLE:
                return True
            t = (m >> 6) & 63
            return not self.attackers(1 - self.turn, t, self.bb[0] | self.bb[1])
        if m & FLAG_EP:
            return self._ep_legal(m)
        return not (blockers >> f & 1) or bool(RAYS[f][(m >> 6) & 63] >> king & 1)

    def _ep_legal(self, m):
        """Exact legality of an en passant capture: king not attacked after it."""
        f, t = m & 63, (m >> 6) & 63
        us = self.turn
        cap = t ^ 8
        occ = (self.bb[0] | self.bb[1]) ^ (1 << f) ^ (1 << cap) | (1 << t)
        k = self.king_sq(us)
        bb = self.bb
        them = bb[1 - us] & ~(1 << cap)
        q = bb[6]
        att = ((KNIGHT_ATT[k] & bb[3]) | (KING_ATT[k] & bb[7]) |
               (PAWN_ATT[us][k] & bb[2] & ~(1 << cap)) |
               (diag(k, occ) & (bb[4] | q)) | (orth(k, occ) & (bb[5] | q))) & them
        return not att

    def checkers(self):
        k = self.king_sq(self.turn)
        return self.attackers(1 - self.turn, k, self.bb[0] | self.bb[1])

    def is_insufficient(self):
        bb = self.bb
        if bb[2] | bb[5] | bb[6]:
            return False
        for color in (1, 0):
            mine = bb[color]
            if mine & bb[3]:
                if not (bin(mine).count('1') <= 2 and not (bb[1 - color] & ~bb[7] & ~bb[6])):
                    return False
            elif mine & bb[4]:
                bishops = bb[4]
                same = (not bishops & BB_DARK) or (not bishops & BB_LIGHT)
                if not (same and not bb[2] and not bb[3]):
                    return False
        return True

    # -- move generation (python-chess order) -----------------------------------
    def gen_pseudo(self, from_mask=BB_ALL, to_mask=BB_ALL):
        bb = self.bb
        us = self.turn
        ours = bb[us]
        theirs = bb[1 - us]
        occ = ours | theirs
        sq = self.sq
        out = []
        app = out.append
        pawns = bb[2]
        non_pawns = ours & ~pawns & from_mask
        while non_pawns:
            f = non_pawns.bit_length() - 1
            non_pawns ^= 1 << f
            pt = sq[f] & 7
            if pt == KNIGHT:
                a = KNIGHT_ATT[f]
            elif pt == BISHOP:
                a = DIAG_ATT[f][DIAG_MASK[f] & occ]
            elif pt == ROOK:
                a = FILE_ATT[f][FILE_MASK[f] & occ] | RANK_ATT[f][RANK_MASK[f] & occ]
            elif pt == QUEEN:
                a = (DIAG_ATT[f][DIAG_MASK[f] & occ] | FILE_ATT[f][FILE_MASK[f] & occ] |
                     RANK_ATT[f][RANK_MASK[f] & occ])
            else:
                a = KING_ATT[f]
            a &= ~ours & to_mask
            if a:
                key = (f << 64) | a
                ml = _MOVES_CACHE.get(key)
                if ml is None:
                    ml = []
                    while a:
                        t = a.bit_length() - 1
                        a ^= 1 << t
                        ml.append(f | (t << 6))
                    if len(_MOVES_CACHE) > 200_000:
                        _MOVES_CACHE.clear()
                    _MOVES_CACHE[key] = ml
                out += ml
        if from_mask & bb[7] and to_mask == BB_ALL and self.castling:
            self._gen_castling(app, occ)
        p = pawns & ours & from_mask
        if not p:
            return out
        caps = p
        patt = PAWN_ATT[us]
        while caps:
            f = caps.bit_length() - 1
            caps ^= 1 << f
            tg = patt[f] & theirs & to_mask
            while tg:
                t = tg.bit_length() - 1
                tg ^= 1 << t
                base = f | (t << 6)
                if t >= 56 or t < 8:
                    app(base | (5 << 12))
                    app(base | (4 << 12))
                    app(base | (3 << 12))
                    app(base | (2 << 12))
                else:
                    app(base)
        if us:
            single = (p << 8) & ~occ & BB_ALL
            double = (single << 8) & ~occ & (BB_RANK_3 | BB_RANK_4)
            step = -8
        else:
            single = (p >> 8) & ~occ
            double = (single >> 8) & ~occ & (BB_RANK_6 | BB_RANK_5)
            step = 8
        single &= to_mask
        double &= to_mask
        while single:
            t = single.bit_length() - 1
            single ^= 1 << t
            base = (t + step) | (t << 6)
            if t >= 56 or t < 8:
                app(base | (5 << 12))
                app(base | (4 << 12))
                app(base | (3 << 12))
                app(base | (2 << 12))
            else:
                app(base)
        while double:
            t = double.bit_length() - 1
            double ^= 1 << t
            app((t + 2 * step) | (t << 6))
        if self.ep >= 0:
            self._gen_ep(app, from_mask, to_mask)
        return out

    def _gen_ep(self, app, from_mask=BB_ALL, to_mask=BB_ALL):
        e = self.ep
        if e < 0 or not (to_mask >> e & 1) or (self.bb[0] | self.bb[1]) >> e & 1:
            return
        us = self.turn
        cap = (self.bb[2] & self.bb[us] & from_mask & PAWN_ATT[1 - us][e] &
               BB_RANKS[4 if us else 3])
        while cap:
            f = cap.bit_length() - 1
            cap ^= 1 << f
            app(f | (e << 6) | FLAG_EP)

    def _gen_castling(self, app, occ):
        us = self.turn
        if us:
            king, rights = 4, ((WK, 7), (WQ, 0))
        else:
            king, rights = 60, ((BK, 63), (BQ, 56))
        if self.sq[king] != KING + 8 * us:
            return
        kbit = 1 << king
        for right, rook in rights:          # h-side first, like scan_reversed
            if not self.castling & right or self.sq[rook] != ROOK + 8 * us:
                continue
            a_side = rook < king
            king_to = king - 2 if a_side else king + 2
            rook_to = king - 1 if a_side else king + 1
            king_path = BETWEEN[king][king_to]
            rook_path = BETWEEN[rook][rook_to]
            rbit = 1 << rook
            if (occ ^ kbit ^ rbit) & (king_path | rook_path | (1 << king_to) | (1 << rook_to)):
                continue
            if self._attacked_for_king(king_path | kbit, occ ^ kbit):
                continue
            if self._attacked_for_king(1 << king_to, occ ^ kbit ^ rbit ^ (1 << rook_to)):
                continue
            app(king | (king_to << 6) | FLAG_CASTLE)

    def _attacked_for_king(self, path, occ):
        them = 1 - self.turn
        while path:
            s = path.bit_length() - 1
            path ^= 1 << s
            if self.attackers(them, s, occ):
                return True
        return False

    def gen_captures(self):
        """python-chess generate_pseudo_legal_captures() order."""
        out = self.gen_pseudo(BB_ALL, self.bb[1 - self.turn])
        if self.ep >= 0:
            self._gen_ep(out.append)
        return out

    def gen_evasions(self, king, checkers):
        """python-chess _generate_evasions() order (pseudo-legal; filter with
        is_safe)."""
        bb = self.bb
        sliders = checkers & (bb[4] | bb[5] | bb[6])
        attacked = 0
        s = sliders
        while s:
            c = s.bit_length() - 1
            s ^= 1 << c
            attacked |= RAYS[king][c] & ~(1 << c)
        out = []
        app = out.append
        k = KING_ATT[king] & ~bb[self.turn] & ~attacked
        while k:
            t = k.bit_length() - 1
            k ^= 1 << t
            app(king | (t << 6))
        checker = checkers.bit_length() - 1
        if (1 << checker) == checkers:
            target = BETWEEN[king][checker] | checkers
            out.extend(self.gen_pseudo(~bb[7] & BB_ALL, target))
            if self.ep >= 0 and not (target >> self.ep & 1):
                last_double = self.ep + (-8 if self.turn else 8)
                if last_double == checker:
                    self._gen_ep(app)
        return out

    def legal_moves(self):
        king = self.king_sq(self.turn)
        ch = self.checkers()
        if ch:
            cand = self.gen_evasions(king, ch)
        else:
            cand = self.gen_pseudo()
        blockers = self.slider_blockers(king)
        return [m for m in cand if self.is_safe(king, blockers, m)]

    def is_pseudo_legal(self, m):
        f, t = m & 63, (m >> 6) & 63
        p = self.sq[f]
        if not p or (p >> 3) != self.turn:
            return False
        return m in self.gen_pseudo(1 << f, BB_ALL if p & 7 == KING else 1 << t)

    # -- static exchange ---------------------------------------------------------
    def see(self, m):
        """Same swap-list SEE as the engine used with python-chess."""
        f, t = m & 63, (m >> 6) & 63
        occ = (self.bb[0] | self.bb[1]) ^ (1 << f)
        if m & FLAG_EP:
            victim = PAWN
            occ ^= 1 << (t ^ 8)
        else:
            victim = self.sq[t] & 7
        gain = [SEE_VALUE[victim]]
        on_square = SEE_VALUE[self.sq[f] & 7]
        side = 1 - self.turn
        bb = self.bb
        piece_bbs = (bb[2], bb[3], bb[4], bb[5], bb[6], bb[7])
        attackers = self.attackers
        while True:
            att = attackers(side, t, occ) & occ & bb[side]
            if not att:
                break
            for pt in range(6):
                b = att & piece_bbs[pt]
                if b:
                    break
            if pt == 5:
                if attackers(1 - side, t, occ ^ (b & -b)) & occ & bb[1 - side]:
                    break
            gain.append(on_square - gain[-1])
            on_square = SEE_VALUE[pt + 1]
            occ ^= b & -b
            side = 1 - side
        for d in range(len(gain) - 1, 0, -1):
            gain[d - 1] = -max(-gain[d - 1], gain[d])
        return gain[0]

    def gives_check(self, m):
        """Exact: does m (pseudo-legal) give check?  Computed on hypothetical
        bitboards instead of making the move."""
        if m & (FLAG_CASTLE | FLAG_EP):
            self.make(m, nnue=False)
            r = self.is_check()
            self.unmake()
            return r
        bb = self.bb
        us = self.turn
        f, t = m & 63, (m >> 6) & 63
        new_pt = (m >> 12) & 7 or (self.sq[f] & 7)
        k = (bb[7] & bb[1 - us]).bit_length() - 1
        fb, tb = 1 << f, 1 << t
        occ = ((bb[0] | bb[1]) & ~fb) | tb
        mine = (bb[us] & ~fb) | tb
        # Our piece bitboards after the move (mover leaves f, lands on t as new_pt).
        own = [0] * 7
        for x in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
            v = bb[x + 1] & ~fb & ~tb
            if x == new_pt:
                v |= tb
            own[x] = v & mine
        if KNIGHT_ATT[k] & own[KNIGHT] or PAWN_ATT[1 - us][k] & own[PAWN]:
            return True
        q = own[QUEEN]
        return bool(diag(k, occ) & (own[BISHOP] | q) or orth(k, occ) & (own[ROOK] | q))

    # -- make / unmake -------------------------------------------------------------
    def make(self, m, nnue=True):
        bb_old = self.bb
        sq_old = self.sq
        bb = bb_old[:]
        sq = sq_old[:]
        f, t = m & 63, (m >> 6) & 63
        promo = (m >> 12) & 7
        us = self.turn
        them = 1 - us
        piece = sq[f]
        pt = piece & 7
        zt = self.zt
        zp = zt.piece
        z = self.z ^ zt.side
        ep_old = self.ep
        if ep_old >= 0:
            z ^= zt.ep[ep_old]
        castling_old = self.castling
        self.stack.append((bb_old, sq_old, castling_old, ep_old, self.halfmove, self.z, m))
        fb, tb = 1 << f, 1 << t
        cap_pt = 0
        cap_sq = t
        if m & FLAG_EP:
            cap_sq = t ^ 8
            cap_pt = PAWN
        else:
            cap_pt = sq[t] & 7
        new_pt = promo or pt
        rook_from = rook_to = -1
        # remove mover
        bb[us] ^= fb
        bb[pt + 1] ^= fb
        sq[f] = 0
        z ^= zp[us][pt][f]
        if cap_pt:
            cb = 1 << cap_sq
            bb[them] ^= cb
            bb[cap_pt + 1] ^= cb
            sq[cap_sq] = 0
            z ^= zp[them][cap_pt][cap_sq]
        if m & FLAG_CASTLE:
            if t > f:
                rook_from, rook_to = f + 3, f + 1
            else:
                rook_from, rook_to = f - 4, f - 1
            rb = (1 << rook_from) | (1 << rook_to)
            bb[us] ^= rb
            bb[ROOK + 1] ^= rb
            sq[rook_from] = 0
            sq[rook_to] = ROOK + 8 * us
            z ^= zp[us][ROOK][rook_from] ^ zp[us][ROOK][rook_to]
        bb[us] |= tb
        bb[new_pt + 1] |= tb
        sq[t] = new_pt + 8 * us
        z ^= zp[us][new_pt][t]
        c = castling_old & CASTLE_KEEP[f] & CASTLE_KEEP[t]
        if c != castling_old:
            z ^= zt.castle[castling_old] ^ zt.castle[c]
        self.castling = c
        if pt == PAWN and (t - f == 16 or f - t == 16):
            e = (f + t) >> 1
            self.ep = e
            z ^= zt.ep[e]
        else:
            self.ep = -1
        self.halfmove = 0 if (pt == PAWN or cap_pt) else self.halfmove + 1
        self.turn = them
        self.bb = bb
        self.sq = sq
        self.z = z
        self.keys.append(z ^ (zt.ep[self.ep] if self.ep >= 0 else 0))
        self.pargs.append((us, them, pt, new_pt, f, t, cap_pt, cap_sq, rook_from, rook_to, bb))
        self.accs.append(None)

    def child_key(self, m):
        """Zobrist key (self.z) of the position after move m, without
        making it (same arithmetic as make())."""
        f, t = m & 63, (m >> 6) & 63
        sq = self.sq
        us = self.turn
        pt = sq[f] & 7
        zt = self.zt
        zp = zt.piece
        z = self.z ^ zt.side
        if self.ep >= 0:
            z ^= zt.ep[self.ep]
        if m & FLAG_EP:
            z ^= zp[1 - us][PAWN][t ^ 8]
        else:
            c = sq[t] & 7
            if c:
                z ^= zp[1 - us][c][t]
        if m & FLAG_CASTLE:
            if t > f:
                z ^= zp[us][ROOK][f + 3] ^ zp[us][ROOK][f + 1]
            else:
                z ^= zp[us][ROOK][f - 4] ^ zp[us][ROOK][f - 1]
        z ^= zp[us][pt][f] ^ zp[us][(m >> 12) & 7 or pt][t]
        c0 = self.castling
        c = c0 & CASTLE_KEEP[f] & CASTLE_KEEP[t]
        if c != c0:
            z ^= zt.castle[c0] ^ zt.castle[c]
        if pt == PAWN and (t - f == 16 or f - t == 16):
            z ^= zt.ep[(f + t) >> 1]
        return z

    def make_null(self):
        zt = self.zt
        self.stack.append((self.bb, self.sq, self.castling, self.ep, self.halfmove, self.z, 0))
        self.pargs.append(None)
        self.accs.append(self.accs[-1])
        z = self.z ^ zt.side
        if self.ep >= 0:
            z ^= zt.ep[self.ep]
        self.ep = -1
        self.halfmove += 1
        self.turn = 1 - self.turn
        self.z = z
        self.keys.append(z)

    def unmake(self):
        (self.bb, self.sq, self.castling, self.ep, self.halfmove, self.z, _m) = self.stack.pop()
        self.turn = 1 - self.turn
        self.keys.pop()
        self.pargs.pop()
        self.accs.pop()

    def acc(self):
        """Current (white, black) accumulators, materialised lazily."""
        accs = self.accs
        a = accs[-1]
        if a is not None:
            return a
        k = len(accs) - 1
        while accs[k] is None:
            k -= 1
        apply = self.nn.apply
        pargs = self.pargs
        for i in range(k + 1, len(accs)):
            args = pargs[i - 1]
            accs[i] = accs[i - 1] if args is None else apply(accs[i - 1], args)
        return accs[-1]

    def last_move(self):
        """Last move made in the search (0 = null move or none)."""
        return self.stack[-1][6] if self.stack else 0

    def piece_bbs10(self):
        bb = self.bb
        ow, ob = bb[1], bb[0]
        return (bb[2] & ow, bb[3] & ow, bb[4] & ow, bb[5] & ow, bb[6] & ow,
                bb[2] & ob, bb[3] & ob, bb[4] & ob, bb[5] & ob, bb[6] & ob)

    def is_repeated(self, floor):
        """Same rule as the engine's python-chess version: the position
        occurred before (same side to move) within the reversible window,
        never looking past `floor` (the last null move)."""
        keys = self.keys
        top = len(keys) - 1
        n = self.halfmove if self.halfmove < top - floor else top - floor
        if n < 4:
            return False
        cur = keys[top]
        for i in range(top - 4, top - n - 1, -2):
            if keys[i] == cur:
                return True
        return False


SEE_VALUE = [0, 100, 320, 330, 500, 900, 20000]


# ==========================================================================
# NNUE state on top of the fast board
# ==========================================================================
class NNUE:
    """Accumulator updates and evaluation for Position (Stockfish 17.1 net).

    Accumulators are int16 arrays of ACC_W = 3072 + 16 columns (network
    features plus the psqt sums split into high/low bytes).  Arrays are never
    modified in place once stored (every update builds a new array), so they
    can be shared freely between the undo stack and the king-square refresh
    cache ("finny table": for each perspective and king square, the last
    accumulator computed with the king there plus the piece bitboards it was
    built from; a king move then only applies the difference)."""

    def __init__(self, model):
        self.model = model
        self.W = model.ft
        self.b = model.bias
        self.finny = [[None] * 64, [None] * 64]
        self.ecache = {}

    def feature_indices(self, bb, pov):
        k = (bb[7] & bb[pov]).bit_length() - 1
        o = KORIENT[pov][k]
        fb = FBASE[pov][k]
        out = []
        for color in (1, 0):
            occ = bb[color]
            base_c = fb[color]
            for pt in (1, 2, 3, 4, 5, 6):
                base = base_c[pt]
                x = bb[pt + 1] & occ
                while x:
                    low = x & -x
                    out.append(base + ((low.bit_length() - 1) ^ o))
                    x ^= low
        return out

    def refresh(self, pos, pov):
        return self.refresh_bb(pos.bb, pov)

    def refresh_bb(self, bb, pov):
        idx = self.feature_indices(bb, pov)
        return self.W[idx].sum(axis=0, dtype=np.int16) + self.b

    def _refresh_cached(self, bb, pov, ksq):
        ow, ob = bb[1], bb[0]
        pieces = (bb[2] & ow, bb[3] & ow, bb[4] & ow, bb[5] & ow, bb[6] & ow, bb[7] & ow,
                  bb[2] & ob, bb[3] & ob, bb[4] & ob, bb[5] & ob, bb[6] & ob, bb[7] & ob)
        entry = self.finny[pov][ksq]
        if entry is None:
            acc = self.refresh_bb(bb, pov)
            self.finny[pov][ksq] = (acc, pieces)
            return acc
        acc, old = entry
        if old == pieces:
            return acc
        o = KORIENT[pov][ksq]
        fb = FBASE[pov][ksq]
        add, rem = [], []
        for j in range(12):
            po, pn = old[j], pieces[j]
            if po != pn:
                base = fb[1 if j < 6 else 0][j % 6 + 1]
                x = pn & ~po
                while x:
                    low = x & -x
                    add.append(base + ((low.bit_length() - 1) ^ o))
                    x ^= low
                x = po & ~pn
                while x:
                    low = x & -x
                    rem.append(base + ((low.bit_length() - 1) ^ o))
                    x ^= low
        W = self.W
        if add:
            acc = acc + (W[add[0]] if len(add) == 1 else W[add].sum(axis=0, dtype=np.int16))
        if rem:
            acc = acc - (W[rem[0]] if len(rem) == 1 else W[rem].sum(axis=0, dtype=np.int16))
        self.finny[pov][ksq] = (acc, pieces)
        return acc

    def apply(self, prev, args):
        """Accumulators after a move, from the accumulators before it and the
        change record made by Position.make() (bb = board after the move)."""
        us, them, pt, new_pt, f, t, cap_pt, cap_sq, rook_from, rook_to, bb = args
        acc_w, acc_b = prev
        W = self.W
        kings = bb[7]
        if pt == 6:
            mine = self._refresh_cached(bb, us, t)
            other = acc_b if us else acc_w
            k = (kings & bb[them]).bit_length() - 1
            o = KORIENT[them][k]
            fb = FBASE[them][k]
            kb = fb[us][6]
            other = other - W[kb + (f ^ o)]
            other += W[kb + (t ^ o)]
            if cap_pt:
                other -= W[fb[them][cap_pt] + (t ^ o)]
            if rook_from >= 0:
                rb = fb[us][4]
                other -= W[rb + (rook_from ^ o)]
                other += W[rb + (rook_to ^ o)]
            return (mine, other) if us else (other, mine)
        k = (kings & bb[1]).bit_length() - 1
        o = KORIENT[1][k]
        fb = FBASE[1][k]
        w = acc_w - W[fb[us][pt] + (f ^ o)]
        w += W[fb[us][new_pt] + (t ^ o)]
        if cap_pt:
            w -= W[fb[them][cap_pt] + (cap_sq ^ o)]
        k = (kings & bb[0]).bit_length() - 1
        o = KORIENT[0][k]
        fb = FBASE[0][k]
        b = acc_b - W[fb[us][pt] + (f ^ o)]
        b += W[fb[us][new_pt] + (t ^ o)]
        if cap_pt:
            b -= W[fb[them][cap_pt] + (cap_sq ^ o)]
        return w, b

    def prefetch(self, pos, moves):
        """Evaluate the children of pos after `moves` (non-king moves) in one
        batch: accumulators by a single row gather, network pass batched,
        scores stored in the eval cache.  Returns {move: (white, black)
        accumulators} for the move loop.  Same numbers as one-by-one."""
        bb = pos.bb
        kings = bb[7]
        us = pos.turn
        them = 1 - us
        if not (bb[0] & ~kings and bb[1] & ~kings):
            return None                         # bare king: mop-up territory
        them_last = _popcount(bb[them] & ~kings) == 1
        sq = pos.sq
        ecache = self.ecache
        kw = (kings & bb[1]).bit_length() - 1
        kb = (kings & bb[0]).bit_length() - 1
        ow, fw = KORIENT[1][kw], FBASE[1][kw]
        ob, fb = KORIENT[0][kb], FBASE[0][kb]
        fwu, fwt, fbu, fbt = fw[us], fw[them], fb[us], fb[them]
        rows, cap_rows, cap_j, todo = [], [], [], []
        for m in moves:
            f = m & 63
            pt = sq[f] & 7
            if pt == KING:
                continue
            t = (m >> 6) & 63
            if m & FLAG_EP:
                cap_sq, cap_pt = t ^ 8, PAWN
            else:
                cap_sq, cap_pt = t, sq[t] & 7
            if cap_pt and them_last:
                continue                         # child may be a mop-up position
            z = pos.child_key(m)
            if z in ecache:
                continue
            new_pt = (m >> 12) & 7 or pt
            rows.append(((fwu[pt] + (f ^ ow), fwu[new_pt] + (t ^ ow)),
                         (fbu[pt] + (f ^ ob), fbu[new_pt] + (t ^ ob))))
            if cap_pt:
                cap_j.append(len(todo))
                cap_rows.append((fwt[cap_pt] + (cap_sq ^ ow), fbt[cap_pt] + (cap_sq ^ ob)))
            todo.append((m, z, pt, new_pt, cap_pt))
        K = len(todo)
        if K < 2 or K > BATCH_MAX:
            return None
        W = self.W
        w, b = pos.acc()
        G = W[rows]                              # (K, 2, 2, ACC_W)
        C = np.stack((w, b)) - G[:, :, 0]
        C += G[:, :, 1]
        if cap_j:
            C[cap_j] -= W[cap_rows]
        # children: side to move is `them`
        if them:
            S, N = C[:, 0], C[:, 1]
        else:
            S, N = C[:, 1], C[:, 0]
        n_all = _popcount(bb[0] | bb[1])
        cnt = [0, _popcount(bb[2]), _popcount(bb[3]), _popcount(bb[4]),
               _popcount(bb[5]), _popcount(bb[6])]
        buckets = {}
        for j, (m, z, pt, new_pt, cap_pt) in enumerate(todo):
            buckets.setdefault((n_all - (1 if cap_pt else 0) - 1) >> 2, []).append(j)
        prop = [0] * K
        model = self.model
        for bucket, js in buckets.items():
            if len(js) == K:
                r = model.propagate_batch(S, N, bucket)
            else:
                r = model.propagate_batch(S[js], N[js], bucket)
            for j, v in zip(js, r):
                prop[j] = v
        D = (S[:, PSQT_HI:] - N[:, PSQT_HI:]).tolist()
        if len(ecache) + K >= EVAL_CACHE_MAX:
            ecache.clear()
        out = {}
        for j, (m, z, pt, new_pt, cap_pt) in enumerate(todo):
            bucket = (n_all - (1 if cap_pt else 0) - 1) >> 2
            d = D[j]
            ps = d[bucket] * 256 + d[8 + bucket]
            psqt = ps // 32 if ps >= 0 else -((-ps) // 32)
            pr = prop[j]
            positional = pr // 16 if pr >= 0 else -((-pr) // 16)
            x = 125 * psqt + 131 * positional
            v = x // 128 if x >= 0 else -((-x) // 128)
            c = psqt - positional
            x = v * (c if c >= 0 else -c)
            v -= x // 18000 if x >= 0 else -((-x) // 18000)
            p_, n_, b_, r_, q_ = cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]
            if cap_pt == PAWN:
                p_ -= 1
            elif cap_pt == KNIGHT:
                n_ -= 1
            elif cap_pt == BISHOP:
                b_ -= 1
            elif cap_pt == ROOK:
                r_ -= 1
            elif cap_pt == QUEEN:
                q_ -= 1
            if new_pt != pt:                     # promotion
                p_ -= 1
                if new_pt == QUEEN:
                    q_ += 1
                elif new_pt == ROOK:
                    r_ += 1
                elif new_pt == BISHOP:
                    b_ += 1
                else:
                    n_ += 1
            x = v * (77777 + 535 * p_ + 781 * n_ + 825 * b_ + 1276 * r_ + 2538 * q_)
            v = x // 77777 if x >= 0 else -((-x) // 77777)
            mm = p_ + 3 * (n_ + b_) + 5 * r_ + 9 * q_
            x = 100 * v / TO_CP_A[mm if mm < 79 else 78]
            cp = int(x + 0.5) if x >= 0 else -int(0.5 - x)
            if not -EVAL_LIMIT < cp < EVAL_LIMIT:
                cp = EVAL_LIMIT - 1 if cp > 0 else 1 - EVAL_LIMIT
            ecache[z] = (cp, True)
            out[m] = (C[j, 0], C[j, 1])
        return out

    # -- evaluation ------------------------------------------------------------
    def evaluate_cached(self, pos):
        """Memoised on the zobrist key (iterative deepening revisits about a
        third of all evaluated positions).  The cache holds the undamped score
        (flagged negative-zero-free: (score, damp)) because the key does not
        include the 50-move counter."""
        e = self.ecache.get(pos.z)
        if e is None:
            if len(self.ecache) >= EVAL_CACHE_MAX:
                self.ecache.clear()
            bb = pos.bb
            k = bb[7]
            if bb[1] & ~k and bb[0] & ~k:          # nobody has a bare king
                e = self.ecache[pos.z] = (self.nnue(pos), True)
            else:
                e = self.ecache[pos.z] = self._static(pos)
        v, damp = e
        if damp and pos.halfmove:
            v -= (v * pos.halfmove) // 212 if v >= 0 else -((-v * pos.halfmove) // 212)
        return v

    def evaluate(self, pos):
        v, damp = self._static(pos)
        if damp and pos.halfmove:
            return damp_50(v, pos.halfmove)
        return v

    def _static(self, pos):
        m = mopup(pos)
        if m is not None:
            return m, False
        return self.nnue(pos), True

    def raw(self, pos):
        """(psqt, positional) exactly as Stockfish's Network::evaluate()."""
        w, b = pos.acc()
        s, n = (w, b) if pos.turn else (b, w)
        bb = pos.bb
        bucket = (_popcount(bb[0] | bb[1]) - 1) >> 2
        hi, lo = PSQT_HI + bucket, PSQT_LO + bucket
        ps = (int(s[hi]) - int(n[hi])) * 256 + int(s[lo]) - int(n[lo])
        psqt = _tdiv(_tdiv(ps, 2), 16)
        positional = _tdiv(self.model.propagate(s, n, bucket), 16)
        return psqt, positional

    def nnue(self, pos):
        """Stockfish 17.1 Eval::evaluate() (big net, no optimism, 50-move
        damping applied separately) in centipawns via SF's to_cp.  The
        network pass is inlined (same arithmetic as raw() + propagate())."""
        a = pos.accs[-1]
        if a is None:
            a = pos.acc()
        bb = pos.bb
        if pos.turn:
            s, n = a
        else:
            n, s = a
        bucket = (_popcount(bb[0] | bb[1]) - 1) >> 2
        hi = PSQT_HI + bucket
        ps = (int(s[hi]) - int(n[hi])) * 256 + int(s[hi + 8]) - int(n[hi + 8])
        M = self.model
        t, x, xs = M._t, M._x, M._xs
        clip, mul = _uclip, np.multiply
        clip(s[:L1], 0, 127, out=t)
        mul(t[:L1H], t[L1H:], out=x[:L1H])
        clip(n[:L1], 0, 127, out=t)
        mul(t[:L1H], t[L1H:], out=x[L1H:L1])
        np.right_shift(xs, 7, out=xs)
        o = np.dot(M.w0[bucket], x).tolist()            # fc_0 / 64 (exact)
        h = M._hl
        for i in range(15):
            v = o[i]
            q = (v * v) // 128.0                          # fc0^2 >> 19
            h[i] = q if q < 127.0 else 127.0
            q = v // 1.0                                  # fc0 >> 6
            h[15 + i] = 0.0 if q < 0.0 else (q if q < 127.0 else 127.0)
        hn = M._hn
        hn[:30] = h[:30]
        o1 = np.dot(M.w1[bucket], hn)
        np.floor(o1, out=o1)
        clip(o1, 0.0, 127.0, out=o1)
        fwd = int(o[15] * 64) * 9600
        pr = (int(np.dot(M.w2[bucket], o1)) + M.b2[bucket]
              + (fwd // 8128 if fwd >= 0 else -((-fwd) // 8128)))
        psqt = ps // 32 if ps >= 0 else -((-ps) // 32)          # (ps / 2) / 16
        positional = pr // 16 if pr >= 0 else -((-pr) // 16)
        x = 125 * psqt + 131 * positional
        v = x // 128 if x >= 0 else -((-x) // 128)
        c = psqt - positional
        x = v * (c if c >= 0 else -c)
        v -= x // 18000 if x >= 0 else -((-x) // 18000)
        pawns = _popcount(bb[2])
        knights, bishops = _popcount(bb[3]), _popcount(bb[4])
        rooks, queens = _popcount(bb[5]), _popcount(bb[6])
        x = v * (77777 + 535 * pawns + 781 * knights + 825 * bishops
                 + 1276 * rooks + 2538 * queens)
        v = x // 77777 if x >= 0 else -((-x) // 77777)
        m = pawns + 3 * (knights + bishops) + 5 * rooks + 9 * queens
        x = 100 * v / TO_CP_A[m if m < 79 else 78]   # std::round: half away from zero
        cp = int(x + 0.5) if x >= 0 else -int(0.5 - x)
        return cp if -EVAL_LIMIT < cp < EVAL_LIMIT else (EVAL_LIMIT - 1 if cp > 0 else 1 - EVAL_LIMIT)


EVAL_LIMIT = 20000
TO_CP_A = []
for _m in range(79):
    _mm = (17 if _m < 17 else 78 if _m > 78 else _m) / 58.0
    TO_CP_A.append(((-13.50030198 * _mm + 40.92780883) * _mm - 36.82753545) * _mm + 386.83004070)



def damp_50(v, halfmove_clock):
    """Stockfish 17: v -= v * rule50 / 212 (shuffling scores drift to 0)."""
    return v - _tdiv(v * halfmove_clock, 212)


# --------------------------------------------------------------------------
# Mop-up evaluation for bare-king endgames.
# The NNUE has no idea how to mate a lone king (KQK scores the same with the
# defending king in the centre or in a corner), which is why Stockfish 12
# switched to its classical eval (KXK) for such positions.  When one side has
# only its king and the other has mating material, score the position as a
# known win plus a gradient toward mate: defending king to the edge (or the
# bishop's corner for KBNK) and attacking king close.
# --------------------------------------------------------------------------
MOPUP_BASE = 2000
MOPUP_VALUE = [0, 100, 320, 330, 500, 900, 0]
CENTRE_DIST = [max(3 - (s & 7), (s & 7) - 4) + max(3 - (s >> 3), (s >> 3) - 4) for s in range(64)]
BB_LIGHT, BB_DARK = chess.BB_LIGHT_SQUARES, chess.BB_DARK_SQUARES


def _dist(a, b):
    return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3)))


def _popcount(x):
    return bin(x).count('1')


if hasattr(int, 'bit_count'):          # Python >= 3.10
    _popcount = int.bit_count


def _bare_king(pos):
    bb = pos.bb
    kings = bb[7]
    return bb[1] == kings & bb[1] or bb[0] == kings & bb[0]


def mopup(pos):
    """Side-to-move score if one side has a bare king and the other can force
    mate with pieces, else None."""
    bb = pos.bb
    kings = bb[7]
    ow, ob = bb[1], bb[0]
    if ow == kings & ow:
        weak = 1
    elif ob == kings & ob:
        weak = 0
    else:
        return None
    strong = 1 - weak
    so = bb[strong]
    q, r = bb[6] & so, bb[5] & so
    b, n, p = bb[4] & so, bb[3] & so, bb[2] & so
    bn_mate = False
    if not (q or r):
        two_bishops = bool(b & BB_LIGHT) and bool(b & BB_DARK)
        bn_mate = bool(b) and bool(n) and not two_bishops
        if not (two_bishops or bn_mate):
            return None          # pawns only / KNNK / same-coloured bishops: leave to NNUE
    wk = (kings & bb[weak]).bit_length() - 1
    sk = (kings & so).bit_length() - 1
    score = MOPUP_BASE
    for pt, x in ((1, p), (2, n), (3, b), (4, r), (5, q)):
        score += MOPUP_VALUE[pt] * _popcount(x)
    score += 40 * CENTRE_DIST[wk] + 20 * (7 - _dist(wk, sk))
    if bn_mate and not (q or r) and _popcount(b) == 1 and _popcount(n) == 1:
        # KBNK: mate only happens in a corner of the bishop's colour.
        corners = (0, 63) if b & BB_DARK else (7, 56)
        score += 60 * (7 - min(_dist(wk, c) for c in corners))
    return score if pos.turn == strong else -score


def _first(t):
    return t[0]


class Timeout(Exception):
    pass


def tt_store_score(score, ply):
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def tt_load_score(score, ply):
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


EXACT, LOWER, UPPER = 0, 1, 2
MOVE_ID = 0x7FFF


class _ZT:
    """Zobrist tables in the layout fastboard expects."""
    piece = [[ZP[0][pt] for pt in range(7)], [ZP[1][pt] for pt in range(7)]]
    side = Z_SIDE
    ep = Z_EP
    castle = [z_castle(sum(1 << CASTLE_RIGHT_SQ[bit] for bit in (1, 2, 4, 8) if c & bit))
              for c in range(16)]


# ==========================================================================
# Search
# ==========================================================================
class Searcher:
    def __init__(self, model):
        self.nn = NNUE(model)
        self.tt = {}
        self.tt_old = {}
        self.root_len = 0
        self.root_move = None
        self.null_floor = 0
        self.null_cutoffs = 0
        self.rfp_cutoffs = 0
        self.lmp_prunes = self.futility_prunes = 0
        self.pos = None
        self.clear()

    tb = None             # chess.syzygy.Tablebase or None
    tb_max = 0

    def set_tablebase(self, path):
        if self.tb is not None:
            self.tb.close()
        self.tb, self.tb_max = None, 0
        if not path or not os.path.isdir(path):
            return
        import chess.syzygy
        names = [f[:-5] for f in os.listdir(path) if f.endswith('.rtbw')]
        if not names:
            return
        self.tb = chess.syzygy.open_tablebase(path)
        self.tb_max = max(len(n.replace('v', '')) for n in names)

    def tb_wdl(self, pos):
        """WDL for the side to move (2 win, 1 cursed win, 0, -1, -2) or None."""
        bb = pos.bb
        b = chess.Board(None)
        b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK] = bb[1], bb[0]
        b.occupied = bb[0] | bb[1]
        b.pawns, b.knights, b.bishops = bb[2], bb[3], bb[4]
        b.rooks, b.queens, b.kings = bb[5], bb[6], bb[7]
        b.turn = bool(pos.turn)
        b.castling_rights = 0
        b.ep_square = pos.ep if pos.ep >= 0 else None
        try:
            return self.tb.probe_wdl(b)
        except (KeyError, ValueError, IndexError, OSError):
            return None

    def tb_root(self, board):
        """Tablebase move for a won or lost root position, else None."""
        if self.tb is None or board.castling_rights or \
                chess.popcount(board.occupied) > self.tb_max:
            return None
        try:
            wdl = self.tb.probe_wdl(board)
            if wdl == 0:
                return None           # drawn: let the search look for chances
            best = None
            for m in board.legal_moves:
                board.push(m)
                try:
                    w = -self.tb.probe_wdl(board)
                    z = -self.tb.probe_dtz(board)
                except KeyError:
                    board.pop()
                    return None
                mate = board.is_checkmate()
                zeroing = board.halfmove_clock == 0
                board.pop()
                # Plies until the next zeroing move (capture / pawn move):
                # a zeroing move itself counts as 1.  Wins: mate now, else the
                # fewest; losses: the most.
                dist = 1 if zeroing else abs(z) + 1
                if w > 0:
                    key = (w, 1 if mate else 0, -dist)
                elif w < 0:
                    key = (w, 0, dist)
                else:
                    key = (w, 0, 0)
                if best is None or key > best[0]:
                    best = (key, m)
            if best is None:
                return None
            sc = TB_WIN - 1 if wdl > 0 else (-(TB_WIN - 1) if wdl < 0 else 0)
            return best[1], sc
        except (KeyError, ValueError, IndexError, OSError):
            return None

    def clear(self):
        self.tt.clear()
        self.tt_old.clear()
        self.history = [0] * (2 * 64 * 64)
        self.killers = [[0, 0] for _ in range(MAX_PLY + 8)]
        self.static_stack = [None] * (MAX_PLY + 8)
        self.countermove = [0] * (2 * 64 * 64)

    def _check_time(self):
        if time.perf_counter() > self.deadline:
            raise Timeout

    # -- move ordering keys (identical to the python-chess version) ------------
    @staticmethod
    def _capture_key(pos):
        sq = pos.sq
        ep = pos.ep

        def key(m):
            to = (m >> 6) & 63
            v = sq[to] & 7
            promo = (m >> 12) & 7
            if not v:          # en passant (pawn onto the ep square) or quiet promotion
                v = PAWN if to == ep and not promo else 0
            return VICTIM[v] * 10 - ATTACKER[sq[m & 63] & 7] + (8000 if promo == QUEEN else 0)
        return key

    @staticmethod
    def _evasion_key(pos):
        cap = Searcher._capture_key(pos)
        theirs = pos.bb[1 - pos.turn]

        def key(m):
            capture = theirs >> ((m >> 6) & 63) & 1 or m & FLAG_EP
            return 100000 + cap(m) if capture or (m >> 12) & 7 else 0
        return key

    # -- quiescence -----------------------------------------------------------
    def qsearch(self, pos, alpha, beta, ply, in_check=None, first_ply=False):
        self.nodes += 1
        if not self.nodes & 255:
            self._check_time()
        nn = self.nn

        if in_check is None:
            in_check = pos.is_check()
        if in_check:
            # In check there is no stand-pat: every evasion must be searched,
            # otherwise checks beyond the horizon are invisible.
            moves = pos.legal_moves()
            if not moves:
                return -MATE_SCORE + ply
            if ply >= MAX_PLY:
                return nn.evaluate(pos)
            moves.sort(key=self._evasion_key(pos), reverse=True)
            best = -INF
            for move in moves:
                pos.make(move)
                score = -self.qsearch(pos, -beta, -alpha, ply + 1)
                pos.unmake()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if score >= beta:
                            break
            return best

        bb = pos.bb
        if not (bb[2] | bb[5] | bb[6]) and pos.is_insufficient():
            return 0
        stand = nn.evaluate_cached(pos)
        if stand >= beta or ply >= MAX_PLY:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        # Pseudo-legal captures; legality is only checked for moves that
        # survive delta/SEE pruning (most never get that far).
        moves = pos.gen_captures()
        us = pos.turn
        pawns7 = bb[2] & bb[us] & (BB_RANK_7 if us else BB_RANK_2)
        if pawns7:
            # Quiet queen promotions (under-promotions are not worth it here).
            moves += [m for m in pos.gen_pseudo(pawns7, BB_BACKRANKS & ~(bb[0] | bb[1]))
                      if (m >> 12) & 7 == QUEEN]
        if not moves:
            if first_ply and QS_CHECKS:
                return self._qsearch_checks(pos, alpha, beta, ply, best)
            return best
        if len(moves) > 1:
            moves.sort(key=self._capture_key(pos), reverse=True)
        sq = pos.sq
        king = blockers = None
        for move in moves:
            promo = (move >> 12) & 7
            if promo:
                if promo != QUEEN:
                    continue
            else:
                victim = sq[(move >> 6) & 63] & 7 or PAWN     # empty target = en passant
                # Delta pruning: even winning this piece for free can't reach alpha.
                if stand + SEE_VALUE[victim] + DELTA_MARGIN <= alpha:
                    continue
                # SEE pruning: only possible loss when the attacker is worth
                # more than the victim, so only pay for SEE then.
                if SEE_VALUE[sq[move & 63] & 7] > SEE_VALUE[victim] and pos.see(move) < 0:
                    continue
            if king is None:
                king = (bb[7] & bb[us]).bit_length() - 1
                blockers = pos.slider_blockers(king)
            if not pos.is_safe(king, blockers, move):
                continue
            pos.make(move)
            score = -self.qsearch(pos, -beta, -alpha, ply + 1)
            pos.unmake()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        return best
        if first_ply and QS_CHECKS:
            return self._qsearch_checks(pos, alpha, beta, ply, best)
        return best

    def _qsearch_checks(self, pos, alpha, beta, ply, best):
        """Quiet moves that give a direct check (first quiescence ply only).
        Check squares come from attack tables around the enemy king, so no
        per-move gives_check() call is needed; SEE-losing checks are skipped."""
        bb = pos.bb
        us = pos.turn
        occ = bb[0] | bb[1]
        empty = ~occ & BB_ALL
        ksq = (bb[7] & bb[1 - us]).bit_length() - 1
        mine = bb[us]
        dg = DIAG_ATT[ksq][DIAG_MASK[ksq] & occ]
        orthg = FILE_ATT[ksq][FILE_MASK[ksq] & occ] | RANK_ATT[ksq][RANK_MASK[ksq] & occ]
        targets = (
            (bb[3] & mine, KNIGHT_ATT[ksq]),
            (bb[4] & mine, dg),
            (bb[5] & mine, orthg),
            (bb[6] & mine, dg | orthg),
            (bb[2] & mine, PAWN_ATT[1 - us][ksq]),
        )
        king = (bb[7] & mine).bit_length() - 1
        blockers = None
        for from_bb, to_bb in targets:
            to_bb &= empty
            if not from_bb or not to_bb:
                continue
            for move in pos.gen_pseudo(from_bb, to_bb):
                if (move >> 12) & 7:
                    continue
                if pos.see(move) < 0:
                    continue
                if blockers is None:
                    blockers = pos.slider_blockers(king)
                if not pos.is_safe(king, blockers, move):
                    continue
                pos.make(move)
                score = -self.qsearch(pos, -beta, -alpha, ply + 1)
                pos.unmake()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if score >= beta:
                            return best
        return best

    # -- main search --------------------------------------------------------
    def search(self, pos, depth, alpha, beta, ply, allow_null=True, in_check=None):
        nn = self.nn
        if ply > 0:
            if pos.halfmove >= 100:
                if not (pos.is_check() and not pos.legal_moves()):
                    return 0
            bb = pos.bb
            if (not (bb[2] | bb[5] | bb[6]) and pos.is_insufficient()) \
                    or pos.is_repeated(self.null_floor):
                # Any repetition (also of a position from the game before
                # the search) is scored as a draw: a deliberately strict rule
                # that keeps the winning side making progress.
                return 0
            if self.tb is not None and pos.halfmove == 0 and not pos.castling \
                    and _popcount(bb[0] | bb[1]) <= self.tb_max:
                w = self.tb_wdl(pos)
                if w is not None:
                    return (TB_WIN - ply) if w == 2 else (-(TB_WIN - ply) if w == -2 else 0)
            # Mate-distance pruning.
            if alpha < ply - MATE_SCORE:
                alpha = ply - MATE_SCORE
            if beta > MATE_SCORE - ply - 1:
                beta = MATE_SCORE - ply - 1
            if alpha >= beta:
                return alpha

        if in_check is None:
            in_check = pos.is_check()
        if in_check:
            depth += 1                       # check extension
        if depth <= 0 or ply >= MAX_PLY:
            return self.qsearch(pos, alpha, beta, ply, in_check, True)

        self.nodes += 1
        if not self.nodes & 255:
            self._check_time()

        key = pos.z
        entry = self.tt.get(key)
        if entry is None:
            entry = self.tt_old.get(key)
        tt_move = 0
        if entry is not None:
            e_depth, e_flag, e_score, tt_move = entry
            if ply > 0 and e_depth >= depth:
                s = tt_load_score(e_score, ply)
                if e_flag == EXACT or (e_flag == LOWER and s >= beta) or \
                        (e_flag == UPPER and s <= alpha):
                    return s

        if IIR and not tt_move and depth >= IIR_MIN_DEPTH and ply > 0:
            # Internal iterative reduction: no TT move means ordering is poor
            # here, so search one ply shallower.
            depth -= 1
        if ply == 0 and self.root_move:
            # Always search the previous iteration's best move first.
            tt_move = self.root_move
        pv_node = beta - alpha > 1
        static_eval = None
        improving = False
        if not in_check and ply > 0:
            static_eval = nn.evaluate_cached(pos)
            if IMPROVING:
                prev = self.static_stack[ply - 2] if ply >= 2 else None
                improving = prev is None or static_eval > prev
        self.static_stack[ply] = static_eval
        if static_eval is not None and not pv_node:
            # Reverse futility pruning (smaller margin when improving).
            margin = RFP_MARGIN * (depth - improving)
            if (depth <= RFP_MAX_DEPTH and abs(beta) < MATE_BOUND
                    and static_eval - margin >= beta):
                self.rfp_cutoffs += 1
                return static_eval - margin

        bb = pos.bb
        us = pos.turn
        # Null-move pruning (not in check, not twice in a row, not near mate
        # bounds, not without non-pawn material: zugzwang risk).
        if (allow_null and not in_check and ply > 0 and depth >= NMP_MIN_DEPTH
                and static_eval is not None and not pv_node
                and beta < MATE_BOUND
                and bb[us] & ~(bb[2] | bb[7])
                and static_eval >= beta):
            r = NMP_BASE_R + depth // NMP_DEPTH_DIV
            saved_floor = self.null_floor
            pos.make_null()
            self.null_floor = len(pos.keys) - 1
            try:
                score = -self.search(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            finally:
                self.null_floor = saved_floor
            pos.unmake()
            if score >= beta:
                self.null_cutoffs += 1
                return beta if score >= MATE_BOUND else score

        if in_check:
            moves = pos.legal_moves()
            if not moves:
                return -MATE_SCORE + ply
            king = blockers = None
        else:
            # Pseudo-legal moves; legality is checked lazily right before a
            # move is searched, so moves after a beta cutoff are never checked.
            moves = None
            king = (bb[7] & bb[us]).bit_length() - 1
            blockers = pos.slider_blockers(king)
        killers = self.killers[ply]
        k0 = killers[0]
        k0_id = k0 & MOVE_ID if k0 else -1
        hist = self.history
        side = 4096 if us else 0
        prev = pos.last_move()
        cm_idx = side + (prev & 63) * 64 + ((prev >> 6) & 63) if prev else -1

        # The TT move is always ordered first; when not in check it is tried
        # before generating the other moves at all.
        first = 0
        if tt_move and not in_check and pos.is_pseudo_legal(tt_move) \
                and pos.is_safe(king, blockers, tt_move):
            first = tt_move

        orig_alpha = alpha
        best, best_move = -INF, 0
        quiets_tried = []
        lmr_row = LMR_TABLE[min(depth, 63)]
        # Shallow-depth pruning of quiet moves (non-PV, not in check);
        # not in bare-king endings: mating there needs precise quiet moves.
        prune_ok = (static_eval is not None and not pv_node
                    and depth <= max(LMP_MAX_DEPTH, FUTILITY_MAX_DEPTH)
                    and not _bare_king(pos))
        lmp_limit = LMP_BASE + depth * depth if depth <= LMP_MAX_DEPTH else 10 ** 9
        futile = (prune_ok and depth <= FUTILITY_MAX_DEPTH
                  and static_eval + FUTILITY_BASE + FUTILITY_PER_PLY * depth <= alpha)
        them_occ_bb = bb[1 - us]
        pawns_bb = bb[2]
        ep_sq = pos.ep
        is_safe = pos.is_safe
        considered = 0
        i = -1
        unchecked_first = bool(first)
        pref = None
        want_pref = BATCH_EVAL and depth == 1 and not in_check
        # Move list: the TT move first; the rest is generated and scored only
        # when needed (after the TT move failed to cut off), appended to the
        # same list.  A plain loop instead of a per-node generator/closure.
        seq = [(10_000_000, first)] if first else []
        generated = False
        idx = 0
        while True:
            if idx == len(seq):
                if generated:
                    break
                generated = True
                if moves is None:
                    moves = pos.gen_pseudo()
                sq = pos.sq
                them_occ = bb[1 - us]
                ep = pos.ep
                pawns = bb[2]
                tt_id = tt_move & MOVE_ID if tt_move else -1
                k1 = killers[1]
                k1_id = k1 & MOVE_ID if k1 else -1
                cm = self.countermove[cm_idx] if cm_idx >= 0 else 0
                cm_id = cm & MOVE_ID if cm else -1
                # Squares attacked by enemy pawns, and our pieces standing on them.
                ep_bb = pawns & them_occ
                if us:
                    pawn_att = ((ep_bb >> 7) & ~BB_FILE_A) | ((ep_bb >> 9) & ~BB_FILE_H)
                else:
                    pawn_att = (((ep_bb << 7) & ~BB_FILE_H) | ((ep_bb << 9) & ~BB_FILE_A)) & BB_ALL
                our_pieces = bb[us] & ~pawns & ~bb[7]
                threatened = our_pieces & pawn_att
                see = pos.see
                scored = []
                append = scored.append
                for m in moves:
                    mid = m & MOVE_ID
                    f = m & 63
                    t = (m >> 6) & 63
                    if mid == tt_id:
                        if first:
                            continue                      # already searched
                        s = 10_000_000
                    elif them_occ >> t & 1:
                        v, a = sq[t] & 7, sq[f] & 7
                        if SEE_VALUE[a] > SEE_VALUE[v] and not (m >> 12) & 7 and see(m) < 0:
                            # Losing capture: after killers/countermove, before quiets.
                            s = BAD_CAPTURE_SCORE + VICTIM[v] // 10
                        else:
                            s = 5_000_000 + VICTIM[v] * 10 - ATTACKER[a]
                    elif t == ep and pawns >> f & 1:
                        s = 5_000_000 + VICTIM[PAWN] * 10 - ATTACKER[PAWN]
                    elif (m >> 12) & 7 == QUEEN:
                        s = 5_000_000 + 8000
                    elif mid == k0_id:
                        s = 4_000_002
                    elif mid == k1_id:
                        s = 4_000_001
                    elif mid == cm_id:
                        s = COUNTERMOVE_SCORE
                    else:
                        s = hist[side + f * 64 + t]
                        if our_pieces >> f & 1:
                            if pawn_att >> t & 1:
                                s -= PAWN_ATTACKED_MALUS
                            elif threatened >> f & 1:
                                s += ESCAPE_BONUS
                        if s > QUIET_CAP:
                            s = QUIET_CAP
                    append((s, m))
                scored.sort(key=_first, reverse=True)
                seq.extend(scored)
                if idx == len(seq):
                    break
            s, move = seq[idx]
            idx += 1
            if unchecked_first:
                unchecked_first = False           # TT move: legality already checked
            elif not in_check and not is_safe(king, blockers, move):
                continue
            considered += 1
            mt = (move >> 6) & 63
            is_quiet = not (them_occ_bb >> mt & 1) and not (move >> 12) & 7 and \
                not (mt == ep_sq and pawns_bb >> (move & 63) & 1)
            if (prune_ok and is_quiet and s < 4_000_000 and best > -MATE_BOUND):
                # Late-move pruning (checking moves are never pruned).
                if considered > lmp_limit and not pos.gives_check(move):
                    self.lmp_prunes += 1
                    continue
                # Futility pruning: a quiet move can't lift us to alpha.
                if futile and not pos.gives_check(move):
                    self.futility_prunes += 1
                    continue
            i += 1
            if want_pref and i == 1:
                # The first move did not cut off: evaluate the moves that
                # will be searched here (legal, not futility/history-pruned)
                # in one batch before going on.
                want_pref = False
                cand = [move]
                for s2, m2 in seq[idx:]:
                    if not is_safe(king, blockers, m2):
                        continue
                    t2 = (m2 >> 6) & 63
                    if (prune_ok and s2 < 4_000_000 and not (them_occ_bb >> t2 & 1)
                            and not (m2 >> 12) & 7
                            and not (t2 == ep_sq and pawns_bb >> (m2 & 63) & 1)
                            and futile):
                        continue                 # will be pruned (unless it checks)
                    cand.append(m2)
                pref = nn.prefetch(pos, cand)
            pos.make(move)
            if pref is not None:
                a = pref.get(move)
                if a is not None:
                    pos.accs[-1] = a
            gives_check = pos.is_check()
            if i == 0:
                score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, True, gives_check)
            else:
                r = 0
                if (depth >= LMR_MIN_DEPTH and i >= LMR_MIN_MOVE and is_quiet and s < 4_000_000
                        and not in_check and not gives_check):
                    r = lmr_row[min(i, 63)]
                    if pv_node:
                        r -= 1
                    if IMPROVING_LMR and not improving:
                        r += 1
                    if s > LMR_GOOD_HISTORY:
                        r -= 1
                    r = max(0, min(r, depth - 2))
                score = -self.search(pos, depth - 1 - r, -alpha - 1, -alpha, ply + 1, True, gives_check)
                if score > alpha and r > 0:
                    score = -self.search(pos, depth - 1, -alpha - 1, -alpha, ply + 1, True, gives_check)
                if alpha < score < beta:          # only possible in PV nodes
                    score = -self.search(pos, depth - 1, -beta, -alpha, ply + 1, True, gives_check)
            pos.unmake()
            if score > best:
                best, best_move = score, move
                if ply == 0:
                    self.root_best = (move, score)
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        if is_quiet:               # quiet move caused a cutoff (incl. TT move)
                            if not k0 or (move & MOVE_ID) != k0_id:
                                killers[1] = k0
                                killers[0] = move
                            if cm_idx >= 0:
                                self.countermove[cm_idx] = move
                            bonus = HIST_BONUS[min(depth, 63)]
                            j = side + (move & 63) * 64 + mt
                            hist[j] += bonus - hist[j] * bonus // HIST_LIMIT
                            for q in quiets_tried:
                                hist[q] -= bonus + hist[q] * bonus // HIST_LIMIT
                        break
            if is_quiet:
                quiets_tried.append(side + (move & 63) * 64 + mt)

        if i < 0:
            return 0                         # no legal move and not in check: stalemate

        flag = LOWER if best >= beta else (EXACT if best > orig_alpha else UPPER)
        if len(self.tt) >= TT_MAX_ENTRIES // 2:
            # Two generations instead of wiping everything at once.
            self.tt_old = self.tt
            self.tt = {}
        self.tt[key] = (depth, flag, tt_store_score(best, ply), best_move)
        return best

    def pv(self, pos, first, max_len=12):
        line = [first]
        pos.make(first)
        made = 1
        seen = {pos.z}
        while len(line) < max_len:
            e = self.tt.get(pos.z) or self.tt_old.get(pos.z)
            if not e or not e[3] or e[3] not in pos.legal_moves():
                break
            pos.make(e[3])
            made += 1
            line.append(e[3])
            if pos.z in seen:
                break
            seen.add(pos.z)
        for _ in range(made):
            pos.unmake()
        return line

    def think(self, board, soft_limit, hard_limit, max_depth=64, info=None, dynamic=False):
        """Iterative deepening on a python-chess board (converted to the fast
        board once).  Returns (best_move as chess.Move, score, depth)."""
        start = time.perf_counter()
        self.deadline = start + hard_limit
        self.nodes = 0
        self.null_cutoffs = 0
        self.rfp_cutoffs = 0
        self.lmp_prunes = self.futility_prunes = 0
        self.null_floor = 0
        pos = self.pos = Position(board, self.nn, _ZT)
        self.killers = [[0, 0] for _ in range(MAX_PLY + 8)]
        root_stack = len(pos.stack)
        self.root_len = len(pos.keys) - 1
        self.root_move = 0
        legal = pos.legal_moves()
        if not legal:
            return None, 0, 0
        tbm = self.tb_root(board) if self.tb is not None else None
        if tbm is not None:
            if info:
                info('info depth 1 score cp %d nodes 0 tbhits 1 pv %s' % (tbm[1], tbm[0].uci()))
            return tbm[0], tbm[1], 1
        best_move, best_score, done = legal[0], 0, 0
        stable = 0                 # iterations in a row with the same best move
        for depth in range(1, max_depth + 1):
            self.root_best = None
            try:
                if ASPIRATION and depth >= ASP_MIN_DEPTH and abs(best_score) < MATE_BOUND:
                    delta = ASP_WINDOW
                    a, b = best_score - delta, best_score + delta
                    while True:
                        score = self.search(pos, depth, a, b, 0)
                        if score <= a:
                            a = max(-INF, a - delta)
                        elif score >= b:
                            b = min(INF, b + delta)
                            if self.root_best is not None:
                                self.root_move = self.root_best[0]
                        else:
                            break
                        delta *= 2
                        if delta > 1000:
                            a, b = -INF, INF
                else:
                    score = self.search(pos, depth, -INF, INF, 0)
            except Timeout:
                while len(pos.stack) > root_stack:
                    pos.unmake()
                # A move that beat the first (previous best) move during the
                # unfinished iteration was searched fully: safe to use it.
                if self.root_best is not None and depth > 1:
                    best_move, best_score = self.root_best
                break
            prev_move, prev_score = best_move, best_score
            best_move, best_score, done = self.root_best[0], score, depth
            stable = stable + 1 if (best_move & MOVE_ID) == (prev_move & MOVE_ID) else 0
            self.root_move = best_move
            elapsed = time.perf_counter() - start
            if info:
                pv = ' '.join(move_uci(m) for m in self.pv(pos, best_move))
                if abs(score) > MATE_BOUND:
                    mate_in = (MATE_SCORE - abs(score) + 1) // 2
                    sc = 'mate %d' % (mate_in if score > 0 else -mate_in)
                else:
                    sc = 'cp %d' % score
                info('info depth %d score %s nodes %d nps %d time %d pv %s'
                     % (depth, sc, self.nodes, self.nodes / max(elapsed, 1e-6),
                        elapsed * 1000, pv))
            if abs(score) > MATE_BOUND or len(legal) == 1:
                break
            limit = soft_limit
            if dynamic and depth >= TM_MIN_DEPTH:
                # Stable best move -> stop earlier; best move just changed or
                # score dropping -> keep thinking (hard limit still applies).
                limit *= TM_STABILITY[min(stable, len(TM_STABILITY) - 1)]
                drop = prev_score - score
                if drop > TM_DROP_MIN and abs(prev_score) < MATE_BOUND:
                    limit *= 1.0 + min(drop / TM_DROP_FULL, 1.0) * TM_DROP_EXTRA
            if elapsed > limit:
                break
        return move_to_chess(best_move), best_score, done


# --------------------------------------------------------------------------
# UCI
# --------------------------------------------------------------------------
def parse_go(line):
    parts = line.split()
    opts = {}
    for i, token in enumerate(parts):
        if token in ('wtime', 'btime', 'winc', 'binc', 'movestogo', 'movetime', 'depth') \
                and i + 1 < len(parts):
            try:
                opts[token] = int(parts[i + 1])
            except ValueError:
                pass
        elif token == 'infinite':
            opts['infinite'] = True
    return opts


def time_budget(opts, turn):
    """Return (soft, hard, max_depth) in seconds."""
    max_depth = opts.get('depth', 64)
    if 'movetime' in opts:
        t = max(0.01, opts['movetime'] / 1000.0 - MOVE_OVERHEAD)
        return t, t, max_depth
    if opts.get('infinite') or ('depth' in opts and 'wtime' not in opts and 'btime' not in opts):
        return 1e9, 1e9, max_depth
    remain = opts.get('wtime' if turn else 'btime', 30000) / 1000.0
    inc = opts.get('winc' if turn else 'binc', 0) / 1000.0
    mtg = opts.get('movestogo')
    if mtg:
        target = remain / max(mtg, 1) + inc * 0.8
    else:
        target = remain * 0.05 + inc * 0.8
    # Keep MOVE_OVERHEAD in reserve for process/pipe latency and the final
    # partial node batch between clock checks.
    hard = max(0.02, min(target * TM_HARD, remain * 0.25, remain - MOVE_OVERHEAD))
    soft = max(0.01, min(target * 0.6, hard))
    return soft, hard, max_depth


def parse_position(line):
    parts = line.split()
    moves = []
    if 'moves' in parts:
        mi = parts.index('moves')
        moves = parts[mi + 1:]
        parts = parts[:mi]
    if len(parts) > 1 and parts[1] == 'fen':
        board = chess.Board(' '.join(parts[2:]))
    else:
        board = chess.Board()
    for uci in moves:
        try:
            board.push_uci(uci)
        except ValueError:
            print('info string illegal move %s ignored' % uci, flush=True)
            break
    return board


def main():
    gc.disable()
    board = chess.Board()
    searcher = Searcher(SF17Network())
    searcher.set_tablebase(os.path.join(ROOT, 'syzygy'))
    out = lambda s: print(s, flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == 'uci':
            out('id name PyStockfish-SF17NNUE')
            out('id author Tester')
            out('option name SyzygyPath type string default <empty>')
            out('uciok')
            continue
        if line == 'isready':
            out('readyok')
            continue
        if line.startswith('setoption'):
            low = line.lower()
            if 'name syzygypath' in low and ' value ' in line:
                searcher.set_tablebase(line.split(' value ', 1)[1].strip())
        elif line == 'ucinewgame':
            board = chess.Board()
            searcher.clear()
        elif line.startswith('position'):
            board = parse_position(line)
        elif line.startswith('go'):
            opts = parse_go(line)
            soft, hard, max_depth = time_budget(opts, board.turn)
            dynamic = ('wtime' in opts or 'btime' in opts) and 'movetime' not in opts
            move, _, _ = searcher.think(board, soft, hard, max_depth, info=out, dynamic=dynamic)
            out('bestmove %s' % (move.uci() if move else '0000'))
            # Collect garbage now, on the opponent's clock, instead of letting
            # a full collection (tens of ms with a big TT) land mid-search.
            gc.collect()
        elif line == 'quit':
            break


if __name__ == '__main__':
    if sys.argv[1:] == ['--setup']:
        # One-time setup: download the network if needed, then convert it.
        if not os.path.exists(os.path.join(ROOT, NET_FILE)) and \
                not os.path.exists(os.path.join(ROOT, NPY_FT)):
            download_network(ROOT)
        print('Converting the network (about 10 s)...', flush=True)
        SF17Network()
        print('NNUE ready: %s, %s' % (NPY_FT, NPY_REST))
    else:
        main()

# CHANGES (vs. original)
# Speed (identical results): move-based accumulator updates instead of two
#   piece_map() diffs per move; weights loaded into RAM instead of memmap;
#   dense layers as exact float32 dot products with folded /64 and bias;
#   one legal-move generation per node instead of up to four (is_checkmate,
#   is_stalemate, legal_moves ...); repetition scan limited to the reversible
#   window; no redundant is_checkmate() inside evaluate().
# Eval bug: the net was always fed [white, black] and the result negated for
#   black.  The network expects [side to move, other] and already returns a
#   side-to-move score; with black to move the old code evaluated the position
#   as if white were to move.  Score conversion now truncates toward zero.
# Search: qsearch searches all evasions when in check (no stand-pat in check);
#   stand-pat before move generation; mate scores by ply from root (were based
#   on remaining depth, inconsistent between search and qsearch); 2-fold
#   repetition = draw in search; transposition table; TT/killer/history move
#   ordering; quiet queen promotions ordered with captures; check extension;
#   mate-distance pruning; partial-iteration best move kept on timeout;
#   principal variation search (null-window scouts, re-search on fail
#   high); late-move reductions for quiet moves (log*log table, less in PV
#   nodes and for high-history moves); history malus;
#   reverse futility pruning (depth <= 4, 120 cp/ply, non-PV);
#   quiescence delta pruning and SEE pruning of losing captures;
#   late-move pruning (depth <= 3, 3 + depth^2 moves, never checks) and futility pruning
#   (depth <= 2, margin 100 + 100*depth) of quiet moves in non-PV nodes;
#   null-move pruning (non-PV nodes only) (R = 2 + depth/4, not in check, not twice in a row,
#   only with non-pawn material, static eval >= beta);
#   depth cap raised from 7 to 64; clock checked every 256 nodes (was
#   1024: up to ~70 ms deadline overshoot) with a 30 ms move-overhead
#   reserve; garbage collection only after bestmove is sent; soft time limit so a new iteration is not
#   started when it cannot finish.
# Speed pass 2: Zobrist keys updated incrementally in push() (also fixes TT
#   collisions: hash() of the python-chess key tuple aliased pieces on
#   f8/g8/h8 with a1/b1/c1 because int hashes are taken mod 2**61-1);
#   king moves rebuild only the mover's half, from a per-king-square cache
#   ("finny table"); pseudo-legal move generation with lazy legality checks;
#   TT move searched before generating the other moves; eval cache keyed by
#   Zobrist; ordering compares squares instead of Move.__eq__; cheaper
#   insufficient-material and is_check handling.
# Refinements: quiet checks at the first quiescence ply; aspiration windows;
#   "improving" flag (RFP margin, LMR); internal iterative reduction.
# Audit fixes: two-generation TT
#   instead of wiping the table when full; previous best root move always
#   searched first; quiet TT-move cutoffs now update killers/history/
#   countermove (were excluded by the ordering-score test).
# Move ordering: SEE splits captures into winning (first) and losing (after
#   killers and countermove, never reduced or pruned); history with gravity;
#   countermove; moves out of / into enemy pawn attacks.
# Endgame conversion: Stockfish-12-style 50-move damping of the eval
#   (v * (100 - halfmove_clock) / 100) so progress beats shuffling; mop-up
#   evaluation when one side has a bare king (known-win base + drive the
#   king to the edge / bishop's corner + bring the attacking king close).
# UCI: go movetime / depth / infinite / movestogo supported; info lines with
#   depth, score, nodes, nps, pv; position parsing no longer depends on FEN
#   having exactly six fields.
