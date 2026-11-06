#!/usr/bin/env python3
"""
Fischer-clock UCI Chess Engine Match Tester
Pairs two UCI engines under Fischer time control (base + increment).
Tracks clocks accurately, validates moves with python-chess, detects
checkmate/stalemate/50-move/threefold, records PGN-like games to .txt,
and estimates Elo difference with approximate uncertainty.
Uses a filtered set of roughly equal openings (longer standard lines only).
Works with regular Python (no PyPy). Suitable for Replit.

Games are played in pairs: same opening, colours swapped.  With --sprt the
match stops as soon as a sequential probability ratio test (pentanomial,
Elo bounds --elo0/--elo1) decides whether ENGINE2 is stronger than ENGINE1.

Crash-safe: every game is appended to --output as soon as it finishes.  If
the run dies, start it again with the same flags and it continues with the
games that are still missing.  Raising --games extends a finished match.
"""
import subprocess
import time
import argparse
import math
from datetime import datetime
import random
import os
import csv
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import chess
import chess.pgn

# Global filled from TSV / PGN files (or fallback)
OPENINGS = []

DEFAULT_IO_DELAY = 0.015

# Keywords that indicate unbalanced, trap, joke, or extreme lines
BAD_KEYWORDS = (
    "fool", "mate", "trap", "blitz", "sacrifice",
    "immortal", "evergreen", "scholar", "fried liver",
    "legal", "leopold", "greco", "boden", "lasker", "siberian",
    "halloween", "danish", "king's gambit", "evans",
    "budapest", "albin", "staunton", "from",
)

def parse_pgn_to_sans(pgn_str):
    """Turn a PGN move string into a clean list of SAN moves."""
    if not pgn_str:
        return []
    tokens = re.sub(r'\d+\.+\s*', ' ', pgn_str)
    tokens = re.sub(r'[!?+#]+', '', tokens)
    tokens = re.sub(r'\{.*?\}', '', tokens)          # remove comments
    tokens = re.sub(r'\(.*?\)', '', tokens)          # remove variations
    return [t for t in tokens.split() if t and not t[0].isdigit() and t not in ('1-0', '0-1', '1/2-1/2', '*')]

def is_roughly_equal(name, sans):
    """Reject short, trap, gambit, and irregular openings."""
    if len(sans) < 8:
        return False
    name_lower = name.lower()
    if any(kw in name_lower for kw in BAD_KEYWORDS):
        return False
    # Reject pure irregular / joke first moves that rarely lead to equal play
    first = sans[0].lower() if sans else ""
    if first in ('a3', 'a4', 'b3', 'b4', 'f3', 'g3', 'g4', 'h3', 'h4', 'na3', 'nh3'):
        # Allow a few known solid ones, but drop most
        if not any(x in name_lower for x in ('english', 'reti', 'catalan', 'king\'s indian attack')):
            return False
    return True

def load_openings_from_tsv(paths, min_moves=8):
    openings = []
    seen = set()
    for path in paths:
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    name = row.get('name', 'Unknown').strip()
                    pgn = row.get('pgn', '').strip()
                    if not pgn or not name:
                        continue
                    sans = parse_pgn_to_sans(pgn)
                    if len(sans) < min_moves or not is_roughly_equal(name, sans):
                        continue
                    key = (name, tuple(sans))
                    if key in seen:
                        continue
                    seen.add(key)
                    openings.append((name, sans))
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return openings

def load_openings_from_pgn(path, min_moves=8):
    """Optional: load from a PGN file of opening lines (one game per line)."""
    openings = []
    try:
        with open(path, encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                name = game.headers.get("Opening", game.headers.get("Event", "PGN Opening"))
                sans = []
                node = game
                while node.variations:
                    node = node.variation(0)
                    sans.append(node.san())
                if len(sans) >= min_moves and is_roughly_equal(name, sans):
                    openings.append((name, sans))
    except Exception as e:
        print(f"Warning: could not load PGN {path}: {e}")
    return openings

def get_openings(args):
    paths = []
    if args.openings:
        if os.path.isdir(args.openings):
            paths = sorted(glob.glob(os.path.join(args.openings, '*.tsv')))
            paths += sorted(glob.glob(os.path.join(args.openings, '*.pgn')))
        else:
            paths = [args.openings]
    else:
        paths = sorted(glob.glob('?.tsv'))

    openings = []
    for p in paths:
        if p.lower().endswith('.tsv'):
            openings.extend(load_openings_from_tsv([p], min_moves=args.min_moves))
        elif p.lower().endswith('.pgn'):
            openings.extend(load_openings_from_pgn(p, min_moves=args.min_moves))

    # Deduplicate
    seen = set()
    unique = []
    for name, sans in openings:
        key = (name, tuple(sans))
        if key not in seen:
            seen.add(key)
            unique.append((name, sans))
    openings = unique

    if openings:
        print(f"Loaded {len(openings)} filtered equal openings")
        return openings

    # Solid fallback set of classic roughly equal lines
    print("No suitable openings file found — using built-in equal set")
    return [
        ("Italian Game", ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5']),
        ("Italian: Giuoco Piano", ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'c3', 'Nf6']),
        ("Ruy Lopez", ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6']),
        ("Ruy Lopez: Morphy", ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6']),
        ("Ruy Lopez: Berlin", ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'Nf6']),
        ("Scotch Game", ['e4', 'e5', 'Nf3', 'Nc6', 'd4', 'exd4']),
        ("Four Knights", ['e4', 'e5', 'Nf3', 'Nc6', 'Nc3', 'Nf6']),
        ("Petroff Defense", ['e4', 'e5', 'Nf3', 'Nf6']),
        ("Sicilian: Najdorf", ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6', 'Nc3', 'a6']),
        ("Sicilian: Classical", ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6', 'Nc3', 'Nc6']),
        ("Sicilian: Scheveningen", ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6', 'Nc3', 'e6']),
        ("Sicilian: Taimanov", ['e4', 'c5', 'Nf3', 'e6', 'd4', 'cxd4', 'Nxd4', 'Nc6']),
        ("Caro-Kann: Classical", ['e4', 'c6', 'd4', 'd5', 'Nc3', 'dxe4', 'Nxe4', 'Bf5']),
        ("Caro-Kann: Advance", ['e4', 'c6', 'd4', 'd5', 'e5', 'Bf5']),
        ("French: Classical", ['e4', 'e6', 'd4', 'd5', 'Nc3', 'Nf6']),
        ("French: Tarrasch", ['e4', 'e6', 'd4', 'd5', 'Nd2', 'c5']),
        ("Queen's Gambit Declined", ['d4', 'd5', 'c4', 'e6']),
        ("QGD: Orthodox", ['d4', 'd5', 'c4', 'e6', 'Nc3', 'Nf6', 'Bg5', 'Be7']),
        ("Slav Defense", ['d4', 'd5', 'c4', 'c6']),
        ("Semi-Slav", ['d4', 'd5', 'c4', 'c6', 'Nf3', 'Nf6', 'Nc3', 'e6']),
        ("King's Indian", ['d4', 'Nf6', 'c4', 'g6', 'Nc3', 'Bg7']),
        ("Nimzo-Indian", ['d4', 'Nf6', 'c4', 'e6', 'Nc3', 'Bb4']),
        ("Queen's Indian", ['d4', 'Nf6', 'c4', 'e6', 'Nf3', 'b6']),
        ("Grünfeld Defense", ['d4', 'Nf6', 'c4', 'g6', 'Nc3', 'd5']),
        ("Catalan", ['d4', 'Nf6', 'c4', 'e6', 'g3']),
        ("English Opening", ['c4', 'e5']),
        ("English: Four Knights", ['c4', 'e5', 'Nc3', 'Nf6', 'Nf3', 'Nc6']),
        ("Reti Opening", ['Nf3', 'd5', 'c4']),
        ("London System", ['d4', 'd5', 'Bf4', 'Nf6', 'e3', 'e6']),
    ]

class UCIEngine:
    def __init__(self, cmd, name, io_delay=DEFAULT_IO_DELAY):
        self.cmd = cmd if isinstance(cmd, list) else cmd.split()
        self.name = name
        self.io_delay = io_delay
        self.proc = None
    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.wait_for("uciok")
        self.send("isready")
        self.wait_for("readyok")
    def send(self, msg):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(msg + "\n")
            self.proc.stdin.flush()
            time.sleep(self.io_delay)
    def read_line(self):
        line = self.proc.stdout.readline()
        return line.strip() if line else ""
    def wait_for(self, expected, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.read_line()
            if expected in line:
                return line
            if not line:
                break
        raise TimeoutError(f"{self.name}: no '{expected}'")
    def get_bestmove(self, go_cmd, max_wait):
        self.send(go_cmd)
        deadline = time.time() + max_wait
        while time.time() < deadline:
            line = self.read_line()
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None
    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=1.5)
        except Exception:
            if self.proc:
                self.proc.kill()

def _elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def match_stats(done, args):
    """Statistics from engine2's point of view (engine2 = the new version).

    Uses complete game pairs (pentanomial model) when available; that is
    both what SPRT needs and a much tighter error bar than treating games as
    independent, because each pair cancels the opening's bias."""
    g2 = {g: 1.0 - r["score"] for g, r in done.items()}       # engine2 points
    games = len(g2)
    wins = sum(1 for v in g2.values() if v == 1.0)
    losses = sum(1 for v in g2.values() if v == 0.0)
    draws = games - wins - losses
    pairs = [(g2[2 * p - 1] + g2[2 * p]) / 2.0
             for p in range(1, args.games // 2 + 1) if 2 * p - 1 in g2 and 2 * p in g2]
    penta = [0] * 5
    for x in pairs:
        penta[int(round(x * 4))] += 1
    st = {"games": games, "w": wins, "d": draws, "l": losses, "pairs": len(pairs), "penta": penta}
    xs = pairs if len(pairs) >= 2 else list(g2.values())
    n = len(xs)
    if n < 2:
        st.update(score=0.5, elo=0.0, lo=0.0, hi=0.0, llr=0.0)
        return st
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    st.update(score=sum(g2.values()) / games, elo=_elo(m),
              lo=_elo(m - 1.96 * se), hi=_elo(m + 1.96 * se))
    llr = 0.0
    if pairs and var > 0:
        s0 = 1.0 / (1.0 + 10 ** (-args.elo0 / 400.0))
        s1 = 1.0 / (1.0 + 10 ** (-args.elo1 / 400.0))
        mp = sum(pairs) / len(pairs)
        vp = sum((x - mp) ** 2 for x in pairs) / len(pairs)
        if vp > 0:
            llr = len(pairs) * (s1 - s0) * (2 * mp - s0 - s1) / (2 * vp)
    st["llr"] = llr
    return st


def sprt_bounds(args):
    return math.log(args.beta / (1 - args.alpha)), math.log((1 - args.beta) / args.alpha)


def sprt_status(st, args):
    lo, hi = sprt_bounds(args)
    if st["llr"] >= hi:
        verdict = "H1 accepted: engine2 is stronger"
    elif st["llr"] <= lo:
        verdict = "H0 accepted: engine2 is not stronger"
    else:
        verdict = "running"
    return (f"LLR {st['llr']:+.2f} ({lo:+.2f}, {hi:+.2f}) [{args.elo0:g}, {args.elo1:g}] {verdict}",
            verdict != "running")


def elo_estimate(score, n):
    if n == 0:
        return 0.0, 0.0
    p = score / n
    if p <= 0.0:
        return -9999.0, 0.0
    if p >= 1.0:
        return 9999.0, 0.0
    diff = -400.0 * math.log10((1.0 / p) - 1.0)
    se = 400.0 / math.sqrt(n)
    return diff, se

def play_one_game(g, args):
    eng1_white = (g % 2 == 1)
    white_cmd = args.engine1 if eng1_white else args.engine2
    black_cmd = args.engine2 if eng1_white else args.engine1
    white_name = args.name1 if eng1_white else args.name2
    black_name = args.name2 if eng1_white else args.name1
    white = UCIEngine(white_cmd, white_name, args.io_delay)
    black = UCIEngine(black_cmd, black_name, args.io_delay)
    try:
        white.start()
        black.start()
        white.send("ucinewgame")
        black.send("ucinewgame")
        white.send("isready")
        white.wait_for("readyok")
        black.send("isready")
        black.wait_for("readyok")
        board = chess.Board()
        wtime = float(args.base)
        btime = float(args.base)
        moves_uci = []
        result = None
        reason = "unknown"

        if not OPENINGS:
            opening_name, opening_sans = ("Starting Position", [])
        else:
            # Games 2p-1 and 2p form a pair: same opening, colours swapped
            # (game parity already alternates colours).  This cancels opening
            # bias and makes every pair a fair mini-match.  Seeded per pair,
            # so a resumed run picks the same opening as an uninterrupted one.
            pair = (g + 1) // 2
            if args.random_openings:
                opening_name, opening_sans = random.Random(f"{args.seed}:pair:{pair}").choice(OPENINGS)
            else:
                opening_name, opening_sans = OPENINGS[(pair - 1) % len(OPENINGS)]

        for san in opening_sans:
            try:
                move = board.parse_san(san)
                board.push(move)
                moves_uci.append(move.uci())
            except Exception:
                break

        if moves_uci:
            pos_cmd = "position startpos moves " + " ".join(moves_uci)
        else:
            pos_cmd = "position startpos"
        white.send(pos_cmd)
        black.send(pos_cmd)
        time.sleep(args.io_delay)
        if args.delay > 0:
            time.sleep(args.delay)

        while not board.is_game_over(claim_draw=args.claimable_draws):
            # By default, stop only when the current position has already
            # occurred three times. python-chess's can_claim_threefold_repetition()
            # also returns True when the player could claim by making a future
            # move, which is legal but is not an already-occurring repetition.
            if args.claimable_draws:
                threefold = board.can_claim_threefold_repetition()
                fifty_moves = board.can_claim_fifty_moves()
            else:
                threefold = board.is_repetition(3)
                fifty_moves = board.halfmove_clock >= 100

            if threefold:
                result = "1/2-1/2"
                reason = ("threefold_repetition_claimable" if args.claimable_draws
                          else "threefold_repetition")
                break
            if fifty_moves:
                result = "1/2-1/2"
                reason = ("fifty_moves_claimable" if args.claimable_draws
                          else "fifty_moves")
                break
            white_to_move = board.turn == chess.WHITE
            current = white if white_to_move else black
            current_time = wtime if white_to_move else btime
            if current_time <= 0.01:
                result = "0-1" if white_to_move else "1-0"
                reason = "flag (time)"
                break
            if not moves_uci:
                pos_cmd = "position startpos"
            else:
                pos_cmd = "position startpos moves " + " ".join(moves_uci)
            current.send(pos_cmd)
            time.sleep(args.io_delay)
            go_cmd = (
                f"go wtime {int(wtime * 1000)} btime {int(btime * 1000)} "
                f"winc {int(args.inc * 1000)} binc {int(args.inc * 1000)}"
            )
            t0 = time.perf_counter()
            bm = current.get_bestmove(go_cmd, max_wait=current_time + 2.0)
            elapsed = time.perf_counter() - t0
            if bm is None or bm in ("0000", "(none)"):
                result = "0-1" if white_to_move else "1-0"
                reason = "no bestmove"
                break
            try:
                move = chess.Move.from_uci(bm)
                if move not in board.legal_moves:
                    result = "0-1" if white_to_move else "1-0"
                    reason = f"illegal move {bm}"
                    break
                board.push(move)
                moves_uci.append(bm)
            except Exception:
                result = "0-1" if white_to_move else "1-0"
                reason = f"bad move {bm}"
                break
            if white_to_move:
                wtime = max(0.0, wtime - elapsed + args.inc)
            else:
                btime = max(0.0, btime - elapsed + args.inc)

        if result is None:
            outcome = board.outcome(claim_draw=args.claimable_draws)
            if outcome is None:
                result = "1/2-1/2"
                reason = "unfinished"
            else:
                if outcome.winner is None:
                    result = "1/2-1/2"
                    reason = outcome.termination.name.lower()
                else:
                    result = "1-0" if outcome.winner == chess.WHITE else "0-1"
                    reason = outcome.termination.name.lower()

        if result == "1-0":
            sc = 1.0 if eng1_white else 0.0
        elif result == "0-1":
            sc = 0.0 if eng1_white else 1.0
        else:
            sc = 0.5

        game = chess.pgn.Game()
        game.headers["Event"] = "Fischer Match"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = result
        game.headers["Opening"] = opening_name
        node = game
        b2 = chess.Board()
        for u in moves_uci:
            m = chess.Move.from_uci(u)
            node = node.add_variation(m)
            b2.push(m)
        pgn_text = str(game) + f"\n{{ {reason} }}\n"
        info = (
            f"Game {g}: {white_name} (W) vs {black_name} (B)\n"
            f"  Opening: {opening_name}\n"
            f"  {result} ({reason})  moves={len(moves_uci)}  Wtime={wtime:.1f} Btime={btime:.1f}"
        )
        return {
            "game": g,
            "score": sc,
            "pgn": pgn_text,
            "info": info,
            "result": result,
            "reason": reason,
        }
    finally:
        white.quit()
        black.quit()


# ---------------------------------------------------------------------------
# Crash-safe results file
#
# The output file is an append-only log.  Every finished game is written as
#     ### GAME <n> score=<engine1 score> result=<result> reason=<reason>
#     <PGN>
#     ### END <n>
# and flushed + fsync'ed immediately.  A game only counts once its END line
# is on disk, so a crash in the middle of a write is harmless: on the next
# run the partial tail is cut off and that game is simply played again.
# ---------------------------------------------------------------------------
HEADER_PREFIX = "### MATCH "
GAME_RE = re.compile(r"^### GAME (\d+) score=([0-9.]+) result=(\S+) reason=(.*)$")
END_RE = re.compile(r"^### END (\d+)$")


def match_signature(args):
    """Settings that must not change between a run and its resumption."""
    return (f"engine1={args.engine1!r} engine2={args.engine2!r} base={args.base} "
            f"inc={args.inc} seed={args.seed} random_openings={args.random_openings} "
            f"openings={args.openings!r} min_moves={args.min_moves} "
            f"claimable_draws={args.claimable_draws} pairs=1")


def load_results(path):
    """Return (header_line, {game: record}, byte offset of the clean end).

    The clean end is just after the last complete '### END' line (or after
    the header line if no game is complete yet).
    """
    done = {}
    header = None
    clean_end = 0
    current = None
    offset = 0
    with open(path, "rb") as f:
        for raw in f:
            offset += len(raw)
            if not raw.endswith(b"\n"):
                break                       # torn last line from a crash
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if header is None:
                if line.startswith(HEADER_PREFIX):
                    header = line
                    clean_end = offset
                    continue
                return None, {}, 0          # not a resumable results file
            m = GAME_RE.match(line)
            if m:
                current = {"game": int(m.group(1)), "score": float(m.group(2)),
                           "result": m.group(3), "reason": m.group(4), "pgn": []}
                continue
            m = END_RE.match(line)
            if m and current is not None and int(m.group(1)) == current["game"]:
                current["pgn"] = "\n".join(current["pgn"])
                done[current["game"]] = current
                current = None
                clean_end = offset
                continue
            if current is not None:
                current["pgn"].append(line)
    return header, done, clean_end


def append_game(path, res):
    record = (f"### GAME {res['game']} score={res['score']} result={res['result']} "
              f"reason={res['reason']}\n{res['pgn'].rstrip()}\n### END {res['game']}\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(record)
        f.flush()
        os.fsync(f.fileno())


def prepare_output(args):
    """Create the results file, or validate and clean it for resuming.
    Returns the dict of already-finished games."""
    header = HEADER_PREFIX + match_signature(args)
    if args.restart and os.path.exists(args.output):
        os.replace(args.output, args.output + ".old")
        print(f"--restart: moved previous results to {args.output}.old")
    if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {}
    old_header, done, clean_end = load_results(args.output)
    if old_header is None:
        raise SystemExit(
            f"{args.output} exists but is not a resumable results file (probably "
            f"written by the old version of this script). Move it away or use --restart.")
    if old_header != header:
        raise SystemExit(
            f"{args.output} belongs to a match with different settings:\n"
            f"  file: {old_header[len(HEADER_PREFIX):]}\n"
            f"  now:  {header[len(HEADER_PREFIX):]}\n"
            f"Use the same flags to resume, a different --output, or --restart.")
    if clean_end < os.path.getsize(args.output):
        # Drop a half-written game and any old summary block after the last
        # complete game; the summary is re-appended at the end of this run.
        with open(args.output, "r+b") as f:
            f.truncate(clean_end)
            f.flush()
            os.fsync(f.fileno())
    return done


def main():
    global OPENINGS
    ap = argparse.ArgumentParser(description="Fischer clock UCI match tester")
    ap.add_argument("--engine1", required=True, help="Launch command for engine 1")
    ap.add_argument("--engine2", required=True, help="Launch command for engine 2")
    ap.add_argument("--games", type=int, default=4, help="Number of games")
    ap.add_argument("--base", type=float, default=30.0, help="Base time (seconds)")
    ap.add_argument("--inc", type=float, default=0.5, help="Increment (seconds)")
    ap.add_argument("--delay", type=float, default=0.0, help="Optional pre-move clock delay (s)")
    ap.add_argument("--io-delay", type=float, default=DEFAULT_IO_DELAY, help="IO streaming delay")
    ap.add_argument("--output", default="match_games.txt", help="Output .txt file")
    ap.add_argument("--name1", default="EngineA", help="Display name 1")
    ap.add_argument("--name2", default="EngineB", help="Display name 2")
    ap.add_argument("--random-openings", action="store_true", help="Pick openings randomly instead of cycling")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel games (0 = auto-detect CPU cores)")
    ap.add_argument("--openings", default="", help="Path to openings.tsv / .pgn or directory containing them")
    ap.add_argument("--min-moves", type=int, default=8, help="Minimum moves required for an opening")
    ap.add_argument("--seed", type=int, default=1, help="Seed for --random-openings (game g always gets the same opening)")
    ap.add_argument("--sprt", action="store_true",
                    help="Sequential test: stop as soon as it is statistically clear whether engine2 "
                         "(the new version) is stronger. --games becomes the maximum.")
    ap.add_argument("--elo0", type=float, default=0.0, help="SPRT H0: engine2 is no better than this (Elo)")
    ap.add_argument("--elo1", type=float, default=10.0, help="SPRT H1: engine2 is at least this much better (Elo)")
    ap.add_argument("--alpha", type=float, default=0.05, help="SPRT false-positive rate")
    ap.add_argument("--beta", type=float, default=0.05, help="SPRT false-negative rate")
    ap.add_argument("--restart", action="store_true", help="Ignore existing --output (moved to <output>.old) and start over")
    ap.add_argument(
        "--claimable-draws",
        action="store_true",
        help="Use FIDE claimable-draw behavior; default requires the current position to already repeat 3 times",
    )
    args = ap.parse_args()

    OPENINGS = get_openings(args)
    done = prepare_output(args)
    todo = [g for g in range(1, args.games + 1) if g not in done]

    cores = os.cpu_count() or 1
    jobs = args.jobs if args.jobs > 0 else cores
    jobs = max(1, min(jobs, len(todo) or 1))
    print(f"Match: {args.name1} vs {args.name2}")
    print(f"{args.games} games | {args.base}s + {args.inc}s Fischer | delay={args.delay}s | io={args.io_delay}s")
    print(f"Parallel games: {jobs} (CPU cores detected: {cores})")
    print(f"Openings available: {len(OPENINGS)}")
    if done:
        s = sum(r["score"] for r in done.values())
        print(f"Resuming {args.output}: {len(done)} game(s) already finished "
              f"({args.name1} {s:.1f}/{len(done)}), {len(todo)} to play")
    print(f"Saving each game to {args.output} as it finishes\n")

    decided = False
    if args.sprt and done:
        line, decided = sprt_status(match_stats(done, args), args)
        print("SPRT so far: " + line)
        if decided:
            print("SPRT already decided; not playing more games (use a new --output to test again).\n")
    if todo and not decided:
        executor = ThreadPoolExecutor(max_workers=jobs)
        queue = list(todo)
        running = {}
        try:
            while queue or running:
                while queue and len(running) < jobs and not decided:
                    g = queue.pop(0)
                    running[executor.submit(play_one_game, g, args)] = g
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    g = running.pop(future)
                    try:
                        res = future.result()
                    except Exception as e:
                        # Not recorded, so it will be replayed on the next run.
                        print(f"Game {g} failed: {e}")
                        continue
                    append_game(args.output, res)   # on disk before we move on
                    done[g] = res
                    st = match_stats(done, args)
                    s = sum(r["score"] for r in done.values())
                    print(res["info"])
                    progress = (f"  [{len(done)}/{args.games} done, {args.name1} {s:.1f}/{len(done)} | "
                                f"{args.name2} {st['elo']:+.0f} Elo ({st['lo']:+.0f} .. {st['hi']:+.0f})")
                    if args.sprt:
                        line, now = sprt_status(st, args)
                        progress += " | " + line
                        if now and not decided:
                            decided = True
                            print("  SPRT decided: finishing the games already running, starting no new ones.")
                    print(progress + "]")
                    print()
        finally:
            executor.shutdown(wait=True)

    results = [done[g] for g in sorted(done)]
    scores = [r["score"] for r in results]
    total = sum(scores)
    n = len(scores)
    wr = total / n if n else 0
    ediff, eerr = elo_estimate(total, n)
    st = match_stats(done, args)
    missing = args.games - n
    summary = (
        f"=== SUMMARY ===\n"
        f"{args.name1} vs {args.name2}\n"
        f"Games played: {n}" + (f"  ({missing} not played - rerun to play them)" if missing and not (args.sprt and sprt_status(st, args)[1]) else "") + "\n"
        f"Score ({args.name1}): {total:.1f}/{n}  ({wr*100:.1f}%)\n"
        f"Est. Elo difference ({args.name1} stronger by): {ediff:+.0f} ± {eerr:.0f}  (old crude formula)\n"
        f"{args.name2} vs {args.name1}: {st['elo']:+.1f} Elo, 95% CI {st['lo']:+.1f} .. {st['hi']:+.1f}"
        f"  (W {st['w']} D {st['d']} L {st['l']} for {args.name2}; {st['pairs']} complete pairs,"
        f" pentanomial {st['penta']})\n"
        + (f"SPRT: {sprt_status(st, args)[0]}  alpha={args.alpha} beta={args.beta}\n" if args.sprt else "") +
        f"Time control: {args.base} + {args.inc} (Fischer)\n"
        f"IO delay used: {args.io_delay}s\n"
        f"Openings: {len(OPENINGS)} filtered equal lines (min_moves={args.min_moves}"
        f"{' / random seed ' + str(args.seed) if args.random_openings else ' / cycled'})\n"
        f"Parallel jobs used: {jobs}\n"
        f"Draw mode: {'claimable' if args.claimable_draws else 'current-position repetition'}\n"
    )
    print("\n" + summary)
    # The summary goes after the games; a later resume strips it and appends
    # a fresh one, so the file always ends with an up-to-date summary.
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(summary)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved full games to {args.output}")


if __name__ == "__main__":
    main()
