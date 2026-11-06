#!/usr/bin/env python3
"""
Lichess bot for the pure-Python Stockfish-17.1 NNUE engine.

Adjustments:
- Supports up to MAX_CONCURRENT games (default: one per CPU core).
- Accepts any rated standard clock game (bullet included).
- Each game runs in its own engine process so clocks cannot mix.

Requirements:
  pip install berserk python-chess numpy

Setup:
  1. Run once: python3 engine.py --setup   (downloads + converts the network)
  2. (optional) Syzygy tablebases: put the files in a folder "syzygy" next to engine.py
  3. Put your Lichess bot token in token.txt
  4. Run this script
"""
import datetime
import os
import subprocess
import threading
import time
import berserk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("token.txt", "r") as f:
    TOKEN = f.read().strip()

ENGINE_PATH = "python3"          # or "python"
# Absolute path, so the bot works no matter which folder you start it from.
ENGINE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine.py")

# Network lag reserve: subtracted from our clock before it is sent to the
# engine.  The engine itself only keeps 30 ms in reserve (tuned for local
# matches); Lichess round trips add more.
LAG_RESERVE_MS = 250
# Decline anything faster than this initial time (seconds); 15 s UltraBullet
# is not playable for a pure-Python engine over the network.
MIN_INITIAL_SECONDS = 60

# Engine is single-threaded. With 4 CPU cores you can run up to 4
# concurrent games. Note: when multiple engines search at the same time
# they share the cores, so each move takes longer wall-clock time.
MAX_CONCURRENT = max(1, os.cpu_count() or 1)   # the engine is single-threaded: one game per core

session = berserk.TokenSession(TOKEN)
client = berserk.Client(session=session)

MY_ID = client.account.get()["id"]
print(f"Logged in as {MY_ID}")

# Active games tracking
active_games = set()
active_lock = threading.Lock()


def to_ms(v):
    """berserk delivers clock values as int ms (gameFull), timedelta
    (gameState, current berserk) or datetime (older berserk)."""
    if isinstance(v, datetime.timedelta):
        return int(v.total_seconds() * 1000)
    if isinstance(v, datetime.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=datetime.timezone.utc)
        return int(v.timestamp() * 1000)
    return int(v or 0)


def start_engine():
    """Start a fresh engine process and do the UCI handshake."""
    proc = subprocess.Popen(
        [ENGINE_PATH, ENGINE_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Engine closed stdout during handshake")
        if "uciok" in line:
            break

    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Engine closed stdout during isready")
        if "readyok" in line:
            break

    return proc


def get_best_move(proc, moves, wtime, btime, winc, binc, game_id="", is_white=True):
    """Send position + go to the engine and return the bestmove UCI string."""
    wtime, btime, winc, binc = to_ms(wtime), to_ms(btime), to_ms(winc), to_ms(binc)
    if is_white:
        wtime = max(50, wtime - LAG_RESERVE_MS)
    else:
        btime = max(50, btime - LAG_RESERVE_MS)
    # Debug: show exactly what times are being sent
    print(f"[{game_id}] DEBUG go wtime={wtime}ms ({wtime/1000:.1f}s) "
          f"btime={btime}ms ({btime/1000:.1f}s) "
          f"winc={winc}ms binc={binc}ms  moves={len(moves)}")

    if moves:
        pos_cmd = "position startpos moves " + " ".join(moves) + "\n"
    else:
        pos_cmd = "position startpos\n"
    proc.stdin.write(pos_cmd)
    proc.stdin.flush()

    go_cmd = f"go wtime {wtime} btime {btime} winc {winc} binc {binc}\n"
    proc.stdin.write(go_cmd)
    proc.stdin.flush()

    start = time.perf_counter()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Engine closed stdout while waiting for bestmove")
        line = line.strip()
        if line.startswith("bestmove"):
            elapsed = time.perf_counter() - start
            parts = line.split()
            move = parts[1] if len(parts) > 1 else "0000"
            print(f"[{game_id}] DEBUG engine returned {move} in {elapsed:.2f}s")
            return move


def play_game(game_id):
    print(f"Starting game {game_id}")
    engine = None
    try:
        engine = start_engine()
        is_white = None
        last_moves = None

        for event in client.bots.stream_game_state(game_id):
            etype = event.get("type")

            if etype == "gameFull":
                state = event.get("state", {})
                white_id = event.get("white", {}).get("id")
                black_id = event.get("black", {}).get("id")
                if white_id == MY_ID:
                    is_white = True
                elif black_id == MY_ID:
                    is_white = False
                else:
                    print(f"Warning: could not determine color for game {game_id}")
                    is_white = None

                moves_str = state.get("moves", "")
                wtime = state.get("wtime", 0)
                btime = state.get("btime", 0)
                winc = state.get("winc", 0)
                binc = state.get("binc", 0)
                status = state.get("status", "started")

            elif etype == "gameState":
                moves_str = event.get("moves", "")
                wtime = event.get("wtime", 0)
                btime = event.get("btime", 0)
                winc = event.get("winc", 0)
                binc = event.get("binc", 0)
                status = event.get("status", "started")

            else:
                continue

            if status not in ("started", "created"):
                print(f"Game {game_id} finished with status: {status}")
                break

            moves = moves_str.split() if moves_str else []

            if moves == last_moves:
                continue
            last_moves = moves

            if is_white is None:
                continue
            our_turn = (len(moves) % 2 == 0) if is_white else (len(moves) % 2 == 1)
            if not our_turn:
                continue

            try:
                print(f"[{game_id}] Our turn (white={is_white}, moves={len(moves)})")
                move = get_best_move(engine, moves, wtime, btime, winc, binc, game_id, is_white)
                if move and move != "0000":
                    client.bots.make_move(game_id, move)
                    print(f"[{game_id}] Played {move}")
                else:
                    print(f"[{game_id}] Engine returned no move")
            except Exception as e:
                print(f"[{game_id}] Engine error: {e}")
                break

    except Exception as e:
        print(f"Game {game_id} error: {e}")
    finally:
        if engine is not None:
            try:
                engine.stdin.write("quit\n")
                engine.stdin.flush()
            except Exception:
                pass
            try:
                engine.terminate()
                engine.wait(timeout=2)
            except Exception:
                engine.kill()
        with active_lock:
            active_games.discard(game_id)
        print(f"Finished game {game_id} (active: {len(active_games)})")


def should_accept(challenge):
    """Return (accept: bool, reason: str)."""
    if challenge.get("variant", {}).get("key") != "standard":
        return False, "variant"
    if not challenge.get("rated"):
        return False, "casual"

    tc = challenge.get("timeControl") or {}
    if tc.get("type") != "clock":
        return False, "timeControl"

    show = tc.get("show") or "?"
    if (tc.get("limit") or 0) < MIN_INITIAL_SECONDS:
        return False, "tooFast"

    with active_lock:
        if len(active_games) >= MAX_CONCURRENT:
            return False, "later"

    return True, show


def main():
    print(f"Bot online (max {MAX_CONCURRENT} concurrent game(s)), waiting for challenges...")
    for event in client.bots.stream_incoming_events():
        etype = event.get("type")

        if etype == "challenge":
            challenge = event["challenge"]
            ok, reason = should_accept(challenge)
            cid = challenge["id"]
            if ok:
                try:
                    client.bots.accept_challenge(cid)
                    print(f"Accepted {reason} challenge {cid}")
                except Exception as e:
                    print(f"Accept failed for {cid}: {e}")
            else:
                try:
                    # Map our reason to a Lichess decline reason when possible
                    decline_reason = "generic"
                    if reason.startswith("tooFast"):
                        decline_reason = "tooFast"
                    elif reason == "timeControl":
                        decline_reason = "timeControl"
                    elif reason == "later":
                        decline_reason = "later"
                    elif reason == "variant":
                        decline_reason = "variant"
                    elif reason == "casual":
                        decline_reason = "casual"
                    client.bots.decline_challenge(cid, reason=decline_reason)
                    print(f"Declined challenge {cid}: {reason}")
                except Exception as e:
                    print(f"Decline failed for {cid}: {e}")

        elif etype == "gameStart":
            game_id = event["game"]["id"]
            with active_lock:
                if len(active_games) >= MAX_CONCURRENT:
                    print(f"Ignoring gameStart {game_id} (already at max concurrent)")
                    continue
                active_games.add(game_id)
            t = threading.Thread(target=play_game, args=(game_id,), daemon=True)
            t.start()

        elif etype == "gameFinish":
            game = event.get("game", {})
            print(f"Game finished: {game.get('id')} status={game.get('status', {}).get('name')}")


if __name__ == "__main__":
    main()
