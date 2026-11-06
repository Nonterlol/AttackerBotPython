# Pure-Python NNUE Chess Engine

A UCI chess engine written entirely in **Python + NumPy**, evaluating positions with
**Stockfish 17.1's neural network** (`nn-1c0000000000.nnue`).

- No C/C++ extensions, no PyTorch/TensorFlow, no handwritten evaluation: every score comes from
  the network. Its output matches Stockfish's own `Network::evaluate()` **bit for bit**
  (verified against a compiled Stockfish 17.1 on 1,100+ positions).
- Single-threaded search; runs on plain CPython.
- Works in any UCI chess GUI, and comes with a Lichess bot script and benchmarking tools.

## Quick start

Requires **Python 3.9+**.

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt
python3 engine.py --setup
```

`--setup` runs once. It downloads the network (75 MB, from Stockfish's official server with a
GitHub mirror as fallback), checks its SHA-256 fingerprint, and converts it into
`sf17_ft.npy` / `sf17_layers.npz` (about 10 seconds). Those files are too large for GitHub, so
they are generated locally and ignored by git.

Check that it works:

```bash
printf 'uci\nisready\nposition startpos\ngo movetime 2000\nquit\n' | python3 engine.py
```

You should see `info depth ...` lines followed by a `bestmove`.

**Manual download (no internet on that machine):** download
`https://tests.stockfishchess.org/api/nn/nn-1c0000000000.nnue`, put it next to `engine.py`, and
run `python3 engine.py --setup`; it will convert the file you supplied instead of downloading.

## Using it in a chess GUI

Add a UCI engine whose command is:

```
python3 /full/path/to/engine.py
```

- **Windows:** use `python` (or `py`) instead of `python3`. If your GUI only accepts an `.exe`
  or `.bat`, create `engine.bat` containing `@python "%~dp0engine.py"` and add that file.
- Works with Cute Chess, Arena, En Croissant, BanksiaGUI, and similar.

### UCI options

| Option | Default | Meaning |
|---|---|---|
| `SyzygyPath` | *(empty)* | Folder with Syzygy tablebase files (`.rtbw`/`.rtbz`). |

If a folder named `syzygy` exists next to `engine.py`, it is used automatically. Tablebases are
optional; with them the engine plays covered endgames perfectly (fastest win, longest defence)
and scores captures into covered endgames exactly during the search. 3-4-5 piece tables (~1 GB)
are available from https://tablebase.lichess.ovh/tables/standard/.

## Lichess bot

`lichess_bot.py` connects the engine to a Lichess **bot account**.

1. Create a Lichess account for the bot (it must have played no games), create an API token with
   the `bot:play` scope, and upgrade the account to a bot account
   ([Lichess bot API docs](https://lichess.org/api#tag/Bot)).
2. Put the token in a file named `token.txt` next to the script. **`token.txt` is in
   `.gitignore`: never commit it.**
3. `pip install berserk`, then run `python3 lichess_bot.py`.

The bot accepts standard-chess challenges with at least 1 minute of initial time
(`MIN_INITIAL_SECONDS`), plays one game per CPU core at most, and keeps a small reserve on the clock for network lag (`LAG_RESERVE_MS`). Settings are
at the top of the script.

## How it works

**Evaluation.** Stockfish 17.1's main network: HalfKAv2_hm features (2 x 22,528 inputs),
a 2 x 3072 feature transformer with 8 PSQT buckets, and 8 layer stacks (3072 -> 16 -> 32 -> 1).
Accumulators are updated incrementally and lazily (only for positions that actually get
evaluated), with a per-king-square refresh cache. The weights are memory-mapped, so several
engine processes share one copy in RAM. The final score uses Stockfish's complexity and material
scaling and its win-rate normalisation to centipawns. The only non-network term is a small
"mop-up" bonus for bare-king endgames (KQK, KRK, KBBK, KBNK), where the network gives no direction
toward mate.

**Search.** Own bitboard move generator (move order matches python-chess exactly), iterative
deepening with aspiration windows, principal variation search, transposition table (Zobrist keys),
null-move pruning, reverse futility pruning, futility pruning, late move reductions, internal
iterative reduction, check extensions, quiescence search with SEE and delta pruning (plus checks
at the first quiescence ply), killer moves, countermoves, history with gravity, and dynamic time
management (more time when the best move is unstable or the score drops).

## Benchmarking

`match.py` plays engine-vs-engine matches with Fischer clocks, paired openings (each opening is
played twice with colours swapped) and crash-safe results (re-run the same command to resume).

```bash
# 1000 games at 10+0.1, 4 at a time
python3 match.py --engine1 "python3 old_engine.py" --engine2 "python3 engine.py" \
    --games 1000 --base 10 --inc 0.1 --random-openings --jobs 4 --output results.txt

# sequential test (SPRT): stops as soon as engine2 is proven better (or not)
python3 match.py --engine1 "python3 old_engine.py" --engine2 "python3 engine.py" \
    --games 1000 --base 10 --inc 0.1 --random-openings --jobs 4 \
    --sprt --elo0 0 --elo1 10 --output sprt.txt

# Elo with a 95% confidence interval from a results file
python3 elo_ci.py results.txt
```

Use at most one parallel game per CPU core (`--jobs`), and don't run other heavy programs
during a match: CPU contention makes the results noisy.

Ideas tested and rejected (no measurable gain in self-play): late move pruning, singular
extensions, Stockfish "optimism", network-guided move ordering, batched evaluation as a
search change, and several pruning variants.

## Credits

- **Network:** `nn-1c0000000000.nnue` by the Stockfish developers, released under
  [CC0](https://github.com/official-stockfish/networks) (public domain).
- **Evaluation pipeline:** follows the Stockfish 17.1 source
  ([official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish), GPL-3.0).
- **python-chess** by Niklas Fiekas (GPL-3.0) for UCI parsing, tablebase probing and match tools.

This project is not affiliated with or endorsed by the Stockfish project.

## License

GPL-3.0 (see `LICENSE`), matching Stockfish and python-chess, whose work this engine builds on.

# AttackerBotPython
The python bot that will put to shame other little c++ bots.
