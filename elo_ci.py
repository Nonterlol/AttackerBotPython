#!/usr/bin/env python3
"""Correct Elo estimate + 95% interval from a results file of match.py.

Works with both the old format ("--- Game i ---" blocks) and the new
crash-safe format ("### GAME" records).  Engine names are read from the PGN
White/Black headers; the second engine (EngineB by default) is reported as
"new" vs the first.

Usage: python3 elo_ci.py results.txt [NEW_NAME]
"""
import math
import re
import sys

path = sys.argv[1]
new = sys.argv[2] if len(sys.argv) > 2 else "EngineB"
text = open(path, encoding="utf-8", errors="replace").read()

w = d = l = 0
for white, black, result in re.findall(
        r'\[White "([^"]*)"\]\s*\[Black "([^"]*)"\][\s\S]*?\[Result "([^"]*)"\]', text):
    if new not in (white, black):
        continue
    if result == "1/2-1/2":
        d += 1
    elif (result == "1-0") == (white == new):
        w += 1
    elif result in ("1-0", "0-1"):
        l += 1
n = w + d + l
if n == 0:
    sys.exit(f"no games with player {new!r} found")
s = (w + d / 2) / n
var = (w + d / 4) / n - s * s
se = math.sqrt(var / n)


def elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / p - 1)


print(f"{new}: W {w}  D {d}  L {l}  ({n} games, draw rate {100 * d / n:.0f}%)")
print(f"score {100 * s:.1f}%  ->  {elo(s):+.1f} Elo, 95% interval "
      f"{elo(s - 1.96 * se):+.1f} .. {elo(s + 1.96 * se):+.1f}")
lo = elo(s - 1.96 * se)
print("significant at 95%" if lo > 0 or elo(s + 1.96 * se) < 0 else "not significant at 95% (interval includes 0)")
