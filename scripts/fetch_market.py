"""Membership and prices into cache/market, with hashes. Run once from the repository root:
    uv run --with yfinance python scripts/fetch_market.py
Prices never go into git: Yahoo's terms, and the repository may be public. Results replay against this snapshot."""
import hashlib, json, logging, subprocess, time, urllib.parse, urllib.request, warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
OUT = Path("cache/market")
OUT.mkdir(parents=True, exist_ok=True)
REPO, NAME = "fja05680/sp500", "S&P 500 Historical Components & Changes (Updated).csv"
commit = subprocess.run(["gh", "api", f"repos/{REPO}/commits/master", "--jq", ".sha"], capture_output=True, text=True, check=True).stdout.strip()
url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{urllib.parse.quote(NAME)}"
(OUT / "membership.csv").write_bytes(urllib.request.urlopen(url, timeout=120).read())
import csv
csv.field_size_limit(10 ** 7)
rows = list(csv.DictReader((OUT / "membership.csv").open()))
ticks = sorted({t.replace(".", "-") for r in rows if r["date"] >= "2004-01-01" for t in r["tickers"].split(",")} | {"RSP", "SPY"})
parts = []
for i in range(0, len(ticks), 100):
    d = yf.download(ticks[i:i + 100], start="2004-01-01", end="2026-10-31", auto_adjust=True, progress=False, threads=False)
    parts.append(d["Close"])
    print(i, int(d["Close"].notna().any().sum()), flush=True)
    time.sleep(3)
close = pd.concat(parts, axis=1)
close = close.loc[:, close.notna().any()].sort_index(axis=1)
close.index.name = "date"
close.to_csv(OUT / "close.csv.gz", compression={"method": "gzip", "mtime": 0})
sha = lambda p: hashlib.sha256((OUT / p).read_bytes()).hexdigest()
manifest = dict(source=REPO, file=NAME, commit=commit, membership_sha256=sha("membership.csv"), close_sha256=sha("close.csv.gz"), requested=len(ticks), priced=close.shape[1],
                first=str(close.index[0].date()), last=str(close.index[-1].date()), fetched=time.strftime("%Y-%m-%d"))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(manifest)
