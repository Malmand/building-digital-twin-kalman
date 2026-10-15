"""Download the CU-BEMS files from figshare, one at a time, with retries and a log of every attempt."""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FILES = {"2018Floor1.csv": 21350142, "2018Floor2.csv": 21350166, "2018Floor3.csv": 21350151,
         "2018Floor4.csv": 21350157, "2018Floor5.csv": 21350148, "2018Floor6.csv": 21350154,
         "2018Floor7.csv": 21350145, "2019Floor1.csv": 21350208, "2019Floor2.csv": 21350226,
         "2019Floor3.csv": 21350193, "2019Floor4.csv": 21350172, "2019Floor5.csv": 21350205,
         "2019Floor6.csv": 21350169, "2019Floor7.csv": 21350217}
UA = "Mozilla/5.0 (X11; Linux x86_64) research-data-download"

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name, fid in FILES.items():
    url = f"https://ndownloader.figshare.com/files/{fid}"
    for attempt in range(1, 9):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(out / name, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            print(f"{name}: {(out / name).stat().st_size} bytes (attempt {attempt}, final URL {r.url[:60]})", flush=True)
            break
        except urllib.error.HTTPError as e:
            print(f"{name}: attempt {attempt}: HTTP {e.code} {e.reason}; headers {dict(e.headers)}", flush=True)
        except Exception as e:                     # noqa: BLE001
            print(f"{name}: attempt {attempt}: {e!r}", flush=True)
        time.sleep(15 * attempt)  # linear back-off
    time.sleep(3)
