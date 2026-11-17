"""Fetch the frozen he1 code (holdout_era.py, wait_public.py) from commit 15dfd9a into a folder and
record their sha256, so he2 reuses exactly the published he1 data code.

MV2 of Munim, Inc. (made with AI agents working in his name). Apache-2.0.
Usage: python fetch_frozen.py OUT_DIR
"""
import hashlib
import json
import os
import sys
import urllib.request

COMMIT = "15dfd9adeaee03f0fd920a9ac70d96ed9255f5ff"
BASE = "https://raw.githubusercontent.com/munimtechnologies/kaggle-holdout-era/%s/" % COMMIT
FILES = ("holdout_era.py", "wait_public.py")
USER_AGENT = "MV2-Munim-Inc-research (mv2@munimtech.com)"


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rec = {"commit": COMMIT, "files": {}}
    for name in FILES:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(body)
        rec["files"][name] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    with open(os.path.join(out_dir, "frozen.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
