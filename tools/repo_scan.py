#!/usr/bin/env python3
"""
Repo scanner for TradingAI_Bot-main
- Finds: missing __init__.py, root-level bot files, 'Close' occurrences, risky df[...] assignments,
  yfinance intraday misuse, duplicate files, likely import collisions.
Run:
    python tools/repo_scan.py
Paste the output back here for a precise patch.
"""
import os, re, sys

ROOT = os.getcwd()

def find_files_with(pattern, exts=(".py",), ignore_dirs=("venv",".venv",".git")):
    out = []
    prog = re.compile(pattern)
    for dp, dns, fns in os.walk(ROOT):
        # ignore common envs
        if any(x in dp for x in ignore_dirs): 
            continue
        for fn in fns:
            if not fn.endswith(exts): 
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                continue
            if prog.search(txt):
                out.append(path)
    return out

def check_init():
    missing = []
    packs = ["src","src/strategies","src/utils","src/core"]
    for p in packs:
        ip = os.path.join(ROOT, p, "__init__.py")
        if not os.path.exists(ip):
            missing.append(ip)
    return missing

def check_root_bot():
    candidates = ["bot.py","Bot.py","main.py"]
    present = []
    for c in candidates:
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            present.append(p)
    # but note: src/main.py is ok; root main.py may conflict
    return present

def check_yf_intraday_usage():
    files = find_files_with(r"yf\.download\(", (".py",))
    risky = []
    for p in files:
        try:
            with open(p,"r",encoding="utf-8") as f:
                s=f.read()
        except Exception:
            continue
        if "interval" in s and ("start=" in s or "end=" in s):
            risky.append(p)
    return risky

def check_close_uppercase():
    return find_files_with(r"\bClose\b", (".py",".ipynb"))

def check_df_assignments():
    # heuristic: df['SOMETHING'] = somefunc(...)
    files = []
    prog = re.compile(r"df\s*\[\s*['\"](?P<col>\w+)['\"]\s*\]\s*=\s*(?P<call>\w+)\(")
    for dp,_,fns in os.walk(ROOT):
        if any(x in dp for x in (".venv","venv",".git")):
            continue
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path,"r",encoding="utf-8") as f:
                    s=f.read()
            except Exception:
                continue
            for m in prog.finditer(s):
                files.append((path, m.group("col"), m.group("call")))
    return files

def check_duplicate_names():
    # same filename in different dirs, may confuse imports
    mapping = {}
    for dp,_,fns in os.walk(ROOT):
        if any(x in dp for x in (".venv","venv",".git")):
            continue
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            mapping.setdefault(fn, []).append(os.path.join(dp, fn))
    dup = {k:v for k,v in mapping.items() if len(v)>1}
    return dup

def main():
    print("=== REPO SCAN START ===\nRoot:", ROOT, "\n")
    missing_inits = check_init()
    if missing_inits:
        print("[!] Missing __init__.py for package dirs (create these):")
        for x in missing_inits: print("  -", x)
    else:
        print("[OK] package __init__.py files look fine or intentionally absent.\n")

    root_bots = check_root_bot()
    if root_bots:
        print("[!] Root-level entry files found (may conflict with src/main.py):")
        for x in root_bots: print("  -", x)
    else:
        print("[OK] No suspicious root-level bot/main files.\n")

    close_files = check_close_uppercase()
    if close_files:
        print("[!] Files referencing 'Close' (uppercase). Standardize to 'close':")
        for x in close_files[:40]: print("  -", x)
    else:
        print("[OK] No 'Close' occurrences found.\n")

    risky_assign = check_df_assignments()
    if risky_assign:
        print("[!] Potential risky df[...] = call(...) assignments (file, column, function):")
        for p,c,call in risky_assign[:60]:
            print(f"  - {p}  assigns column '{c}' from function '{call}()' — verify it returns Series")
    else:
        print("[OK] No obvious risky df assignments detected.\n")

    yf_risky = check_yf_intraday_usage()
    if yf_risky:
        print("[!] yfinance intraday misuse (start/end + interval). Use period='60d' for intraday:")
        for x in yf_risky: print("  -", x)
    else:
        print("[OK] No obvious intraday yfinance misuse found.\n")

    dup = check_duplicate_names()
    if dup:
        print("[!] Duplicate filenames detected (same filename in multiple dirs):")
        for k,v in dup.items():
            print(f"  - {k}:")
            for p in v: print("      ", p)
    else:
        print("[OK] No duplicate filenames found.\n")

    print("=== REPO SCAN END ===")

if __name__ == '__main__':
    main()
