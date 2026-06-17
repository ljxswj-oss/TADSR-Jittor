#!/usr/bin/env python3
from __future__ import annotations
import json, site, sys
from pathlib import Path

def existing(paths):
    return [str(p) for p in paths if p.exists()]

site_paths = []
try:
    site_paths += site.getsitepackages()
except Exception:
    pass
try:
    site_paths.append(site.getusersitepackages())
except Exception:
    pass

candidates_h = [Path("/usr/local/cuda/include/cudnn.h"), Path("/usr/include/cudnn.h"), Path("/mnt/data/sj/local/cudnn/include/cudnn.h")]
candidates_lib = [Path("/mnt/data/sj/local/cudnn/lib/libcudnn.so"), Path("/usr/local/cuda/lib64/libcudnn.so")]
for sp in site_paths:
    base = Path(sp)
    candidates_h.append(base / "nvidia/cudnn/include/cudnn.h")
    candidates_lib.append(base / "nvidia/cudnn/lib/libcudnn.so")
    candidates_lib.append(base / "nvidia/cudnn/lib/libcudnn.so.9")

found_h = existing(candidates_h)
found_lib = existing(candidates_lib)
recommend = []
if found_h:
    inc = str(Path(found_h[0]).parent)
    recommend += [
        f"export CPATH={inc}:$CPATH",
        f"export C_INCLUDE_PATH={inc}:$C_INCLUDE_PATH",
        f"export CPLUS_INCLUDE_PATH={inc}:$CPLUS_INCLUDE_PATH",
    ]
if found_lib:
    lib = str(Path(found_lib[0]).parent)
    recommend.append(f"export LD_LIBRARY_PATH={lib}:$LD_LIBRARY_PATH")

out = {
    "python": sys.executable,
    "site_paths": site_paths,
    "checked_headers": [str(p) for p in candidates_h],
    "checked_libs": [str(p) for p in candidates_lib],
    "found_headers": found_h,
    "found_libs": found_lib,
    "recommend_exports": recommend,
    "cudnn_ready": bool(found_h and found_lib),
}
Path("experiments/full_repro/gpu_setup").mkdir(parents=True, exist_ok=True)
Path("experiments/full_repro/gpu_setup/cudnn_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
