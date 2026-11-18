#!/usr/bin/env python3
"""Stop early unless the task, data, model and training runtime match."""
import argparse
import json
from pathlib import Path

def check(task, data, weights, device="0", require_cuda=True):
    import yaml
    f=Path(data)
    w=Path(weights)
    if not f.is_file():
        raise FileNotFoundError(f"dataset YAML missing: {f}")
    if not w.is_file() or w.stat().st_size == 0:
        raise FileNotFoundError(f"weights missing/empty: {w}")
    if w.suffix != ".pt":
        raise ValueError("train initial weights must be a .pt checkpoint, not ONNX")
    cfg=yaml.safe_load(f.read_text(encoding="utf-8"))
    names=cfg.get("names")
    n=len(names) if isinstance(names,(list,dict)) else 0
    if n == 0 or cfg.get("nc",n) != n:
        raise ValueError("dataset class count / names mismatch")
    root=Path(cfg.get("path") or f.parent)
    if not root.is_absolute():
        root=(f.parent/root).resolve()
    counts={}
    for split in ("train","val"):
        images=root/cfg[split]
        if not images.is_dir():
            raise FileNotFoundError(f"{split} images directory absent: {images}")
        found=[x for x in images.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
        if not found:
            raise ValueError(f"no {split} images: {images}")
        counts[split]=len(found)
        labels=root/"labels"/split
        for img in found:
            txt=labels/(img.stem+".txt")
            if not txt.exists():
                raise FileNotFoundError(f"missing label: {txt}")
            for line in txt.read_text().splitlines():
                parts=line.split()
                need=5 if task=="detect" else 7
                if len(parts)<need or (task=="detect" and len(parts)!=5) or (task=="segment" and (len(parts)-1)%2):
                    raise ValueError(f"invalid {task} label format: {txt}")
                cid=int(parts[0])
                values=[float(x) for x in parts[1:]]
                if not 0<=cid<n or any(not 0<=x<=1 for x in values):
                    raise ValueError(f"invalid class/coordinate: {txt}")
    if require_cuda and str(device).lower() not in ("cpu","mps"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing GPU training")
    return {"task":task,"classes":n,"images":counts,"weights":str(w.resolve()),"device":device}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",required=True,choices=["detect","segment"])
    p.add_argument("--data",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--device",default="0")
    p.add_argument("--allow-no-cuda",action="store_true")
    a=p.parse_args()
    print(json.dumps(check(a.task,a.data,a.weights,a.device,not a.allow_no_cuda)))

if __name__=="__main__":
    main()
