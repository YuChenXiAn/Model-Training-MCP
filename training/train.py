#!/usr/bin/env python3
"""Train and validate YOLO26 via pinned Ultralytics version."""
import argparse
from pathlib import Path

try:
    from .preflight import check
except ImportError:
    from preflight import check

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["detect","segment"],required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--device",default="0")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--project",default=None)
    p.add_argument("--name",default="experiment")
    a=p.parse_args()
    if a.epochs<1 or a.batch<1 or a.imgsz<32:
        p.error("invalid epochs/batch/imgsz")
    print("Preflight:",check(a.task,a.data,a.weights,a.device),flush=True)
    import ultralytics
    if ultralytics.__version__!="8.4.78":
        raise RuntimeError(f"requires ultralytics==8.4.78; installed {ultralytics.__version__}")
    from ultralytics import YOLO
    model=YOLO(a.weights,task=a.task)
    if model.task!=a.task:
        raise ValueError(f"weight task {model.task} mismatches requested {a.task}")
    results=model.train(data=str(Path(a.data).resolve()),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,
                        device=a.device,workers=a.workers,seed=42,project=a.project,name=a.name)
    best=Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError(f"training ended without best checkpoint: {best}")
    print(f"BEST_CHECKPOINT={best}",flush=True)
    print("VALIDATING_CHECKPOINT",flush=True)
    metrics=YOLO(str(best)).val(data=str(Path(a.data).resolve()),device=a.device,imgsz=a.imgsz)
    print(f"VALIDATION_METRICS={metrics.results_dict}",flush=True)
    return results

if __name__=="__main__":
    main()
