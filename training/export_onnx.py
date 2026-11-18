#!/usr/bin/env python3
"""Export best.pt; verify file creation but do not assume YOLOv8-compatible decoding."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",required=True)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--opset",type=int,default=12)
    a=p.parse_args()
    w=Path(a.weights)
    if not w.is_file() or w.suffix!=".pt":
        raise FileNotFoundError("a valid best.pt checkpoint is required")
    from ultralytics import YOLO
    path=Path(YOLO(str(w)).export(format="onnx",imgsz=a.imgsz,opset=a.opset,nms=False))
    if not path.exists() or path.stat().st_size==0:
        raise RuntimeError("ONNX export failed")
    print(f"EXPORTED_ONNX={path}")
    try:
        import onnx
        model=onnx.load(str(path))
        onnx.checker.check_model(model)
        print("ONNX_CHECK=passed")
        print("INPUTS",[(i.name,[str(d.dim_value or d.dim_param) for d in i.type.tensor_type.shape.dim]) for i in model.graph.input])
        print("OUTPUTS",[(o.name,[str(d.dim_value or d.dim_param) for d in o.type.tensor_type.shape.dim]) for o in model.graph.output])
    except ImportError:
        print("ONNX_CHECK=skipped (install onnx for shape and graph verification)")

if __name__=="__main__":
    main()
