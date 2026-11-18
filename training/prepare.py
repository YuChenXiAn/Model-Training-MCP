#!/usr/bin/env python3
"""Prepare YOLO detect from VOC XML or YOLO segment from LabelMe JSON."""
import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def convert_voc(path, classes):
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("VOC annotation missing size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    if width <= 0 or height <= 0:
        raise ValueError("invalid width/height")
    lines = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        if label not in classes:
            raise ValueError("unknown label: " + label)
        bb = obj.find("bndbox")
        if bb is None:
            raise ValueError("missing bndbox")
        x1 = max(0.0, min(float(bb.findtext("xmin")), width))
        x2 = max(0.0, min(float(bb.findtext("xmax")), width))
        y1 = max(0.0, min(float(bb.findtext("ymin")), height))
        y2 = max(0.0, min(float(bb.findtext("ymax")), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid bbox")
        vals = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
        lines.append(str(classes.index(label)) + " " + " ".join(f"{v:.6f}" for v in vals))
    return lines

def convert_labelme(path, classes):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    width, height = int(data["imageWidth"]), int(data["imageHeight"])
    if width <= 0 or height <= 0:
        raise ValueError("invalid width/height")
    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in classes:
            raise ValueError("unknown label: " + label)
        typ = shape.get("shape_type", "polygon")
        pts = shape.get("points", [])
        if typ == "rectangle" and len(pts) == 2:
            (x1,y1),(x2,y2) = pts
            pts = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        if len(pts) < 3:
            raise ValueError("polygon must have >=3 points")
        xy = [(max(0.,min(float(x),width))/width, max(0.,min(float(y),height))/height) for x,y in pts]
        area = abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[(i+1)%len(xy)][0]*xy[i][1] for i in range(len(xy)))/2)
        if area <= 1e-9:
            raise ValueError("zero-area polygon")
        lines.append(str(classes.index(label)) + " " + " ".join(f"{v:.6f}" for pt in xy for v in pt))
    return lines

def prepare(source, output, task, classes, seed=42, val_ratio=0.1):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("output must differ from source")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be nonempty and unique")
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio must be in (0,1)")
    suffix = ".xml" if task == "detect" else ".json"
    converter = convert_voc if task == "detect" else convert_labelme
    samples = []
    errors = []
    seen = set()
    for ann in sorted(source.rglob("*" + suffix)):
        if output.resolve() in ann.resolve().parents:
            continue
        img = next((ann.with_suffix(e) for e in IMAGE_EXTS if ann.with_suffix(e).is_file()), None)
        if img is None:
            errors.append(f"{ann}: matching image missing")
            continue
        try:
            lines = converter(ann, classes)
            key = str(ann.relative_to(source).with_suffix("")).replace("/", "__")
            if key in seen:
                raise ValueError("duplicate key: " + key)
            seen.add(key)
            samples.append((key, img, lines))
        except (ValueError, KeyError, TypeError, ET.ParseError, json.JSONDecodeError) as exc:
            errors.append(f"{ann}: {exc}")
    if errors:
        raise ValueError("invalid dataset:\n" + "\n".join(errors[:30]))
    if len(samples) < 2:
        raise ValueError("at least 2 annotated image samples required")
    random.Random(seed).shuffle(samples)
    n_val = max(1, round(len(samples)*val_ratio))
    n_val = min(n_val, len(samples)-1)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}; use a new output directory")
    for i,(key,img,lines) in enumerate(samples):
        split = "val" if i < n_val else "train"
        idir, ldir = output/"images"/split, output/"labels"/split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, idir/(key+img.suffix.lower()))
        (ldir/(key+".txt")).write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
    import yaml
    cfg = {"path":str(output.resolve()), "train":"images/train", "val":"images/val",
           "nc":len(classes), "names":{i:n for i,n in enumerate(classes)}}
    (output/"dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"train":len(samples)-n_val, "val":n_val, "classes":classes, "task":task}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["detect","segment"],required=True)
    p.add_argument("--source",default="/home/data")
    p.add_argument("--output",required=True)
    p.add_argument("--classes",nargs="+",required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--val-ratio",type=float,default=0.1)
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output,a.task,a.classes,a.seed,a.val_ratio),ensure_ascii=False))

if __name__ == "__main__":
    main()
