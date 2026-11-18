import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

spec=importlib.util.spec_from_file_location("prepare",Path(__file__).resolve().parents[1]/"training"/"prepare.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_segment_polygon(tmp_path):
    p=tmp_path/"a.json"
    p.write_text(json.dumps({"imageWidth":100,"imageHeight":100,"shapes":[{"label":"person","points":[[10,10],[30,10],[30,30],[10,30]],"shape_type":"polygon"}]}))
    r=mod.convert_labelme(p,["person"])
    assert len(r)==1
    assert len(r[0].split())==9
    assert r[0].startswith("0 0.100000")

def test_segment_unknown_class_rejected(tmp_path):
    p=tmp_path/"a.json"
    p.write_text(json.dumps({"imageWidth":100,"imageHeight":100,"shapes":[{"label":"rat","points":[[1,1],[9,1],[9,9]]}]}))
    with pytest.raises(ValueError,match="unknown label"):
        mod.convert_labelme(p,["person"])

def test_voc_detect(tmp_path):
    p=tmp_path/"a.xml"
    root=ET.Element("annotation")
    sz=ET.SubElement(root,"size")
    ET.SubElement(sz,"width").text="100"
    ET.SubElement(sz,"height").text="100"
    ob=ET.SubElement(root,"object")
    ET.SubElement(ob,"name").text="person"
    bb=ET.SubElement(ob,"bndbox")
    for k,v in [("xmin",10),("ymin",20),("xmax",30),("ymax",40)]:
        ET.SubElement(bb,k).text=str(v)
    ET.ElementTree(root).write(p)
    assert mod.convert_voc(p,["person"])==["0 0.200000 0.300000 0.200000 0.200000"]

def test_empty_dataset_rejected(tmp_path):
    src=tmp_path/"src"
    src.mkdir()
    with pytest.raises(ValueError,match="at least 2"):
        mod.prepare(src,tmp_path/"dst","detect",["person"])

def test_prepare_split(tmp_path):
    src=tmp_path/"src";src.mkdir()
    for i in range(10):
        (src/f"{i}.jpg").write_bytes(b"fake")
        (src/f"{i}.json").write_text(json.dumps({"imageWidth":100,"imageHeight":100,"shapes":[{"label":"person","points":[[1,1],[9,1],[9,9]]}]}))
    out=tmp_path/"out"
    result=mod.prepare(src,out,"segment",["person"])
    assert (result["train"],result["val"])==(9,1)
    assert len(list((out/"labels"/"train").glob("*.txt")))==9
    assert len(list((out/"labels"/"val").glob("*.txt")))==1
