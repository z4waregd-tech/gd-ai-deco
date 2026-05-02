"""
Dataset Visual Tester (Fixed)
Instead of building a level from scratch (which crashes),
this script takes the original full.gmd, keeps its valid header,
and just recolors the objects based on gameplay/deco classification:
  - RED  (Color 1000) = Gameplay
  - BLUE (Color 1001) = Decoration

Usage: python test_dataset_visual.py Bloodbath
"""
import json
import sys
import gzip
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from build_datasets import (
    extract_level_data, decode_level_string,
    parse_level_objects, GAMEPLAY_IDS
)


def build_visual_gmd(level_name="Bloodbath"):
    # Paths
    json_path  = Path("datasets") / f"{level_name}.json"
    full_gmd   = Path("levels") / level_name / "full.gmd"
    layout_gmd = Path("levels") / level_name / "layout.gmd"

    if not json_path.exists():
        print(f"Error: {json_path} not found! Run build_datasets.py first.")
        return
    if not full_gmd.exists():
        print(f"Error: {full_gmd} not found!")
        return

    # Load the pre-built dataset to know which objects are gameplay vs deco
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gameplay_list = data.get("gameplay", [])
    deco_list     = data.get("deco", [])
    print(f"Loaded {level_name}: {len(gameplay_list)} gameplay, {len(deco_list)} deco")

    # Build a signature set for gameplay objects
    gameplay_sigs = Counter()
    for obj in gameplay_list:
        sig = (
            round(float(obj.get("2", "0")), 1),
            round(float(obj.get("3", "0")), 1),
        )
        gameplay_sigs[sig] += 1

    # Parse the original full.gmd level string
    _, full_b64 = extract_level_data(full_gmd)
    full_raw    = decode_level_string(full_b64)

    # Split into header + objects
    parts  = full_raw.split(";", 1)
    header = parts[0]

    # Add our two test colors into the header's kS38 channel list
    test_colors = "1_255_2_0_3_0_6_1000|1_0_2_100_3_255_6_1001|"
    if "kS38," in header:
        header = header.replace("kS38,", f"kS38,{test_colors}")
    else:
        header += f",kS38,{test_colors}"

    # Recolor every object based on classification
    all_objects = parse_level_objects(full_raw)
    remaining   = dict(gameplay_sigs)
    new_obj_strings = []

    for obj in all_objects:
        obj_id = obj.get("1", "1")
        sig = (
            round(float(obj.get("2", "0")), 1),
            round(float(obj.get("3", "0")), 1),
        )

        is_gameplay = False
        if obj_id in GAMEPLAY_IDS:
            is_gameplay = True
        elif remaining.get(sig, 0) > 0:
            remaining[sig] -= 1
            is_gameplay = True

        if is_gameplay:
            obj["20"] = "1"      # Layer 1
            obj["21"] = "1000"   # RED
        else:
            obj["20"] = "2"      # Layer 2
            obj["21"] = "1001"   # BLUE

        new_obj_strings.append(",".join(f"{k},{v}" for k, v in obj.items()))

    # Reassemble full level string with original valid header
    final_level_string = header + ";" + ";".join(new_obj_strings) + ";"

    # Compress and encode
    compressed = gzip.compress(final_level_string.encode("utf-8"))
    b64_out    = base64.urlsafe_b64encode(compressed).decode("utf-8")

    # Write back into the original XML structure (preserves all GD metadata)
    tree = ET.parse(full_gmd)
    root = tree.getroot()
    dict_node = root.find("dict")
    children  = list(dict_node)

    for i in range(len(children) - 1):
        if children[i].tag == "k" and children[i].text == "k4":
            children[i + 1].text = b64_out
            break

    out_path = f"visual_test_{level_name}.gmd"
    tree.write(out_path, encoding="utf-8")
    print(f"\nSaved to: {out_path}")
    print("Import into GD. In the editor:")
    print("  - RED  objects (Color 1000) = Gameplay / hitboxes")
    print("  - BLUE objects (Color 1001) = Decoration")


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "Bloodbath"
    build_visual_gmd(level)
