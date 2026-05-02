"""
Dataset Visual Tester
Creates a .gmd file from a dataset JSON where:
  - Gameplay objects are forced to Layer 1 (visible as a distinct color in GD)
  - Deco objects are forced to Layer 2
This lets you visually verify the gameplay/deco split is correct inside GD.

Usage: python test_dataset_visual.py Bloodbath
"""
import json
import sys
import gzip
import base64
import xml.etree.ElementTree as ET
from pathlib import Path


def build_visual_gmd(level_name="Bloodbath"):
    # Load the dataset JSON
    json_path = Path("datasets") / f"{level_name}.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found! Run build_datasets.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gameplay = data.get("gameplay", [])
    deco = data.get("deco", [])
    print(f"Loaded {level_name}: {len(gameplay)} gameplay, {len(deco)} deco")

    # Limit to a safe sample size to avoid crashes from massive levels
    SAMPLE = 500
    gameplay = gameplay[:SAMPLE]
    deco     = deco[:SAMPLE]
    print(f"Sampling first {SAMPLE} of each for safety.")

    # Build the GD level string
    # Start with a minimal valid level header (no speed portals = no time markers crash)
    header = "kS38,1_255_2_0_3_0_6_1000|1_0_2_102_3_255_6_1001|,kA13,0,kA15,0,kA16,0,kA14,0,kS39,0,kA17,0"
    gd_string = header + ";"

    # Add gameplay objects — force Layer 1, give them a RED color (color channel 1000)
    for obj in gameplay:
        obj["20"] = "1"      # Layer 1
        obj["21"] = "1000"   # Color channel 1000 = RED
        obj_str = ",".join(f"{k},{v}" for k, v in obj.items()) + ";"
        gd_string += obj_str

    # Add deco objects — force Layer 2, give them a BLUE color (color channel 1001)
    for obj in deco:
        obj["20"] = "2"      # Layer 2
        obj["21"] = "1001"   # Color channel 1001 = BLUE
        obj_str = ",".join(f"{k},{v}" for k, v in obj.items()) + ";"
        gd_string += obj_str

    # Compress and encode
    compressed = gzip.compress(gd_string.encode("utf-8"))
    b64_out = base64.urlsafe_b64encode(compressed).decode("utf-8")

    # Use the original level's .gmd as XML template (just swap out the level data)
    template_candidates = [
        Path("levels") / level_name / "full.gmd",
        Path("levels") / level_name / "layout.gmd",
        Path("gameplay.gmd"),
    ]
    template = None
    for candidate in template_candidates:
        if candidate.exists():
            template = candidate
            break

    if template is None:
        print("Error: No template .gmd file found to base the output on!")
        return

    tree = ET.parse(template)
    root = tree.getroot()
    dict_node = root.find("dict")
    children = list(dict_node)

    for i in range(len(children) - 1):
        if children[i].tag == "k" and children[i].text == "k4":
            children[i + 1].text = b64_out
            break

    out_path = f"visual_test_{level_name}.gmd"
    tree.write(out_path, encoding="utf-8")
    print(f"\nSaved to: {out_path}")
    print("Import into GD and check:")
    print("  - Layer 1 objects (RED)  = Gameplay / hitboxes")
    print("  - Layer 2 objects (BLUE) = Decoration")
    print("If they look right, the dataset split is working!")


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "Bloodbath"
    build_visual_gmd(level)
