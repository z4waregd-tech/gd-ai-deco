"""
Dataset Diagnostic Tool
Checks each level's dataset JSON and inspects objects by their layer property.
In GD: Layer 1 = Gameplay/Hitboxes, Layer 2+ = Decoration
"""
import json
from pathlib import Path
from collections import Counter

DATASETS_DIR = Path("datasets")

def inspect_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    level_name = data.get("level", json_file.stem)
    gameplay = data.get("gameplay", [])
    deco = data.get("deco", [])
    
    print(f"\n{'='*50}")
    print(f"Level: {level_name}")
    print(f"{'='*50}")
    print(f"  Gameplay objects : {len(gameplay)}")
    print(f"  Deco objects     : {len(deco)}")
    
    # Check layer distribution in gameplay
    gp_layers = Counter(obj.get("20", "1") for obj in gameplay)
    deco_layers = Counter(obj.get("20", "1") for obj in deco)
    
    print(f"\n  --- Gameplay Layer Distribution (key '20') ---")
    for layer, count in sorted(gp_layers.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99):
        label = " <-- should be layer 1!" if layer == "1" else (" <-- WRONG! deco in gameplay?" if layer != "1" else "")
        print(f"    Layer {layer:>4}: {count} objects{label}")

    print(f"\n  --- Deco Layer Distribution (key '20') ---")
    for layer, count in sorted(deco_layers.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99):
        label = " <-- WRONG! gameplay in deco?" if layer == "1" else ""
        print(f"    Layer {layer:>4}: {count} objects{label}")

    # Sample a few gameplay and deco objects to visually inspect
    print(f"\n  --- Sample Gameplay Objects (first 3) ---")
    for obj in gameplay[:3]:
        obj_id = obj.get("1", "?")
        x = obj.get("2", "?")
        y = obj.get("3", "?")
        layer = obj.get("20", "1")
        group = obj.get("57", "-")
        print(f"    ID:{obj_id:>6}  X:{float(x):>8.1f}  Y:{float(y):>7.1f}  Layer:{layer}  Groups:{group}")

    print(f"\n  --- Sample Deco Objects (first 3) ---")
    for obj in deco[:3]:
        obj_id = obj.get("1", "?")
        x = obj.get("2", "?")
        y = obj.get("3", "?")
        layer = obj.get("20", "1")
        group = obj.get("57", "-")
        print(f"    ID:{obj_id:>6}  X:{float(x):>8.1f}  Y:{float(y):>7.1f}  Layer:{layer}  Groups:{group}")

print("GD AI Decorator - Dataset Inspector")
print("Checking layer distributions in each dataset...\n")

for json_file in sorted(DATASETS_DIR.glob("*.json")):
    inspect_dataset(json_file)

print(f"\n{'='*50}")
print("DONE! Check the layer distributions above.")
print("Gameplay objects should mostly be on layer 1.")
print("Deco objects should mostly be on layer 2+.")
print("If gameplay has many non-layer-1 objects, the split is wrong!")
