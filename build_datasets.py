import xml.etree.ElementTree as ET
import base64
import gzip
import json
from pathlib import Path
from collections import Counter


LEVELS_DIR = Path("levels")
DATASETS_DIR = Path("datasets")

DATASETS_DIR.mkdir(exist_ok=True)

# Object IDs that are ALWAYS gameplay, never deco (portals, orbs, pads, speed changes, spikes, saws, main blocks, triggers)
GAMEPLAY_IDS = {
    "12", "13", "47", "111", "660", "745", "1331", "1933",  # Game mode portals
    "99", "101",          # Size portals
    "10", "11",           # Gravity portals
    "286", "287",         # Dual portals
    "45", "46",           # Mirror portals
    "200", "201", "202", "203", "1334",  # Speed portals
    "36", "84", "141", "1022", "1333", "1704", "1751",  # Orbs
    "35", "67", "140", "1332", "1594",  # Pads
    "747", "749",         # Teleport portals
    "152", "153", "1708", "1709", # Standard saws
    "61", "142", "143", "551", # Slopes
}


def extract_level_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    dict_node = root.find("dict")
    children = list(dict_node)

    level_data = None
    level_name = "Unknown"

    for i in range(len(children) - 1):
        if children[i].tag == "k":
            key = children[i].text

            if key == "k4":
                level_data = children[i + 1].text

            elif key == "k2":
                level_name = children[i + 1].text

    if level_data is None:
        raise Exception(f"No k4 found in {file_path}")

    return level_name, level_data


def decode_level_string(level_data):
    decoded = base64.urlsafe_b64decode(level_data + "==")
    raw_level = gzip.decompress(decoded).decode("utf-8")
    return raw_level


def parse_pairs(parts):
    data = {}

    for i in range(0, len(parts), 2):
        if i + 1 >= len(parts):
            break

        data[parts[i]] = parts[i + 1]

    return data


def parse_level_objects(raw_level):
    parts = raw_level.split(";")
    object_strings = parts[1:]

    objects = []

    for obj in object_strings:
        if not obj.strip():
            continue

        obj_data = parse_pairs(obj.split(","))
        objects.append(obj_data)

    return objects

def parse_level_header(raw_level):
    header = raw_level.split(";")[0]
    header_parts = header.split(",")

    header_data = parse_pairs(header_parts)

    channels = {}

    if "kS38" in header_data:
        for channel in header_data["kS38"].split("|"):
            if not channel.strip():
                continue

            props = parse_pairs(channel.split("_"))

            channel_id = props.get("6")
            r = props.get("1", "255")
            g = props.get("2", "255")
            b = props.get("3", "255")

            if channel_id:
                channels[channel_id] = {
                    "r": r,
                    "g": g,
                    "b": b
                }

    return channels

def object_signature(obj):
    x = float(obj.get("2", "0"))
    y = float(obj.get("3", "0"))
    rot = float(obj.get("6", "0"))
    scale = float(obj.get("32", "1"))
    
    return (
        round(x, 1),
        round(y, 1),
        int(round(rot)), 
        round(scale, 1)
    )


def build_dataset(level_folder):
    full_file = level_folder / "full.gmd"
    layout_file = level_folder / "layout.gmd"

    if not full_file.exists():
        print(f"[SKIP] Missing full.gmd in {level_folder.name}")
        return

    if not layout_file.exists():
        print(f"[SKIP] Missing layout.gmd in {level_folder.name}")
        return

    level_name, full_data = extract_level_data(full_file)
    _, layout_data = extract_level_data(layout_file)

    full_raw = decode_level_string(full_data)
    layout_raw = decode_level_string(layout_data)

    full_objects = parse_level_objects(full_raw)
    layout_objects = parse_level_objects(layout_raw)

    layout_signatures = {
        object_signature(obj)
        for obj in layout_objects
    }

    theme_file = level_folder / "theme.txt"
    if theme_file.exists():
        with open(theme_file, "r", encoding="utf-8") as f:
            theme = f.read().strip()

    channels = parse_level_header(full_raw)
    
    # 1. Gameplay is the layout file
    gameplay = layout_objects

    # 2. Deco = full level MINUS the layout objects.
    # We match by exact Position (X, Y).
    # We DO NOT match by ID or Rotation anymore!
    # If the layout has a spike at (50, 50), and the decorator replaced it with a
    # "colorable spike" (different ID), it will still successfully match the position!
    layout_signatures = Counter()
    for obj in layout_objects:
        sig = (
            round(float(obj.get("2", "0")), 1),
            round(float(obj.get("3", "0")), 1),
        )
        layout_signatures[sig] += 1

    deco = []
    remaining = dict(layout_signatures)
    for obj in full_objects:
        obj_id = obj.get("1", "1")
        sig = (
            round(float(obj.get("2", "0")), 1),
            round(float(obj.get("3", "0")), 1),
        )

        # Always keep known gameplay objects out of deco
        if obj_id in GAMEPLAY_IDS:
            if remaining.get(sig, 0) == 0:
                gameplay.append(obj)
            else:
                remaining[sig] -= 1
            continue

        if remaining.get(sig, 0) > 0:
            remaining[sig] -= 1
        else:
            deco.append(obj)

    dataset = {
        "level": level_name,
        "theme": theme,
        "channels": channels,
        "gameplay_count": len(gameplay),
        "deco_count": len(deco),
        "gameplay": gameplay,
        "deco": deco
    }

    output_path = DATASETS_DIR / f"{level_folder.name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, separators=(",", ":"))

    print(f"[OK] {level_folder.name}")
    print(f"Gameplay: {len(gameplay)}")
    print(f"Deco: {len(deco)}")
    print()


def auto_build():
    for level_folder in LEVELS_DIR.iterdir():
        if level_folder.is_dir():
            build_dataset(level_folder)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_folder = LEVELS_DIR / sys.argv[1]
        if target_folder.exists() and target_folder.is_dir():
            build_dataset(target_folder)
        else:
            print(f"[ERROR] Level folder not found: {sys.argv[1]}")
    else:
        auto_build()