import xml.etree.ElementTree as ET
import base64
import gzip
import json
from pathlib import Path


LEVELS_DIR = Path("levels")
DATASETS_DIR = Path("datasets")

DATASETS_DIR.mkdir(exist_ok=True)


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
    # We round X and Y because copy-pasting objects in GD sometimes causes <0.1 precision drift
    x = float(obj.get("2", "0"))
    y = float(obj.get("3", "0"))
    rot = float(obj.get("6", "0"))
    scale = float(obj.get("32", "1"))
    
    return (
        round(x, 1),        # x (rounded to 1 decimal)
        round(y, 1),        # y (rounded to 1 decimal)
        int(round(rot)),    # rotation
        round(scale, 1)     # scale
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
    
    # 1. Gameplay is literally just the layout file! No more signature guessing!
    gameplay = layout_objects
    
    # 2. Deco is the full level. We'll use a spatial hash grid (15 units) to remove exact duplicate layout hitboxes
    layout_grid = set()
    for obj in layout_objects:
        x = float(obj.get("2", "0"))
        y = float(obj.get("3", "0"))
        layout_grid.add((round(x/15.0), round(y/15.0)))
        
    deco = []
    for obj in full_objects:
        x = float(obj.get("2", "0"))
        y = float(obj.get("3", "0"))
        grid_pos = (round(x/15.0), round(y/15.0))
        
        # If this area doesn't have a layout hitbox, it's strictly deco!
        # Even if it does naturally overlap, the generator will handle it beautifully.
        if grid_pos not in layout_grid:
            deco.append(obj)
            
    # For safety, if deco completely empties out due to grid overlaps, fallback to full objects
    if len(deco) < 100:
        deco = full_objects

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


auto_build()