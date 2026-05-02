import torch
import torch.nn.functional as F
from dataset import GDTokenizer
from model import GDEncoderDecoderTransformer
import xml.etree.ElementTree as ET
import base64
import gzip


def generate_deco(theme="Hellish, Red, Demon, 2.1.", max_tokens=1024):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Generating on: {device}")

    # ── 1. Load Tokenizer & Model ─────────────────────────────────────────────
    tokenizer = GDTokenizer()
    try:
        tokenizer.load("vocab.json")
    except FileNotFoundError:
        print("Error: vocab.json not found! Train the model first.")
        return

    vocab_size = len(tokenizer.vocab)
    model = GDEncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        max_seq_len=1024,
    ).to(device)

    try:
        model.load_state_dict(torch.load("gd_decorator_model.pth", map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Error: gd_decorator_model.pth not found! Run train.py first.")
        return

    print("Model loaded successfully!")

    # ── 2. Extract gameplay from gameplay.gmd ─────────────────────────────────
    try:
        from build_datasets import extract_level_data, decode_level_string, parse_level_objects
    except ImportError:
        print("Required functions from build_datasets.py not found.")
        return

    try:
        level_name, b64_data = extract_level_data("gameplay.gmd")
        raw_level_string = decode_level_string(b64_data)
        gameplay_objects = parse_level_objects(raw_level_string)
    except Exception as e:
        print(f"Failed to read gameplay.gmd: {e}")
        return

    print(f"Loaded {len(gameplay_objects)} objects from gameplay.gmd!")

    # ── 3. Build encoder source sequence: theme + gameplay ────────────────────
    src_tokens = ["[THEME]"]
    for word in theme.replace(",", "").split():
        src_tokens.append(f"<T:{word}>")

    src_tokens.append("[GP_START]")
    for obj in gameplay_objects:
        src_tokens.extend(tokenizer.tokenize_object(obj, is_gameplay=True))
    src_tokens.append("[GP_END]")

    # Truncate to max encoder length
    max_src = 512
    src_ids = [tokenizer.vocab.get(t, tokenizer.vocab["[UNK]"]) for t in src_tokens]
    if len(src_ids) > max_src:
        # Keep theme + [GP_START] header, trim gameplay from the middle
        print(f"  (Gameplay too long, trimming to {max_src} tokens)")
        src_ids = src_ids[:max_src]

    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    # ── 4. Encoder: run once, cache memory ───────────────────────────────────
    print("\nEncoding gameplay sequence...")
    with torch.no_grad():
        memory = model.encode(src_tensor)   # [1, src_len, d_model]

    # ── 5. Autoregressive decoder loop ───────────────────────────────────────
    print("Starting decoder generation...")

    deco_start_id = tokenizer.vocab.get("[DECO_START]", 1)
    deco_end_id   = tokenizer.vocab.get("[DECO_END]",   1)

    generated_ids = [deco_start_id]

    with torch.no_grad():
        for i in range(max_tokens):
            tgt_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
            logits = model.decode_step(tgt_tensor, memory)  # [1, tgt_len, vocab]
            next_logits = logits[0, -1, :]                  # last timestep

            # Temperature + Top-P sampling
            temperature = 0.7
            scaled = next_logits / temperature
            top_p = 0.9

            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove_full = remove.scatter(-1, sorted_idx, remove)
            scaled[remove_full] = -float("Inf")

            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == deco_end_id:
                print("\nAI reached [DECO_END]!")
                break

            generated_ids.append(next_id)
            token_str = tokenizer.inverse_vocab.get(next_id, "[UNK]")
            print(token_str, end=" ", flush=True)

    generated_tokens = [tokenizer.inverse_vocab.get(i, "[UNK]") for i in generated_ids[1:]]

    # ── 6. Parse tokens → GD object string ───────────────────────────────────
    print("\n\nParsing and packing into ai_decorated.gmd...")
    gd_string = ""
    current_obj = {}
    channels_dict = {}
    current_ch = None

    for token in generated_tokens:
        if token.startswith("<CH:"):
            current_ch = token[4:-1]
            if current_ch not in channels_dict:
                channels_dict[current_ch] = {"r": "255", "g": "255", "b": "255"}
        elif token.startswith("<CR:") and current_ch:
            channels_dict[current_ch]["r"] = token[4:-1]
        elif token.startswith("<CG:") and current_ch:
            channels_dict[current_ch]["g"] = token[4:-1]
        elif token.startswith("<CB:") and current_ch:
            channels_dict[current_ch]["b"] = token[4:-1]
        elif token == "<OBJ>":
            current_obj = {}
        elif token.startswith("<ID:"):
            current_obj["1"] = token[4:-1]
        elif token.startswith("<X:"):
            current_obj["2"] = token[3:-1]
        elif token.startswith("<Y:"):
            current_obj["3"] = token[3:-1]
        elif token.startswith("<C:"):
            current_obj["21"] = token[3:-1]
        elif token.startswith("<R:"):
            current_obj["6"] = token[3:-1]
        elif token.startswith("<S:"):
            current_obj["32"] = token[3:-1]
        elif token.startswith("<G:"):
            g_id = token[3:-1]
            current_obj["57"] = (current_obj["57"] + f".{g_id}") if "57" in current_obj else g_id
        elif token == "</OBJ>":
            if "1" in current_obj:
                gd_string += ",".join(f"{k},{v}" for k, v in current_obj.items()) + ";"

    # ── 7. Inject color channels & rebuild level string ───────────────────────
    ch_string = ""
    for ch_id, rgb in channels_dict.items():
        ch_string += f"1_{rgb['r']}_2_{rgb['g']}_3_{rgb['b']}_6_{ch_id}|"

    header_parts = raw_level_string.split(";", 1)
    if ch_string:
        if "kS38" in header_parts[0]:
            header_parts[0] = header_parts[0].replace("kS38,", f"kS38,{ch_string}")
        else:
            header_parts[0] += f",kS38,{ch_string}"

    final_level_string = header_parts[0] + ";" + header_parts[1] + gd_string

    compressed = gzip.compress(final_level_string.encode("utf-8"))
    b64_out = base64.urlsafe_b64encode(compressed).decode("utf-8")

    tree = ET.parse("gameplay.gmd")
    root = tree.getroot()
    children = list(root.find("dict"))
    for i in range(len(children) - 1):
        if children[i].tag == "k" and children[i].text == "k4":
            children[i + 1].text = b64_out

    tree.write("ai_decorated.gmd", encoding="utf-8")
    print("\nSUCCESS: Saved fully playable level to ai_decorated.gmd")


if __name__ == "__main__":
    generate_deco(theme="Hellish, Red, Demon")
