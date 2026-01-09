'''Used to make the data splits for the EndoSLAM dataset'''

import argparse, os, re, random, glob

# Pattern to extract ID at the end of filename

RGB_RE = re.compile(r'^image_(\d+)\.png$')
DEPTH_RE = re.compile(r'^aov_image_(\d+)\.png$')

def index_by_id(folder, pattern):
    index = {}
    for f in glob.glob(os.path.join(folder, "*.png")):
        name = os.path.basename(f)
        m = pattern.match(name)
        if m:
            index[int(m.group(1))] = os.path.abspath(f)
    return index

def main():
    ap = argparse.ArgumentParser(description="Create EndoSLAM train/val filelists")
    ap.add_argument("--rgb_dir", required=True, default="/home/jovyan/DAv2_training/Frames")
    ap.add_argument("--depth_dir", required=True, default="/home/jovyan/DAv2_training/Pixelwise_Depths")
    ap.add_argument("--out_dir", default="/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/EndoSLAM")
    ap.add_argument("--val_ratio", type=float, default=0.10)  # 90/10 split
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rgb_map = index_by_id(args.rgb_dir, RGB_RE)
    depth_map = index_by_id(args.depth_dir, DEPTH_RE)

    common_ids = sorted(set(rgb_map).intersection(depth_map))

    # Assert 1-to-1 matching
    if len(rgb_map) != len(depth_map):
        print(f"Warning: RGB={len(rgb_map)}, Depth={len(depth_map)}, matched={len(common_ids)}")
        print("   Some frames do not have matches, unmatched files will be ignored.")

    pairs = [(rgb_map[i], depth_map[i]) for i in common_ids]

    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def write_list(path, items):
        with open(path, "w") as f:
            for r, d in items:
                f.write(f"{r} {d}\n")

    write_list(os.path.join(args.out_dir, "train.txt"), train_pairs)
    write_list(os.path.join(args.out_dir, "val.txt"), val_pairs)

    print(f"EndoSLAM filelists created successfully!")
    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
