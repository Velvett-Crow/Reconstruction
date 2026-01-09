'''Creating the .txt file for C3VDv2 dataset used during fine-tuning'''

import argparse, os, re, glob, random

# Patterns to extract the numeric frame ID
RGB_RE = re.compile(r'^(\d+)\.png$')
DEPTH_RE = re.compile(r'^(\d+)_depth\.tiff$')

def index_by_id(folder, pattern):
    index = {}
    for p in glob.glob(os.path.join(folder, "*")):
        name = os.path.basename(p)
        m = pattern.match(name)
        if m:
            index[int(m.group(1))] = os.path.abspath(p)
    return index

def write_list(path, pairs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rgb, depth in pairs:
            f.write(f"{rgb} {depth}\n")

def main():
    ap = argparse.ArgumentParser(description="Create C3VDv2 val & test filelists")
    ap.add_argument("--rgb_dir", required=True, help="Path to RGB folder")
    ap.add_argument("--depth_dir", required=True, help="Path to Depth folder (.tiff)")
    ap.add_argument("--out_dir", default="dataset/splits/c3vdv2")
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=82)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rgb_map = index_by_id(args.rgb_dir, RGB_RE)
    depth_map = index_by_id(args.depth_dir, DEPTH_RE)

    common_ids = sorted(set(rgb_map).intersection(depth_map))
    
    # Pair RGB with Depth using the same ID
    pairs = [(rgb_map[i], depth_map[i]) for i in common_ids]

    random.seed(args.seed)
    random.shuffle(pairs)

    # Build splits
    n_val = min(args.n_val, len(pairs))
    n_test = min(args.n_test, len(pairs) - n_val)

    val_pairs = pairs[:n_val]
    test_pairs = pairs[n_val:n_val + n_test]
    all_pairs = pairs[:]

    # Save
    write_list(os.path.join(args.out_dir, "val.txt"), val_pairs)
    write_list(os.path.join(args.out_dir, "test.txt"), test_pairs)
    write_list(os.path.join(args.out_dir, "all.txt"), all_pairs)

    print(f"C3VDv2 filelists created successfully!")
    print(f"Total matched pairs: {len(pairs)}")
    print(f"Val:  {len(val_pairs)}  to {os.path.join(args.out_dir, 'val.txt')}")
    print(f"Test: {len(test_pairs)} to {os.path.join(args.out_dir, 'test.txt')}")
    print(f"All:  {len(all_pairs)}  to {os.path.join(args.out_dir, 'all.txt')}")

if __name__ == "__main__":
    main()
