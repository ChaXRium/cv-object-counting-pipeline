from pathlib import Path
import random
import shutil
import argparse
import csv
import sys


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def gather_images(root: Path):
    classes = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            images = [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
            if images:
                classes.append((p.name, images))
    return classes


def make_split(files, ratios, rng):
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def ensure_dirs(base: Path, splits, class_name):
    for s in splits:
        d = base / s / class_name
        d.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        writer.writerows(rows)


def split_dataset(input_dir: Path, output_dir: Path, train=0.7, val=0.15, test=0.15, seed=42, move=False):
    rng = random.Random(seed)
    classes = gather_images(input_dir)
    if not classes:
        print(f'No image classes found in {input_dir!s}', file=sys.stderr)
        return

    ratios = (train, val, test)
    splits = ['train', 'val', 'test']
    counts = {s: 0 for s in splits}
    csv_rows = {s: [] for s in splits}

    for class_name, images in classes:
        ensure_dirs(output_dir, splits, class_name)
        tr, va, te = make_split(images, ratios, rng)

        for src in tr:
            dst = output_dir / 'train' / class_name / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            counts['train'] += 1
            csv_rows['train'].append([str(dst.relative_to(output_dir)), class_name])

        for src in va:
            dst = output_dir / 'val' / class_name / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            counts['val'] += 1
            csv_rows['val'].append([str(dst.relative_to(output_dir)), class_name])

        for src in te:
            dst = output_dir / 'test' / class_name / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            counts['test'] += 1
            csv_rows['test'].append([str(dst.relative_to(output_dir)), class_name])

    # write csvs
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'train.csv', csv_rows['train'])
    write_csv(output_dir / 'val.csv', csv_rows['val'])
    write_csv(output_dir / 'test.csv', csv_rows['test'])

    print('Split complete:')
    for s in splits:
        print(f'  {s}: {counts[s]} images')
    print(f'CSV files written to {output_dir!s}')


def parse_args():
    p = argparse.ArgumentParser(description='Stratified train/val/test splitter for image datasets')
    p.add_argument('--input-dir', '-i', required=True, help='Root folder with class subfolders containing images')
    p.add_argument('--output-dir', '-o', required=True, help='Output folder to create train/val/test subfolders')
    p.add_argument('--train', type=float, default=0.7, help='Train ratio (default 0.7)')
    p.add_argument('--val', type=float, default=0.15, help='Validation ratio (default 0.15)')
    p.add_argument('--test', type=float, default=0.15, help='Test ratio (default 0.15)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    p.add_argument('--move', action='store_true', help='Move files instead of copying')
    return p.parse_args()


def main():
    args = parse_args()
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print('Error: train+val+test ratios must sum to 1.0', file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        print(f'Input folder not found: {input_dir!s}', file=sys.stderr)
        sys.exit(1)

    split_dataset(input_dir, output_dir, train=args.train, val=args.val, test=args.test, seed=args.seed, move=args.move)


if __name__ == '__main__':
    main()
