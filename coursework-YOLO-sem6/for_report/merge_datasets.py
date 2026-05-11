import os, shutil, random
from pathlib import Path

random.seed(42)

BASE = Path.cwd()
SIGNS = BASE / "data_trafic_signs"
OBJECTS = BASE / "data_trafic_objects" / "export"
OUT = BASE / "data_all"
OFFSET = 155

for d in ["images/train", "labels/train", "images/val", "labels/val"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

print("Copying signs...")
for split in ["train", "val"]:
    for f in (SIGNS / "images" / split).iterdir():
        shutil.copy2(f, OUT / "images" / split / f.name)
    for f in (SIGNS / "labels" / split).iterdir():
        shutil.copy2(f, OUT / "labels" / split / f.name)

print("Copying objects...")
all_images = list((OBJECTS / "images").glob("*.jpg"))
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
splits = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

for split, images in splits.items():
    for img in images:
        name = img.stem
        lbl = OBJECTS / "labels" / f"{name}.txt"
        shutil.copy2(img, OUT / "images" / split / img.name)
        if lbl.exists():
            new_lines = []
            with open(lbl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    parts[0] = str(int(parts[0]) + OFFSET)
                    new_lines.append(" ".join(parts))
            with open(OUT / "labels" / split / f"{name}.txt", "w") as f:
                f.write("\n".join(new_lines))

print(f"Train: {len(list((OUT/'images/train').iterdir()))}")
print(f"Val:   {len(list((OUT/'images/val').iterdir()))}")
