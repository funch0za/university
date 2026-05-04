import os, sys, shutil, glob
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

for d in ['train_annotation/images', 'train_annotation/labels',
          'test_annotation/images', 'test_annotation/labels']:
    os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)

train_json = None
val_json = None

for name in ['train_anno.json', 'train_anno_reduced.json']:
    for base in [PROJECT_DIR, DATA_DIR]:
        path = os.path.join(base, name)
        if os.path.exists(path) and train_json is None:
            train_json = path

for name in ['val_anno.json']:
    for base in [PROJECT_DIR, DATA_DIR]:
        path = os.path.join(base, name)
        if os.path.exists(path) and val_json is None:
            val_json = path

if not train_json or not val_json:
    print(f'JSON not found: train={train_json}, val={val_json}')
    exit(1)

print(f'Train: {os.path.basename(train_json)}')
print(f'Val:   {os.path.basename(val_json)}')

shutil.copy(train_json, os.path.join(PROJECT_DIR, 'train_annotation', 'train_anno.json'))
shutil.copy(val_json, os.path.join(PROJECT_DIR, 'test_annotation', 'val_anno.json'))

sys.path.insert(0, os.path.join(PROJECT_DIR, 'JSON2YOLO'))
from general_json2yolo import convert_coco_json

for split, folder in [('train', 'train_annotation'), ('val', 'test_annotation')]:
    print(f'Converting {split}...')
    convert_coco_json(os.path.join(PROJECT_DIR, folder))
    src = os.path.join(PROJECT_DIR, 'new_dir', 'labels',
                       'train_anno' if split == 'train' else 'val_anno')
    dst = os.path.join(PROJECT_DIR, folder, 'labels')
    for f in tqdm(os.listdir(src), desc=f'{split} labels'):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

patterns = [
    os.path.join(DATA_DIR, 'rtsd-frames', 'rtsd-frames', '*.jpg'),
    os.path.join(DATA_DIR, 'rtsd-frames', '*.jpg'),
    os.path.join(PROJECT_DIR, 'rtsd-frames', 'rtsd-frames', '*.jpg'),
]
frames = [f for p in patterns for f in glob.glob(p)]
print(f'Images found: {len(frames)}')

if not frames:
    print('No images found')
    exit(1)

for split, folder in [('train', 'train_annotation'), ('val', 'test_annotation')]:
    labels_dir = os.path.join(PROJECT_DIR, folder, 'labels')
    images_dir = os.path.join(PROJECT_DIR, folder, 'images')
    labels_set = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}

    copied = 0
    for img in frames:
        name = os.path.splitext(os.path.basename(img))[0]
        if name in labels_set:
            dst_path = os.path.join(images_dir, os.path.basename(img))
            if not os.path.exists(dst_path):
                shutil.copy2(img, dst_path)
                copied += 1
    print(f'{split}: {copied} images copied')

new_dir = os.path.join(PROJECT_DIR, 'new_dir')
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)

train_n = len(os.listdir(os.path.join(PROJECT_DIR, 'train_annotation/images')))
val_n = len(os.listdir(os.path.join(PROJECT_DIR, 'test_annotation/images')))
print(f'Done: train={train_n}, val={val_n}')
