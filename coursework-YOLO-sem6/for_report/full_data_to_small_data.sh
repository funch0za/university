mkdir -p ./data_small/images/train ./data_small/labels/train
mkdir -p ./data_small/images/val ./data_small/labels/val

# 16 000 случайных картинок
find ./train_annotation/images -name "*.jpg" | sort -R | head -n 16000 | while read img; do
    basename=$(basename "$img" .jpg)
    cp "$img" ./data_small/images/train/
    cp "./train_annotation/labels/${basename}.txt" ./data_small/labels/train/ 2>/dev/null || true
done

# Val: взять 2 000 случайных картинок
find ./test_annotation/images -name "*.jpg" | sort -R | head -n 2000 | while read img; do
    basename=$(basename "$img" .jpg)
    cp "$img" ./data_small/images/val/
    cp "./test_annotation/labels/${basename}.txt" ./data_small/labels/val/ 2>/dev/null || true
done

echo "Train images: $(ls ./data_small/images/train/ | wc -l)"
echo "Val images:   $(ls ./data_small/images/val/ | wc -l)"
du -sh ./data_small/
