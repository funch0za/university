echo "Demonstration..."

# проверка что в data/test лежит картинка нужного размера 
# сделать usage

#mkdir -p data/test
#find train_annotation/images -type f -name "*.jpg" | head -10 | xargs -I {} cp {} data/test/

if find ./demo_data -maxdepth 0 -empty | grep -q .; then
    echo "No images in ./demo_data/"
fi

python yolov5/detect.py \
--source ./demo_data \
--weights coursework/coursework_train/weights/best.pt \
--save-txt --save-conf \
--project coursework \
--name demo \
--imgsz 1280 \
--conf-thres 0.25 \
--device mps

