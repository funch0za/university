#python3 -m venv venv
#source ./venv/bin/active

#!/bin/bash
set -e

START_STAGE="${1:-1}"

if [ "$START_STAGE" -eq 0 ]; then
    echo "---------------------- CLEAN ALL ----------------------"
	rm -rf ./JSON2YOLO ./yolov5 ./test_annotation ./train_annotation ./trafic_signs.yaml
	exit
fi

if [ "$START_STAGE" -le 1 ]; then
    echo "---------------------- PART 1 ----------------------"
    echo "Install python packages..."
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
fi

if [ "$START_STAGE" -le 2 ]; then
    echo "---------------------- PART 2 ----------------------"
    echo "Preparing dataset..."
    #mkdir -p ./data
    #unzip rtsd-dataset.zip ./data
fi

if [ "$START_STAGE" -le 3 ]; then
    echo "---------------------- PART 3 ----------------------"
    echo "Preparing to work with the YOLO model..."
    git clone https://github.com/ultralytics/JSON2YOLO
    git clone https://github.com/ultralytics/yolov5
fi

if [ "$START_STAGE" -le 4 ]; then
    echo "---------------------- PART 4 ----------------------"
    echo "Converting coco format to yolo format..."
    echo "Patching JSON2YLOLO..."
    python3 ./src/patch_json2yolo.py
    echo "Converting..."
    python3 ./src/convert_coco_to_yolo.py
fi

if [ "$START_STAGE" -le 5 ]; then
    echo "---------------------- PART 5 ----------------------"
	echo "Creating yaml file for YOLO..."
    python3 ./src/create_yaml.py
fi

if [ "$START_STAGE" -le 6 ]; then
    echo "---------------------- PART 6 ----------------------"
	echo "Training..."

	python yolov5/train.py \
		--img 640 \
		--batch 16 \
		--epochs 3 \
		--data trafic_signs.yaml \
		--weights yolov5s.pt \
		--project hackaton_trafic_signs \
		--name check_run \
		--device mps \
		--workers 2
	exit
    python yolov5/train.py \
		--img 1280 \
		--batch 8 \
		--epochs 40 \
		--data trafic_signs.yaml \
		--weights yolov5m6.pt \
		--project coursework_trafic_signs \
		--name yolov5m6_results \
		--device mps \
		--workers 4 \
		--cache ram \
		--cos-lr
fi

# обучение
# хз, тут мини фронтенд надо написать мб
# типо режим проверки или полное обучение
# python3 ./src/yolov5/train.py -blablabla

# тестирование
# python3 ./src/yolov5/detect.py -blablabla

# возможность загрузить СВОЕ изображение
# python3 ./src/yolov5/detect.py -blablabla


# ???????????????????????????
# echo "📄 Создание solution.csv..."                                                                                                   
#python3 "$SRC_DIR/postprocess.py"      
