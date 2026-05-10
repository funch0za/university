.PHONY: all clean demo venv data repos convert yaml train detect help

EPOCHS ?= 3
BATCH ?= 16
IMG_SIZE ?= 640
WEIGHTS ?= yolov5s.pt
DEVICE ?= mps
WORKERS ?= 2
CONF_THRES ?= 0.25

VENV = ./venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all: train demo

train: yaml
	@echo "=== TRAINING ==="
	$(PYTHON) yolov5/train.py \
		--img $(IMG_SIZE) \
		--batch $(BATCH) \
		--epochs $(EPOCHS) \
		--data trafic_signs.yaml \
		--weights $(WEIGHTS) \
		--project coursework \
		--name coursework_train \
		--device $(DEVICE) \
		--workers $(WORKERS)

demo: detect

detect: train
	@echo "=== DEMO ==="
	@if [ ! -d ./demo_data ] || [ -z "$$(ls -A ./demo_data 2>/dev/null)" ]; then \
		echo "WARNING: ./demo_data is empty or missing!"; \
	fi
	$(PYTHON) yolov5/detect.py \
		--source ./demo_data \
		--weights coursework/coursework_train/weights/best.pt \
		--save-txt --save-conf \
		--project coursework \
		--name demo \
		--imgsz $(IMG_SIZE) \
		--conf-thres $(CONF_THRES) \
		--device $(DEVICE)

venv:
	@if [ ! -f "$(VENV)/bin/activate" ]; then \
		echo "=== Creating venv ==="; \
		python3 -m venv $(VENV); \
	fi
	@echo "=== Activating venv ==="
	$(PIP) install --upgrade pip -q

data: venv
	@echo "=== Preparing dataset ==="
	mkdir -p ./data
	tar -xzf rts_dataset.tar.gz
	mv ./archive/* ./data/
	rm -rf ./archive

repos:
	@echo "=== Cloning repos ==="
	@[ -d JSON2YOLO ] || git clone https://github.com/ultralytics/JSON2YOLO
	@[ -d yolov5 ]    || git clone https://github.com/ultralytics/yolov5
	$(PIP) install -r yolov5/requirements.txt

deps: venv
	@echo "=== Installing dependencies ==="
	$(PIP) install -r requirements.txt

convert: deps repos data
	@echo "=== Converting COCO to YOLO ==="
	@echo "Patching JSON2YOLO..."
	$(PYTHON) ./src/patch_json2yolo.py
	@echo "Converting..."
	$(PYTHON) ./src/convert_coco_to_yolo.py

yaml: convert
	@echo "=== Creating YAML config ==="
	$(PYTHON) ./src/create_yaml.py

clean:
	@echo "=== CLEAN ALL ==="
	rm -rf ./JSON2YOLO ./yolov5
	rm -rf ./test_annotation ./train_annotation
	rm -rf ./trafic_signs.yaml
	rm -rf $(VENV)

clean_data:
	@echo "=== CLEAN DATA FOR TRAIN PIPELINE ==="
	rm -rf ./data

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
