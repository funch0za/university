.PHONY: all clean demo data venv repos convert yaml train detect help

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

all: train

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
		--workers $(WORKERS) \
		--hyp hyp.finetune.yaml

demo: detect
	$(PYTHON) gui.py ./coursework/demo/

detect: python_packages
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
	@if [ ! -d ./data ] || [ -z "$$(ls -A ./data 2>/dev/null)" ]; then \
		mkdir -p ./data; \
		tar -xzf rts_dataset.tar.gz; \
		mv ./archive/* ./data/; \
		rm -rf ./archive; \
	else \
		echo "Dataset already exists, skipping..."; \
	fi

python_packages: venv
	@echo "=== Installing python packages ==="
	$(PIP) install -r requirements.txt

repos: python_packages
	@echo "=== Cloning repos ==="
	@[ -d JSON2YOLO ] || git clone https://github.com/ultralytics/JSON2YOLO
	@[ -d yolov5 ]    || git clone https://github.com/ultralytics/yolov5
	$(PIP) install -r yolov5/requirements.txt 

convert: python_packages repos data
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
	rm -rf ./coursework
	rm -rf yolov5s.pt

clean_data:
	@echo "=== CLEAN DATA FOR TRAIN PIPELINE ==="
	rm -rf ./data

clean_demo:
	@echo "=== CLEAN RESULT FROM DEMO ==="
	rm -rf ./coursework/demo*

help:
	@echo ""
	@echo "\033[1mUsage:\033[0m make [target] [VARIABLE=value ...]"
	@echo ""
	@echo "\033[1mTargets:\033[0m"
	@echo "  \033[36mall\033[0m              Train model"
	@echo "  \033[36mtrain\033[0m            Train model"
	@echo "  \033[36mdemo\033[0m             Run GUI demo"
	@echo "  \033[36mdetect\033[0m           Run YOLO detect on ./demo_data"
	@echo "  \033[36mvenv\033[0m             Create Python virtual environment"
	@echo "  \033[36mpython_packages\033[0m  Install Python dependencies"
	@echo "  \033[36mdata\033[0m             Extract dataset"
	@echo "  \033[36mrepos\033[0m            Clone YOLOv5 and JSON2YOLO"
	@echo "  \033[36mconvert\033[0m          Convert COCO to YOLO format"
	@echo "  \033[36myaml\033[0m             Generate YOLO config"
	@echo "  \033[36mclean\033[0m            Remove all without dataset"
	@echo "  \033[36mclean_data\033[0m       Remove dataset only"
	@echo "  \033[36mclean_demo\033[0m       Remove demo results"
	@echo "  \033[36mhelp\033[0m             Show this message"
	@echo ""
	@echo "\033[1mVariables:\033[0m"
	@echo "  EPOCHS      (default: 3)"
	@echo "  BATCH       (default: 16)"
	@echo "  IMG_SIZE    (default: 640)"
	@echo "  WEIGHTS     (default: yolov5s.pt)"
	@echo "  DEVICE      (default: mps)"
	@echo "  WORKERS     (default: 2)"
	@echo "  CONF_THRES  (default: 0.25)"
