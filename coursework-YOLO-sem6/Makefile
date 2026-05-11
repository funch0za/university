VENV = ./venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

py_venv:
	@if [ ! -f "$(VENV)/bin/activate" ]; then \
		echo "=== Creating venv ==="; \
		python3 -m venv $(VENV); \
	fi
	@echo "=== Activating venv ==="
	$(PIP) install --upgrade pip -q

py_pkgs: py_venv
	@echo "=== Installing python packages ==="
	$(PIP) install -r requirements.txt

install_yolov5: py_venv
	@echo "=== Installing YOLOv5 ==="
	@[ -d yolov5 ]    || git clone https://github.com/ultralytics/yolov5
	$(PIP) install -r yolov5/requirements.txt 

install_JSON2YOLO:
	@echo "=== Installing JSON2YOLO ==="
	@[ -d JSON2YOLO ] || git clone https://github.com/ultralytics/JSON2YOLO

convert:
	@echo "=== Converting COCO to YOLO ==="
	@echo "Patching JSON2YOLO..."
	$(PYTHON) ./src/patch_json2yolo.py
	@echo "Converting..."
	$(PYTHON) ./src/convert_coco_to_yolo.py

full_pipeline: py_pkgs install_yolov5 install_JSON2YOLO
