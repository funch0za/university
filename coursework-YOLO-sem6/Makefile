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

install_yolo: py_venv
	@[ -d yolov5 ] || git clone https://github.com/ultralytics/yolov5
	$(PIP) install -r yolov5/requirements.txt

train: py_pkgs install_yolo
	$(PYTHON) yolov5/train.py \
		--img 640 \
		--batch 8 \
		--epochs 100 \
		--patience 30 \
		--data trafic_signs_and_objects.yaml \
		--weights yolov5s.pt \
		--project coursework_signs_objects \
		--name signs_and_objects \
		--device mps \
		--workers 1 \
		--hyp hyp.yaml \

test_train_1_epoch: py_pkgs install_yolo
	$(PYTHON) yolov5/train.py \
		--img 640 \
		--batch 8 \
		--epochs 1 \
		--data trafic_signs_and_objects.yaml \
		--weights yolov5s.pt \
		--project coursework_signs_objects \
		--name test_1epoch \
		--device mps \
		--workers 1 \
		--hyp hyp.yaml \