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

