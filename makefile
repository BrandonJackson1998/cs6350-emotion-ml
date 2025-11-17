VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Targets:"
	@echo "install                     - Install environment necessary to support this project."
	@echo "install-deb                 - Install OS packages necessary to support this project. Assumes apt/dpkg package management system."
	@echo "install-pip                 - Install Python pakcages necessary to suport this project."
	@echo "benchmark                   - Run the emotion recognition benchmark training."
	@echo "test-full-dataset           - Test the full dataset training implementation."
	@echo "evaluate-baseline           - Evaluate the baseline pre-trained model performance."
	@echo "adaptive-training           - Run adaptive training focusing on worst emotions."
	@echo "temporal-analysis           - Run temporal emotion analysis on video frames."
	@echo "temporal-analysis-custom    - Run temporal analysis with custom model."
	@echo "test-temporal               - Run tests for temporal emotion analysis system."
	@echo

$(VENV):
	python3.12 -m venv $(VENV)

install: install-deb install-pip

install-deb:
	@echo python3.12-venv is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python3.12-venv ffmpeg; do \
		dpkg -l | egrep '^ii *'$${package}' ' 2>&1 > /dev/null || sudo apt install $${package}; \
	done

install-pip: $(VENV)
	source $(VENV)/bin/activate; pip3 install --upgrade -r requirements.txt

install-mac: install-deb-mac install-pip
	
install-deb-mac:
	@echo python@3.12 is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python@3.12 ffmpeg; do \
		brew list --versions $${package} 2>&1 > /dev/null || brew install $${package}; \
	done

benchmark:
	source $(VENV)/bin/activate; python -m src.benchmark

test-full-dataset:
	source $(VENV)/bin/activate; python test_full_dataset.py

evaluate-baseline:
	source $(VENV)/bin/activate; python evaluate_baseline.py

adaptive-training:
	source $(VENV)/bin/activate; python adaptive_training.py --initial-model experiments/full_dataset_single_epoch_20251114_090055/epoch_1_model.pt --epochs 10

temporal-analysis:
	source $(VENV)/bin/activate; python src/temporal_emotion_analyzer.py compare/test/ --output compare/outputs/temporal_analysis

temporal-analysis-custom:
	@echo "Usage: make temporal-analysis-custom MODEL_PATH=/path/to/model.pt"
	@echo "Example: make temporal-analysis-custom MODEL_PATH=experiments/best_model.pt"
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH not specified"; \
		echo "Please provide a model path: make temporal-analysis-custom MODEL_PATH=/path/to/model.pt"; \
		exit 1; \
	fi
	source $(VENV)/bin/activate; python src/temporal_emotion_analyzer.py compare/test/ --model-path $(MODEL_PATH) --output compare/outputs/temporal_analysis_custom

test-temporal:
	source $(VENV)/bin/activate; python test_temporal_analysis.py

run: benchmark
