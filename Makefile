.PHONY: help install data train cv evaluate pretrained-eval predict test lint dashboard clean

help:
	@echo "Targets:"
	@echo "  install         install the package (editable) + dev extras"
	@echo "  data            reconstruct the dataset from the rendered PNGs"
	@echo "  train           train GlyphCNN from scratch and write reports/"
	@echo "  cv              multi-seed cross-validation -> reports/scratch_cv.json"
	@echo "  evaluate        evaluate the freshly trained checkpoint on the test split"
	@echo "  pretrained-eval evaluate the historical pretrained model on all glyphs"
	@echo "  predict         write predicted labels for the processed dataset"
	@echo "  test            run the pytest suite"
	@echo "  lint            run ruff"
	@echo "  dashboard       regenerate the static results dashboard (site/index.html)"

install:
	pip install -e ".[dev]"

data:
	python scripts/reconstruct_dataset.py --src data/raw_png/images --out data/processed

train:
	python -m glyphcnn train

cv:
	python scripts/cross_validate.py --seeds 5

evaluate:
	python -m glyphcnn evaluate --checkpoint models/glyphcnn.pth --split test

pretrained-eval:
	python -m glyphcnn evaluate --checkpoint models/pretrained_model.pth --split all

predict:
	python -m glyphcnn predict --checkpoint models/glyphcnn.pth --input data/processed/dataset.npz --output predlabels.txt

test:
	pytest -q

lint:
	ruff check src tests scripts

dashboard:
	python scripts/build_dashboard.py

clean:
	rm -rf reports/*.json reports/figures/*.png predlabels.txt
