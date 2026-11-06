SUBSET ?=

.PHONY: train up down test lint

train:
	SUBSET=$(SUBSET) python training/train.py

up:
	k3d cluster create --config cluster/k3d.yaml
	kubectl apply -f cluster/seaweedfs.yaml
	kubectl wait --for=condition=available deployment/seaweedfs -n data --timeout=120s
	python upload_s3.py
	docker build -t command-parser-service:latest .
	k3d image import command-parser-service:latest -c cloud-assgn
	kubectl apply -f k8s/deployment.yaml
	kubectl wait --for=condition=available deployment/command-parser -n pipeline --timeout=180s

down:
	k3d cluster delete cloud-assgn

test:
	SKIP_MODEL_LOAD=1 pytest tests/ -v

lint:
	ruff check service/ worker/ training/ tests/
