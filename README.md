# Command Parser Service — Cloud Computing Assignment

- **Student Name:** Abhinav Kumar Jha
- **Roll Number:** 230101044
- **GitHub Repository:** [https://github.com/Abhinav4518/Cloud-Computing-Assignment](https://github.com/Abhinav4518/Cloud-Computing-Assignment)
- **GitHub Release (Model Assets & Checksums):** [https://github.com/Abhinav4518/Cloud-Computing-Assignment/releases/tag/v1.0.0](https://github.com/Abhinav4518/Cloud-Computing-Assignment/releases/tag/v1.0.0)
- **Submission Commit Tag:** final

---

## 1. Prerequisites
- Docker & k3d (v5+)
- kubectl
- Python 3.11+ (pip install -r requirements.txt)

---

## 2. Build & Deployment Instructions

Automated cluster bring-up, S3 model upload, image build, and deployment:

    make up

Manual step-by-step deployment:

    k3d cluster create --config cluster/k3d.yaml
    kubectl apply -f cluster/seaweedfs.yaml
    kubectl wait --for=condition=available deployment/seaweedfs -n data --timeout=120s
    python upload_s3.py
    docker build -t command-parser-service:latest .
    k3d image import command-parser-service:latest -c cloud-assgn
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    kubectl wait --for=condition=available deployment/command-parser -n pipeline --timeout=180s

---

## 3. Testing, Evaluation & Verification Instructions

Code linting and unit tests:

    make lint
    make test

Model evaluation and 95% quality gate check:

    python tools/score.py results/student_v1_preds.jsonl data/validation.jsonl --out results/student_v1_validation.json
    python tools/score.py results/student_v1_preds.jsonl data/validation.jsonl --min-validity 95

Load testing with Locust:

    locust -f tools/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000

Network NACL simulation:

    python tools/nacl_sim.py network/nacl.json <packets_file.jsonl>

---

## 4. Shutdown & Teardown Instructions

Tear down the k3d cluster and free container resources:

    make down
