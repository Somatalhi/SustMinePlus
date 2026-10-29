"""Write jobs.yaml: one Kubernetes Job per (algo, seed, budget).
Usage:  python make_jobs.py --budget 500 --seeds 30 --workers 16
Then:   kubectl apply -f jobs.yaml
Each job copies the code from the volume (/data/sustmine-search, refreshed by
update-job.yaml) into /work, links results/ back to the volume, and runs one
run_exp1_mip.py call. bench_phases.pkl travels with the repository.
"""
import argparse

TEMPLATE = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: sm-{algo}-s{seed}-b{budget}
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 172800
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: run
        image: python:3.12-slim
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            mkdir -p /work && cp -r /data/sustmine-search/. /work/ && rm -rf /work/results
            mkdir -p /data/sustmine-search/results && ln -s /data/sustmine-search/results /work/results
            cd /work
            pip install -q -r requirements.txt
            python -c "import pandas as pd; bp=pd.read_pickle('bench_phases.pkl'); print(len(bp), 'bench-phases,', bp.pb.max(), 'pushbacks')"
                        python -u run_exp1_mip.py --algo {algo} --seed {seed} --budget {budget} 
            --workers {workers} \\
                   --time_limit 60 --retry_time_limit 300 --retry_gap 0.05
        resources:
          requests: {{cpu: "{workers}", memory: "{mem}Gi"}}
          limits:   {{cpu: "{workers}", memory: "{mem}Gi"}}
        volumeMounts:
        - {{name: data, mountPath: /data}}
      volumes:
      - name: data
        persistentVolumeClaim: {{claimName: sustmine-data}}
"""

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, required=True)
ap.add_argument("--seeds", type=int, default=30)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--algos", nargs="+", default=["nsga2", "nsga3", "lhs"])
a = ap.parse_args()

mem = max(4, a.workers)
docs = [TEMPLATE.format(algo=al, seed=s, budget=a.budget, workers=a.workers, mem=mem)
        for al in a.algos for s in range(a.seeds)]
open("jobs.yaml", "w").write("".join(docs))
print(f"jobs.yaml: {len(docs)} jobs ({a.algos} x {a.seeds} seeds, budget {a.budget}, {a.workers} cpu each)")
