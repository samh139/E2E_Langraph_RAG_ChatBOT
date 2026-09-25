#project_root/scripts/run_clustering.py

from app.clustering.cluster_job import run_clustering_job

if __name__ == "__main__":
    run_clustering_job(dry_run=False)
