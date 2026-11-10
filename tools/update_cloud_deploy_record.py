"""Update results/cloud_deploy.json commit metadata from the current git commit."""

import json
import os
import subprocess
import sys


def update_cloud_deploy_record():
    cloud_deploy_path = "results/cloud_deploy.json"
    if not os.path.exists(cloud_deploy_path):
        print(f"Error: {cloud_deploy_path} not found.")
        sys.exit(1)

    with open(cloud_deploy_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get current git commit sha and commit timestamp
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        last_commit_at = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], text=True
        ).strip()
    except Exception as e:
        print(f"Error reading git metadata: {e}")
        sys.exit(1)

    data["commit_sha"] = commit_sha
    data["last_commit_at"] = last_commit_at

    # Explicitly ensure cloud runtime performance measurements remain null until measured on cloud host
    data["first_request_latency_ms"] = None
    data["measured_rollout_fps"] = None
    data["measured_memory_rss_mb"] = None

    with open(cloud_deploy_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {cloud_deploy_path}:")
    print(f"  commit_sha:     {commit_sha}")
    print(f"  last_commit_at: {last_commit_at}")
    return data


if __name__ == "__main__":
    update_cloud_deploy_record()

