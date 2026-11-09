import os
import sys
import time
import json
import subprocess
import psutil
from selenium import webdriver
from selenium.webdriver.edge.options import Options

def measure_streamlit_server_memory():
    port = 8504
    env = os.environ.copy()
    env["STREAMLIT_AUTORUN_BENCHMARK"] = "1"
    env["MUJOCO_GL"] = "glfw"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    print(f"Starting Streamlit server on port {port}...")
    server_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # Wait for server to become responsive
        import urllib.request
        health_url = f"http://localhost:{port}/_stcore/health"
        server_ready = False
        for _ in range(30):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                time.sleep(0.5)

        if not server_ready:
            raise RuntimeError("Streamlit server failed to start within 15 seconds")

        print("Streamlit server is ready. Launching headless browser to trigger session...")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        driver = webdriver.Edge(options=options)

        try:
            driver.get(f"http://localhost:{port}")
            print("Connected to Streamlit page. Allowing demo rollout and Whisper to execute...")
            
            # Monitor memory of the Streamlit process tree during execution
            parent = psutil.Process(server_proc.pid)
            peak_wset_bytes = 0
            peak_rss_bytes = 0

            # Monitor for 20 seconds or until peak stabilizes
            for _ in range(40):
                time.sleep(0.5)
                procs = [parent] + parent.children(recursive=True)
                total_rss = 0
                total_peak = 0
                for p in procs:
                    try:
                        m = p.memory_info()
                        total_rss += m.rss
                        total_peak += getattr(m, "peak_wset", m.rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if total_rss > peak_rss_bytes:
                    peak_rss_bytes = total_rss
                if total_peak > peak_wset_bytes:
                    peak_wset_bytes = total_peak

            print("Demo execution finished.")
        finally:
            driver.quit()

        final_rss_mb = round(peak_rss_bytes / (1024 * 1024), 2)
        final_peak_wset_mb = round(peak_wset_bytes / (1024 * 1024), 2)

        print(f"Streamlit Server Process Peak RSS: {final_rss_mb} MB")
        print(f"Streamlit Server Process Peak Working Set: {final_peak_wset_mb} MB")

        res = {
            "measured_process": "streamlit_server_process_tree",
            "condition": "full demo run (faster-whisper int8 model loaded + MuJoCo rollout rendered + video encoded)",
            "peak_rss_mb": final_rss_mb,
            "peak_working_set_mb": final_peak_wset_mb,
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        }

        os.makedirs("results", exist_ok=True)
        with open("results/streamlit_memory.json", "w") as f:
            json.dump(res, f, indent=2)

        print("Saved memory metrics to results/streamlit_memory.json")
        return res

    finally:
        print("Terminating Streamlit server process...")
        try:
            parent = psutil.Process(server_proc.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()


if __name__ == "__main__":
    measure_streamlit_server_memory()
