"""Streamlit server memory auditor on Linux."""








import os
import sys
import time
import json
import subprocess
import threading
import psutil


sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None


def get_process_tree_memory(pid: int) -> dict:

    try:
        parent = psutil.Process(pid)
        procs = [parent] + parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"rss_mb": 0.0, "peak_wset_mb": 0.0}

    total_rss = 0
    total_peak = 0
    for p in procs:
        try:
            m = p.memory_info()
            total_rss += m.rss
            total_peak += getattr(m, "peak_wset", m.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return {
        "rss_mb": round(total_rss / (1024 * 1024), 2),
        "peak_wset_mb": round(total_peak / (1024 * 1024), 2),
    }


class MemoryMonitor:
    def __init__(self, pid: int, sample_interval: float = 0.2):
        self.pid = pid
        self.sample_interval = sample_interval
        self.running = False
        self.peak_rss_mb = 0.0
        self.peak_wset_mb = 0.0
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        while self.running:
            mem = get_process_tree_memory(self.pid)
            if mem["rss_mb"] > self.peak_rss_mb:
                self.peak_rss_mb = mem["rss_mb"]
            if mem["peak_wset_mb"] > self.peak_wset_mb:
                self.peak_wset_mb = mem["peak_wset_mb"]
            time.sleep(self.sample_interval)

    def stop(self) -> dict:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "peak_wset_mb": self.peak_wset_mb,
        }


def run_session_with_selenium(port: int, session_id: int):

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,1024")

    print(f"[Session {session_id}] Launching Chrome...", flush=True)
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(f"http://localhost:{port}")

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "button"))
        )
        print(f"[Session {session_id}] App loaded. Clicking presets...", flush=True)

        presets = ["Reach Left", "Reach Right", "Reach Up"]
        for p in presets:
            try:

                buttons = driver.find_elements(By.XPATH, f"//button[contains(., '{p}')]")
                if buttons:
                    buttons[0].click()
                    print(f"[Session {session_id}] Clicked '{p}', waiting for rollout...", flush=True)
                    time.sleep(6)
                else:
                    print(f"[Session {session_id}] Button for '{p}' not found", flush=True)
            except Exception as e:
                print(f"[Session {session_id}] Error clicking '{p}': {e}", flush=True)

        print(f"[Session {session_id}] Finished 3 presets successfully.", flush=True)
    finally:
        driver.quit()


def run_session_with_apptest(session_id: int):

    from streamlit.testing.v1 import AppTest
    print(f"[AppTest Session {session_id}] Running AppTest session...", flush=True)
    at = AppTest.from_file("app/app.py", default_timeout=45)
    at.run()

    for b in at.button:
        if any(kw in b.label for kw in ["Reach Left", "Reach Right", "Reach Up"]):
            print(f"[AppTest Session {session_id}] Clicking {b.label}", flush=True)
            b.click().run()
            time.sleep(1)


def main():
    port = 8501
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
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

    print(f"Starting Streamlit server on port {port} (MUJOCO_GL=osmesa)...", flush=True)
    server_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:

        import urllib.request
        health_url = f"http://localhost:{port}/_stcore/health"
        server_ready = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                time.sleep(0.5)

        if not server_ready:
            raise RuntimeError("Streamlit server failed to start within 20s")
        print("Streamlit server is ready.", flush=True)


        time.sleep(2)
        base_mem = get_process_tree_memory(server_proc.pid)
        print(f"Server Baseline Memory: {base_mem['rss_mb']} MB RSS", flush=True)


        has_selenium = False
        try:
            import selenium
            has_selenium = True
        except ImportError:
            pass




        print("\n--- MEASURING 1 SESSION (3 Presets + Whisper Loaded) ---", flush=True)
        monitor1 = MemoryMonitor(server_proc.pid)
        monitor1.start()

        if has_selenium:
            try:
                run_session_with_selenium(port, session_id=1)
            except Exception as e:
                print(f"Selenium failed: {e}. Falling back to AppTest...", flush=True)
                run_session_with_apptest(session_id=1)
        else:
            run_session_with_apptest(session_id=1)

        mem1 = monitor1.stop()
        print(f"1 Session Peak Memory: {mem1['peak_rss_mb']} MB RSS | {mem1['peak_wset_mb']} MB Working Set", flush=True)




        print("\n--- MEASURING 3 CONCURRENT SESSIONS ---", flush=True)
        monitor3 = MemoryMonitor(server_proc.pid)
        monitor3.start()

        threads = []
        for i in range(1, 4):
            if has_selenium:
                t = threading.Thread(target=run_session_with_selenium, args=(port, i))
            else:
                t = threading.Thread(target=run_session_with_apptest, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        mem3 = monitor3.stop()
        print(f"3 Concurrent Sessions Peak Memory: {mem3['peak_rss_mb']} MB RSS | {mem3['peak_wset_mb']} MB Working Set", flush=True)


        res = {
            "platform": sys.platform,
            "os": "Linux Ubuntu (CI Runner)",
            "gl_backend": "osmesa",
            "server_baseline_rss_mb": base_mem["rss_mb"],
            "single_session": {
                "condition": "1 session running 3 presets sequentially (Reach Left, Right, Up) with faster-whisper int8 loaded",
                "peak_rss_mb": mem1["peak_rss_mb"],
                "peak_working_set_mb": mem1["peak_wset_mb"],
            },
            "three_concurrent_sessions": {
                "condition": "3 concurrent sessions simultaneously running 3 presets with faster-whisper int8 loaded",
                "peak_rss_mb": mem3["peak_rss_mb"],
                "peak_working_set_mb": mem3["peak_wset_mb"],
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        os.makedirs("results", exist_ok=True)
        out_path = "results/streamlit_memory_v2.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        print("\n=== FINAL MEMORY RESULTS (results/streamlit_memory_v2.json) ===")
        print(json.dumps(res, indent=2), flush=True)
        return res

    finally:
        print("Stopping Streamlit server...", flush=True)
        try:
            parent = psutil.Process(server_proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except Exception:
            pass
        server_proc.kill()


if __name__ == "__main__":
    main()
