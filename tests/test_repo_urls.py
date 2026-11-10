"""Test verifying that all GitHub repository references point to Saksham-19-cyber/voice-robot-arm-edge."""

import os
import re
import subprocess

EXPECTED_REPO = "saksham-19-cyber/voice-robot-arm-edge"
BINARY_EXTENSIONS = {".png", ".gif", ".zip", ".onnx", ".pyc", ".wav", ".mp4", ".pt", ".pth", ".data"}


def test_no_wrong_github_repo_urls():
    """Fail if any github.com/<owner>/<repo> differs from Saksham-19-cyber/voice-robot-arm-edge or allowlist."""
    allowlist = set()
    allowlist_path = os.path.join("tests", "github_url_allowlist.txt")
    if os.path.isfile(allowlist_path):
        with open(allowlist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    token = line.split("#")[0].strip().lower()
                    if token.endswith(".git"):
                        token = token[:-4]
                    allowlist.add(token)

    try:
        files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    except Exception:
        files = []
        for root, dirs, filenames in os.walk("."):
            if ".git" in dirs:
                dirs.remove(".git")
            if "venv" in dirs:
                dirs.remove("venv")
            for fn in filenames:
                files.append(os.path.join(root, fn))

    pattern = r'github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)'
    violations = []

    for path in files:
        if any(path.lower().endswith(ext) for ext in BINARY_EXTENSIONS):
            continue
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    for m in re.finditer(pattern, line):
                        owner = m.group(1).lower()
                        repo = m.group(2).lower()
                        if repo.endswith(".git"):
                            repo = repo[:-4]
                        slug = f"{owner}/{repo}"
                        if slug != EXPECTED_REPO and slug not in allowlist:
                            violations.append((path, line_no, m.group(0), slug))
        except Exception:
            pass

    assert not violations, (
        f"Found {len(violations)} disallowed GitHub repository URL(s):\n"
        + "\n".join([f"  {p}:{l} -> {raw} (slug: {slug})" for p, l, raw, slug in violations])
    )

