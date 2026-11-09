"""Tests verifying all README asset links exist on disk."""
import os
import re

README_PATH = "README.md"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def test_readme_assets_exist():
    assert os.path.exists(README_PATH), f"{README_PATH} must exist"
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()


    image_pattern = r"!\[[^\]]*\]\((?!https?://)([^)]+)\)"
    matches = re.findall(image_pattern, content)


    image_targets = [
        m.split("?")[0].strip()
        for m in matches
        if any(m.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        or m.startswith("assets/")
        or m.startswith("results/")
    ]

    assert len(image_targets) > 0, "Expected to find at least one local image asset in README.md"

    missing = []
    for rel_path in image_targets:
        norm_path = os.path.normpath(rel_path)
        if not os.path.isfile(norm_path):
            missing.append(rel_path)

    assert not missing, f"The following referenced images in README.md are missing on disk: {missing}"
