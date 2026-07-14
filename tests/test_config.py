from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_label_mapping_is_contiguous():
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    labels = config["model"]["label_names"]
    assert len(labels) == config["model"]["num_labels"]
    assert len(set(labels)) == len(labels)


def test_canonical_dockerfile_is_used():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "dockerfile: Dockerfile" in compose
    assert not (ROOT / "deployment" / "Dockerfile").exists()
