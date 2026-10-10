"""Constants shared by tests. Paths are absolute: the working directory is a temporary one
(see tests/conftest.py)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example.yaml"

# Commit SHA of AgentPublic/piaf pinned by the repository configs.
PIAF_REVISION = "bda8c063bc7297180796cd835d1974c0bc71c521"

# Golden fingerprint of configs/example.yaml. It changes whenever a result-affecting value of the
# example (or a default it relies on) changes. If the change is intended, update this value; it
# means runs made with the previous config are no longer comparable or reusable from cache.
EXAMPLE_FINGERPRINT = "ef0d7e55e98b147b09fbba1ca295f21962f823ce0d133b9cdea16015d485500d"

# Mini PIAF file with deliberate traps (duplicated context, missing answer, empty question...).
PIAF_MINI = REPO_ROOT / "tests" / "fixtures" / "piaf_mini.json"
