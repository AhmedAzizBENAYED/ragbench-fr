"""Constants shared by tests. Paths are absolute: the working directory is a temporary one
(see tests/conftest.py)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example.yaml"

# Commit SHA of AgentPublic/piaf pinned by the repository configs.
PIAF_REVISION = "bda8c063bc7297180796cd835d1974c0bc71c521"

# Commit SHA of intfloat/multilingual-e5-base pinned by the repository configs.
E5_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"

# Golden fingerprint of configs/example.yaml. It changes whenever a result-affecting value of the
# example (or a default it relies on) changes. If the change is intended, update this value; it
# means runs made with the previous config are no longer comparable or reusable from cache.
EXAMPLE_FINGERPRINT = "987f020a5a8f2001a7d682ea25612e214817a028599dbaa31695e4f471fddea1"

# Mini PIAF file with deliberate traps (duplicated context, missing answer, empty question...).
PIAF_MINI = REPO_ROOT / "tests" / "fixtures" / "piaf_mini.json"
