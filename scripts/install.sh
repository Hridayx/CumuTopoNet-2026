#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_ENV="${CUMUTOPO_ENV:-$TASK_ROOT/.venv}"
TASK_BASE_PYTHON="${CUMUTOPO_BASE_PYTHON:-python3.11}"
"$TASK_BASE_PYTHON" -m venv "$TASK_ENV"
"$TASK_ENV/bin/python" -m pip install --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$TASK_ENV/bin/python" -m pip install -r requirements-core.lock
if [[ "$(uname -s)" == Darwin ]]; then
  "$TASK_ENV/bin/python" -m pip install 'torch==2.8.0'
else
  # Select a PyTorch wheel compatible with the local GPU driver before experiments.
  TASK_TORCH_INDEX="${CUMUTOPO_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
  "$TASK_ENV/bin/python" -m pip install 'torch==2.8.0' --index-url "$TASK_TORCH_INDEX"
fi
"$TASK_ENV/bin/python" -m pip install --no-deps -e '.[test]'
"$TASK_ENV/bin/python" -m pip check
"$TASK_ENV/bin/python" -m pip freeze > "$TASK_ENV/installed-versions.txt"
echo "Installed. Activate $TASK_ENV and run cumutopo --help."
