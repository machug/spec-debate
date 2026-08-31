#!/usr/bin/env bash
# Print the path to a Python interpreter that can import litellm.
#
# Everything except that one path goes to stderr, so callers can do:
#   PY=$(bash bootstrap.sh) && "$PY" debate.py providers
#
# The venv lives in a cache directory, NOT in the plugin directory. Plugin
# installs are version-keyed (.../spec-debate/1.11.0/), so a venv stored beside
# the code would be rebuilt from scratch on every plugin update.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requirements="$(cd "$script_dir/../../.." && pwd)/requirements.txt"
venv="${SPEC_DEBATE_VENV:-${XDG_CACHE_HOME:-$HOME/.cache}/spec-debate/venv}"
venv_python="$venv/bin/python"

# Note: litellm has no __version__ attribute. Import it and read the installed
# distribution version instead, or a working install looks like a failure.
usable() {
    [ -x "$1" ] && "$1" -c 'import litellm, importlib.metadata as m; m.version("litellm")' >/dev/null 2>&1
}

if usable "$venv_python"; then
    echo "$venv_python"
    exit 0
fi

if command -v python3 >/dev/null 2>&1 && usable "$(command -v python3)"; then
    command -v python3
    exit 0
fi

if [ ! -f "$requirements" ]; then
    echo "bootstrap: requirements.txt not found at $requirements" >&2
    exit 1
fi

echo "spec-debate: installing dependencies into $venv (first run only)..." >&2
mkdir -p "$(dirname "$venv")"

# --clear rebuilds a venv whose base interpreter was removed, which happens
# whenever a version manager prunes the Python the venv was created from.
if command -v uv >/dev/null 2>&1; then
    uv venv --clear "$venv" >&2
    VIRTUAL_ENV="$venv" uv pip install --quiet --requirement "$requirements" >&2
else
    python3 -m venv --clear "$venv" >&2
    "$venv_python" -m pip install --quiet --upgrade pip >&2
    "$venv_python" -m pip install --quiet --requirement "$requirements" >&2
fi

if ! usable "$venv_python"; then
    echo "bootstrap: dependency install finished but litellm still will not import" >&2
    exit 1
fi

echo "$venv_python"
