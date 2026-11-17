# Usage from the repository root: source scripts/env.sh
export RSN_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$RSN_HOME/scripts/env.local.sh" ]; then
    source "$RSN_HOME/scripts/env.local.sh"
fi
export RSN_ENV_DIR="${RSN_ENV_DIR:-$RSN_HOME/.venv}"
export RSN_RUNTIME_DIR="${RSN_RUNTIME_DIR:-$RSN_HOME/.runtime}"
export PYTHONPATH="$RSN_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$RSN_ENV_DIR/bin:$PATH"
export TMPDIR="${TMPDIR:-$RSN_RUNTIME_DIR/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RSN_RUNTIME_DIR/matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
