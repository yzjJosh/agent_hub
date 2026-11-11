#!/bin/bash
# Runs pytest inside the Agent Hub container.
#
# Usage: scripts/run_tests.sh [pytest args...]
#
# Parallel by default with -n 4 --dist worksteal. Go serial with -n 0, pick
# another size with -n 8, or otherwise pass any of -n, --numprocesses, --dist
# to take over the parallel settings entirely. Repo-root paths are mapped into
# the container.
set -e

CONTAINER="agent_hub"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo "The ${CONTAINER} container does not exist. Create it with 'docker compose up -d ${CONTAINER}' from the compose project that defines it." >&2
        exit 1
    fi
    echo "Starting ${CONTAINER} container..."
    docker start "${CONTAINER}" >/dev/null
    # Give the container time to come up.
    sleep 5
fi

# Map the host repo root to the folder it is mounted at in the container.
# A worktree under .worktrees is mounted at /worktrees/<name>, and the main
# checkout is mounted at /app.
REPO_ROOT="$(git rev-parse --show-toplevel)"
if [ "$(git rev-parse --absolute-git-dir)" != "$(git rev-parse --path-format=absolute --git-common-dir)" ]; then
    CONTAINER_ROOT="/worktrees/$(basename "$REPO_ROOT")"
else
    CONTAINER_ROOT="/app"
fi

# Paths under the repo root are turned into paths inside the container.
args=()
for arg in "$@"; do
    if [[ "$arg" == "$REPO_ROOT"/* ]]; then
        args+=("${arg#"$REPO_ROOT"/}")
    else
        args+=("$arg")
    fi
done

# Tests run in parallel with -n 4 --dist worksteal by default.
# A -n, --numprocesses, or --dist argument means the caller picks the parallel settings.
# Runs are serial when the image has no pytest-xdist.
xdist=("-n" "4" "--dist" "worksteal")
for arg in "$@"; do
    case "$arg" in
        -n* | --numprocesses* | --dist*)
            xdist=()
            break
            ;;
    esac
done

if [ ${#xdist[@]} -gt 0 ] && ! docker exec "${CONTAINER}" python -c "import xdist" >/dev/null 2>&1; then
    echo "pytest-xdist is not installed in ${CONTAINER}, running serial" >&2
    xdist=()
fi

docker exec -w "$CONTAINER_ROOT" ${CONTAINER} python -m pytest ${xdist[@]+"${xdist[@]}"} ${args[@]+"${args[@]}"}
