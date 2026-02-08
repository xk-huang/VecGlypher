#!/usr/bin/env bash
# set -euo pipefail
set -e

# $1 is the root dir of the eval results
# $2 is the storage_cli bucket to upload to
# $3 is the number of concurrent jobs (optional, default 10)

if [[ -z "$1" ]]; then
    ROOT="outputs/250818-oxford_5000-100_fonts-apply_word_sep"
else
    ROOT="$1"
fi
if [[ -z "$2" ]]; then
    MB="workspace/svg_glyph_llm"
else
    MB="$2"
fi
if [[ -z "$3" ]]; then
    MAX_JOBS=10
else
    MAX_JOBS="$3"
fi

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
}


log "Root dir: $ROOT"
log "Storage bucket: $MB"
log "Max concurrent jobs: $MAX_JOBS"

# Function to upload a single directory
upload_dir() {
    local dir="$1"
    if [[ "$dir" == *DONE_INFER ]]; then
        dir="$(dirname "$dir")"
    fi
    local mb="$2"
    local remote="$mb/$dir"


    if [[ ! -d "$dir" ]]; then
        log "Error: Non-existent dir: $dir" >&2
        return 1
    fi

    # if dry_run is set, just print the command instead of running it
    if [[ -n "${dry_run:-}" ]]; then
        log "[DRY RUN] would upload: $dir"
        return 0
    fi
    log "Uploading: $dir ..."

    # Create remote directory and upload with error handling
    if ! storage_cli --prod-use-cython-client mkdirs "$remote" 2>/dev/null; then
        log "Warning: Failed to create remote dir $remote" >&2
    fi

    if storage_cli --prod-use-cython-client putr "$dir" "$remote" --threads 20 --jobs 10 --overwrite 2>/dev/null; then
        log "Successfully uploaded: $dir"
    else
        log "Error: Failed to upload $dir" >&2
        return 1
    fi
}

# Export function so it's available to subprocesses
export -f upload_dir
export -f log
export MB
export dry_run

# Use xargs for efficient parallel processing
# This automatically manages job pools and starts new jobs as soon as others finish

log_file="$ROOT/upload.log"
log "Starting upload process: $ROOT, see log at $log_file"

find "$ROOT"  -type f -name "done_eval" -print0 \
| while IFS= read -r -d '' f; do
    # NOTE: .../exp_job/flog/done_eval -> .../exp_job/
    dir=$(dirname $(dirname "$f"))
    find "$dir" -mindepth 1 -maxdepth 1 -type d ! -name '*decoded*' -printf '%h/%f\0'
  done \
| xargs -0 -P "$MAX_JOBS" -I {} bash -c 'upload_dir "$@"' _ {} "$MB" | tee "$log_file"

log "All uploads completed, see log at $log_file"
