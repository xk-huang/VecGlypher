#/bin/bash
set -e

storage_cli --prod-use-cython-client getr --threads 20 --jobs 10 workspace/svg_glyph_llm/data/processed data/processed

storage_cli --prod-use-cython-client getr --threads 20 --jobs 10 workspace/svg_glyph_llm/data/processed_envato data/processed_envato
