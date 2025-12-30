#!/bin/bash
# ============================================================================
# Configuration file for ViPE Inference Pipeline
# ============================================================================
# Modify the values below to customize the pipeline behavior.

# Paths
WORKING_DIR="data_preprocess"  # Output directory
DATA_DIR="data_preprocess/example"  # Input data directory (read-only)

# Frame range
START_FRAME=0
# END_FRAME=$((START_FRAME + 49 - 1))  # Auto-calculated if not set
# Or set directly: END_FRAME=925

# Rendering
POINT_SIZE="5.0"

# Multiprocessing
BATCH_SIZE=3  # Number of parallel processes (recommended: 6-8)

# Advanced settings (usually no need to modify)
UUID_MAPPING_FILE="${WORKING_DIR}/take_name_to_uuid_mapping.json"

# Auto-calculate END_FRAME if not set
if [[ -z "${END_FRAME:-}" ]]; then
    END_FRAME=$((START_FRAME + 49 - 1))
fi

# Validate directories
if [[ ! -d "$WORKING_DIR" ]]; then
    echo "⚠️  WARNING: WORKING_DIR does not exist: $WORKING_DIR" >&2
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "⚠️  WARNING: DATA_DIR does not exist: $DATA_DIR" >&2
fi
