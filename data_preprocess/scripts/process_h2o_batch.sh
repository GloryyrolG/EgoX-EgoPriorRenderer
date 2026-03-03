#!/bin/bash
# Batch process H2O sequences with different windowing strategies
# Usage: bash process_h2o_batch.sh h1 0 cam0 [strategy] [subject] [run_post]
#   strategy: single (default) | no_overlap | overlap

set -e

# h2o_to_egox.py expects h2o_root as dataset root and appends --subject internally.
# Override by exporting H2O_ROOT/TEXT_ROOT/OUTPUT_ROOT if needed.
H2O_ROOT="${H2O_ROOT:-/mnt/shared/dses/h2o}"
TEXT_ROOT="${TEXT_ROOT:-/mnt/shared/dses/egoworld/h2o/text}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared/dses/egox/h2o_batch}"
SCENE=${1:-"h1"}
SEQUENCE=${2:-"0"}
EXO_CAM=${3:-"cam0"}
STRATEGY=${4:-"single"}  # single | no_overlap | overlap
SUBJECT=${5:-"subject1"}
RUN_POST=${6:-${RUN_POST:-"1"}}  # 1: run ego2exo post pipeline, 0: conversion only
EGOX_ENV="${EGOX_ENV:-egox-egopriorrenderer}"
FLAT_SINGLE_OUTPUT="${FLAT_SINGLE_OUTPUT:-1}"  # 1: single strategy writes directly to OUTPUT_ROOT
FLAT_MULTI_OUTPUT="${FLAT_MULTI_OUTPUT:-1}"    # 1: no_overlap/overlap also write to OUTPUT_ROOT
RESUME_WINDOWS="${RESUME_WINDOWS:-1}"          # 1: skip windows that already have complete outputs
DEPTH_ARTIFACT_POLICY="${DEPTH_ARTIFACT_POLICY:-any}"  # any | ego | exo | both

# Optional upstream: extract H2O subset/full from tar before processing.
EXTRACT_FROM_TAR="${EXTRACT_FROM_TAR:-1}"      # 1: untar before processing, 0: skip untar stage
TAR_EXTRACT_MODE="${TAR_EXTRACT_MODE:-partial}"  # partial | full
TAR_SOURCE_DIR="${TAR_SOURCE_DIR:-$H2O_ROOT}"  # where *.tar.gz is stored

WINDOW_SIZE=49  # EgoX fixed window
TOTAL_FRAMES="${TOTAL_FRAMES:-}"  # auto-detect if empty

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

resolve_subject_tar_path() {
    local candidates
    candidates=$(find "$TAR_SOURCE_DIR" -maxdepth 1 -type f -name "${SUBJECT}*.tar.gz" | sort)
    if [[ -z "$candidates" ]]; then
        echo "❌ No tar.gz found for subject=${SUBJECT} under $TAR_SOURCE_DIR"
        return 1
    fi

    # Prefer latest lexicographically (usually *_vX_Y.tar.gz).
    echo "$candidates" | tail -n 1
}

has_required_h2o_subset() {
    local base="${H2O_ROOT}/${SUBJECT}/${SCENE}/${SEQUENCE}"
    local exo_base="${base}/${EXO_CAM}"
    local ego_base="${base}/cam4"

    [[ -f "${exo_base}/cam_intrinsics.txt" ]] &&
    [[ -f "${ego_base}/cam_intrinsics.txt" ]] &&
    [[ -d "${exo_base}/rgb" ]] &&
    [[ -d "${exo_base}/cam_pose" ]] &&
    [[ -d "${ego_base}/rgb" ]] &&
    [[ -d "${ego_base}/cam_pose" ]]
}

extract_h2o_from_tar() {
    if [[ "$EXTRACT_FROM_TAR" != "1" ]]; then
        return 0
    fi

    local tar_path
    tar_path=$(resolve_subject_tar_path) || return 1

    mkdir -p "$H2O_ROOT"

    if [[ "$TAR_EXTRACT_MODE" == "partial" ]]; then
        if has_required_h2o_subset; then
            echo "✅ [UNTAR] Required subset already exists, skip untar."
            return 0
        fi
    fi

    echo -e "\n[UNTAR] SUBJECT=${SUBJECT} MODE=${TAR_EXTRACT_MODE}"
    echo "[UNTAR] tar: $tar_path"
    echo "[UNTAR] dst: $H2O_ROOT"

    if [[ "$TAR_EXTRACT_MODE" == "full" ]]; then
        tar -xzf "$tar_path" -C "$H2O_ROOT" "${SUBJECT}/"
    elif [[ "$TAR_EXTRACT_MODE" == "partial" ]]; then
        local seq_base="${SUBJECT}/${SCENE}/${SEQUENCE}"
        local exo_base="${seq_base}/${EXO_CAM}"
        local ego_base="${seq_base}/cam4"
        tar -xzf "$tar_path" -C "$H2O_ROOT" \
            "${exo_base}/cam_intrinsics.txt" \
            "${exo_base}/rgb/" \
            "${exo_base}/cam_pose/" \
            "${ego_base}/cam_intrinsics.txt" \
            "${ego_base}/rgb/" \
            "${ego_base}/cam_pose/"
    else
        echo "❌ Unsupported TAR_EXTRACT_MODE=${TAR_EXTRACT_MODE}, expected partial|full"
        return 1
    fi
}

merge_meta_entry() {
    local src_meta="$1"
    local dst_meta="$2"
    python - "$src_meta" "$dst_meta" <<'PY'
import json, os, sys
src_path, dst_path = sys.argv[1], sys.argv[2]
src = json.load(open(src_path))
entry = src["test_datasets"][0]
if os.path.exists(dst_path):
    dst = json.load(open(dst_path))
else:
    dst = {"test_datasets": []}
arr = dst.get("test_datasets", [])
idx = None
for i, e in enumerate(arr):
    if e.get("take_name") == entry.get("take_name"):
        idx = i
        break
if idx is None:
    arr.append(entry)
else:
    arr[idx] = entry
dst["test_datasets"] = arr
with open(dst_path, "w") as f:
    json.dump(dst, f, indent=2)
PY
}

merge_meta_entry_locked() {
    local src_meta="$1"
    local dst_meta="$2"
    local lock_file="${OUTPUT_ROOT}/.meta_merge.lock"
    mkdir -p "$OUTPUT_ROOT"
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 200
            merge_meta_entry "$src_meta" "$dst_meta"
        ) 200>"$lock_file"
    else
        # Fallback without file lock (less safe under concurrent writers)
        merge_meta_entry "$src_meta" "$dst_meta"
    fi
}

run_post_pipeline() {
    local out_dir="$1"
    local meta_json="$2"
    if [[ ! -f "$meta_json" ]]; then
        echo "❌ Missing meta.json: $meta_json"
        return 1
    fi

    local take_name
    take_name=$(python - "$meta_json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))["test_datasets"][0]
print(m["take_name"])
PY
)

    local end_rel
    end_rel=$(python - "$meta_json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))["test_datasets"][0]
print(len(m["ego_extrinsics"]) - 1)
PY
)

    local ego_video="${out_dir}/videos/${take_name}/ego.mp4"
    local vipe_out="${out_dir}/vipe_results"
    local vipe_take="${vipe_out}/${take_name}"

    echo -e "\n[Post] ego2exo ViPE infer: $ego_video"
    conda run -n "$EGOX_ENV" python -m vipe.cli.main infer "$ego_video" \
        --output "$vipe_out" \
        --pipeline lyra \
        --start_frame 0 \
        --end_frame "$end_rel"

    echo -e "\n[Post] Render exo_Prior.mp4"
    conda run -n "$EGOX_ENV" python scripts/render_vipe_pointcloud.py \
        --input_dir "$vipe_take" \
        --artifact_name ego \
        --out_dir "${out_dir}/videos/${take_name}" \
        --out_dir_no_append \
        --meta_json_path "$meta_json" \
        --render_target exo \
        --start_frame 0 \
        --end_frame "$end_rel" \
        --point_size 5.0 \
        --use_mean_bg \
        --no_aria

    # Compatibility for downstream training code that still resolves ego_prior_path -> ego_Prior.mp4.
    # For ego2exo flow we render exo_Prior.mp4, so create a sibling alias.
    local exo_prior="${out_dir}/videos/${take_name}/exo_Prior.mp4"
    local ego_prior_alias="${out_dir}/videos/${take_name}/ego_Prior.mp4"
    if [[ -f "$exo_prior" ]]; then
        ln -sfn "$(basename "$exo_prior")" "$ego_prior_alias"
    fi

    echo -e "\n[Post] Convert depth zip -> npy"
    conda run -n "$EGOX_ENV" python scripts/convert_depth_zip_to_npy.py \
        --depth_path "${vipe_take}/depth" \
        --egox_depthmaps_path "${out_dir}/depth_maps"
}

count_depth_npy() {
    local depth_dir="$1"
    if [[ -d "$depth_dir" ]]; then
        find "$depth_dir" -maxdepth 1 -type f -name 'depth_*.npy' | wc -l
    else
        echo 0
    fi
}

is_post_complete_for_clip() {
    local out_dir="$1"
    local take_name="$2"
    local expected_frames="$3"

    local exo_prior="${out_dir}/videos/${take_name}/exo_Prior.mp4"
    local depth_take_dir="${out_dir}/depth_maps/${take_name}"
    local ego_depth_count
    local exo_depth_count
    ego_depth_count=$(count_depth_npy "${depth_take_dir}/ego")
    exo_depth_count=$(count_depth_npy "${depth_take_dir}/exo")

    if [[ ! -f "$exo_prior" ]]; then
        return 1
    fi

    case "$DEPTH_ARTIFACT_POLICY" in
        "ego")
            [[ "$ego_depth_count" -eq "$expected_frames" ]]
            ;;
        "exo")
            [[ "$exo_depth_count" -eq "$expected_frames" ]]
            ;;
        "both")
            [[ "$ego_depth_count" -eq "$expected_frames" && "$exo_depth_count" -eq "$expected_frames" ]]
            ;;
        "any"|*)
            [[ "$ego_depth_count" -eq "$expected_frames" || "$exo_depth_count" -eq "$expected_frames" ]]
            ;;
    esac
}

process_clip() {
    local out_dir="$1"
    local start="$2"
    local end="$3"
    local merged_meta="${OUTPUT_ROOT}/meta_all.json"
    local start_pad
    local end_pad
    start_pad=$(printf '%06d' "$start")
    end_pad=$(printf '%06d' "$end")
    local expected_frames=$((end - start + 1))
    local take_name="${SUBJECT}_${SCENE}_${SEQUENCE}_${EXO_CAM}_${start_pad}_${end_pad}"
    local clip_meta="${out_dir}/meta_${SUBJECT}_${SCENE}_${SEQUENCE}_${EXO_CAM}_${start_pad}_${end_pad}.json"
    local post_complete=0

    if [[ "$RESUME_WINDOWS" == "1" && "$RUN_POST" == "1" ]]; then
        if is_post_complete_for_clip "$out_dir" "$take_name" "$expected_frames"; then
            post_complete=1
            if [[ -f "$clip_meta" ]]; then
                echo "✅ [RESUME] Window already complete, skipping compute: ${take_name}"
                merge_meta_entry_locked "$clip_meta" "$merged_meta"
                return 0
            fi
            echo "⚠ [RESUME] Outputs complete but clip meta missing: ${take_name}"
            echo "  Regenerating conversion metadata only, then merging into meta_all.json"
        fi
    fi

    if [[ -n "$DEFAULT_PROMPT" ]]; then
        python data_preprocess/h2o_to_egox.py \
            --h2o_root "$H2O_ROOT" \
            --subject "$SUBJECT" \
            --scene "$SCENE" \
            --sequence "$SEQUENCE" \
            --exo_cam "$EXO_CAM" \
            --output_dir "$out_dir" \
            --start_frame "$start" \
            --end_frame "$end" \
            --fps 30 \
            --text_root "$TEXT_ROOT" \
            --default_prompt "$DEFAULT_PROMPT" \
            --take_name_with_range
    else
        python data_preprocess/h2o_to_egox.py \
            --h2o_root "$H2O_ROOT" \
            --subject "$SUBJECT" \
            --scene "$SCENE" \
            --sequence "$SEQUENCE" \
            --exo_cam "$EXO_CAM" \
            --output_dir "$out_dir" \
            --start_frame "$start" \
            --end_frame "$end" \
            --fps 30 \
            --text_root "$TEXT_ROOT" \
            --take_name_with_range
    fi
    cp "${out_dir}/meta.json" "$clip_meta"

    if [[ "$RUN_POST" == "1" ]]; then
        if [[ "$RESUME_WINDOWS" == "1" && "$post_complete" == "1" ]]; then
            echo "✅ [RESUME] Post outputs already complete, skipping post stage: ${take_name}"
        else
            run_post_pipeline "$out_dir" "$clip_meta"
        fi
    fi

    if [[ -f "$clip_meta" ]]; then
        merge_meta_entry_locked "$clip_meta" "$merged_meta"
    fi
}

echo "=========================================="
echo "H2O Batch Processing (ego2exo-ready)"
echo "=========================================="
echo "Scene: $SCENE"
echo "Sequence: $SEQUENCE"
echo "Exo camera: $EXO_CAM"
echo "Strategy: $STRATEGY"
echo "Subject: $SUBJECT"
echo "H2O_ROOT: $H2O_ROOT"
echo "TEXT_ROOT: $TEXT_ROOT"
echo "DEFAULT_PROMPT: ${DEFAULT_PROMPT:-<none>}"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "EGOX_ENV: $EGOX_ENV"
echo "RUN_POST: $RUN_POST"
echo "FLAT_SINGLE_OUTPUT: $FLAT_SINGLE_OUTPUT"
echo "FLAT_MULTI_OUTPUT: $FLAT_MULTI_OUTPUT"
echo "EXTRACT_FROM_TAR: $EXTRACT_FROM_TAR"
echo "TAR_EXTRACT_MODE: $TAR_EXTRACT_MODE"
echo "TAR_SOURCE_DIR: $TAR_SOURCE_DIR"
echo "=========================================="

extract_h2o_from_tar

if [[ -z "$TOTAL_FRAMES" ]]; then
    rgb_dir="${H2O_ROOT}/${SUBJECT}/${SCENE}/${SEQUENCE}/${EXO_CAM}/rgb"
    if [[ -d "$rgb_dir" ]]; then
        TOTAL_FRAMES=$(find "$rgb_dir" -maxdepth 1 -type f -name '*.png' | wc -l)
    else
        TOTAL_FRAMES=450
    fi
fi
echo "TOTAL_FRAMES: $TOTAL_FRAMES"

case "$STRATEGY" in
    "single")
        # Strategy A: Single clip (0-48)
        echo -e "\n📹 Processing single clip (frames 0-48)..."
        if [[ "$FLAT_SINGLE_OUTPUT" == "1" ]]; then
            process_clip "$OUTPUT_ROOT" 0 48
        else
            process_clip "$OUTPUT_ROOT/${SUBJECT}_${SCENE}_${SEQUENCE}_${EXO_CAM}_single" 0 48
        fi
        ;;

    "no_overlap")
        # Strategy B: Non-overlapping windows (0-48, 49-97, 98-146, ...)
        echo -e "\n📹 Processing non-overlapping windows..."
        CLIP_ID=0
        for START in $(seq 0 $WINDOW_SIZE $((TOTAL_FRAMES - WINDOW_SIZE))); do
            END=$((START + WINDOW_SIZE - 1))
            echo -e "\nClip $CLIP_ID: frames $START-$END"
            if [[ "$FLAT_MULTI_OUTPUT" == "1" ]]; then
                process_clip "$OUTPUT_ROOT" "$START" "$END"
            else
                process_clip "$OUTPUT_ROOT/${SUBJECT}_${SCENE}_${SEQUENCE}_${EXO_CAM}_clip${CLIP_ID}" "$START" "$END"
            fi

            CLIP_ID=$((CLIP_ID + 1))
        done
        echo -e "\n✅ Processed $CLIP_ID non-overlapping clips"
        ;;

    "overlap")
        # Strategy C: Overlapping windows with 50% overlap (stride=25)
        echo -e "\n📹 Processing overlapping windows (50% overlap, stride=25)..."
        STRIDE=25
        CLIP_ID=0
        for START in $(seq 0 $STRIDE $((TOTAL_FRAMES - WINDOW_SIZE))); do
            END=$((START + WINDOW_SIZE - 1))
            echo -e "\nClip $CLIP_ID: frames $START-$END"
            if [[ "$FLAT_MULTI_OUTPUT" == "1" ]]; then
                process_clip "$OUTPUT_ROOT" "$START" "$END"
            else
                process_clip "$OUTPUT_ROOT/${SUBJECT}_${SCENE}_${SEQUENCE}_${EXO_CAM}_overlap_clip${CLIP_ID}" "$START" "$END"
            fi

            CLIP_ID=$((CLIP_ID + 1))
        done
        echo -e "\n✅ Processed $CLIP_ID overlapping clips"
        ;;

    *)
        echo "❌ Unknown strategy: $STRATEGY"
        echo "   Valid options: single | no_overlap | overlap"
        exit 1
        ;;
esac

if [[ -f "${OUTPUT_ROOT}/meta_all.json" ]]; then
    cp "${OUTPUT_ROOT}/meta_all.json" "${OUTPUT_ROOT}/meta.json"
fi

echo -e "\n=========================================="
echo "✅ Batch processing complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_ROOT"
if [[ "$RUN_POST" == "1" ]]; then
    echo "Completed: conversion + ego2exo vipe/render/depthmaps"
else
    echo "Completed: conversion only (set run_post=1 to enable ego2exo post steps)"
fi
