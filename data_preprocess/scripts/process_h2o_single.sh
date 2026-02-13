#TODO: test
#!/bin/bash
# Process a single H2O sequence for EgoX
# Usage: bash process_h2o_single.sh h1 0 cam0

set -e

# Configuration
H2O_ROOT="/data448/rongyu_chen/dses/h2o/subject1"
OUTPUT_ROOT="./h2o_processed"
SCENE=${1:-"h1"}
SEQUENCE=${2:-"0"}
EXO_CAM=${3:-"cam0"}

echo "=========================================="
echo "H2O to EgoX Processing Pipeline"
echo "=========================================="
echo "Scene: $SCENE"
echo "Sequence: $SEQUENCE"
echo "Exo camera: $EXO_CAM"
echo "=========================================="

# Step 1: Convert H2O to EgoX format
echo -e "\n[Step 1/3] Converting H2O to EgoX format..."
python data_preprocess/h2o_to_egox.py \
    --h2o_root "$H2O_ROOT" \
    --scene "$SCENE" \
    --sequence "$SEQUENCE" \
    --exo_cam "$EXO_CAM" \
    --output_dir "$OUTPUT_ROOT/${SCENE}_${SEQUENCE}_${EXO_CAM}" \
    --fps 30 \
    --start_frame 0 \
    --end_frame 48  # EgoX processes 49 frames (0-48)

OUTPUT_DIR="$OUTPUT_ROOT/${SCENE}_${SEQUENCE}_${EXO_CAM}"
EXO_VIDEO="$OUTPUT_DIR/videos/${SCENE}_${SEQUENCE}_${EXO_CAM}/exo.mp4"
META_JSON="$OUTPUT_DIR/meta.json"

# Extract intrinsics from meta.json
INTRINSICS=$(python3 -c "
import json
with open('$META_JSON') as f:
    data = json.load(f)
    K = data['test_datasets'][0]['camera_intrinsics']
    print(f'[[{K[0][0]}, 0, {K[0][2]}], [0, {K[1][1]}, {K[1][2]}], [0, 0, 1]]')
")

# Step 2: Run ViPE inference
echo -e "\n[Step 2/3] Running ViPE inference..."
vipe infer \
    --video "$EXO_VIDEO" \
    --output "$OUTPUT_DIR/vipe_output" \
    --assume_fixed_camera_pose \
    --pipeline lyra \
    --use_exo_intrinsic_gt "$INTRINSICS" \
    --start_frame 0 \
    --end_frame 48

# Step 3: Render ego_Prior video
echo -e "\n[Step 3/3] Rendering ego_Prior video..."
python scripts/render_vipe_pointcloud.py \
    --meta_json "$META_JSON" \
    --vipe_result_dir "$OUTPUT_DIR/vipe_output" \
    --output_dir "$OUTPUT_DIR/videos/${SCENE}_${SEQUENCE}_${EXO_CAM}"

echo -e "\n=========================================="
echo "✅ Processing complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Meta JSON: $META_JSON"
echo "Ego Prior video: $OUTPUT_DIR/videos/${SCENE}_${SEQUENCE}_${EXO_CAM}/ego_Prior.mp4"
echo -e "\nYou can now run EgoX inference with this meta.json file."
