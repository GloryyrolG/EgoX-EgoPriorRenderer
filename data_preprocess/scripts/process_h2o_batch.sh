#TODO: test
#!/bin/bash
# Batch process H2O sequences with different windowing strategies
# Usage: bash process_h2o_batch.sh h1 0 cam0 [strategy]
#   strategy: single (default) | no_overlap | overlap

set -e

H2O_ROOT="/data448/rongyu_chen/dses/h2o/subject1"
OUTPUT_ROOT="./h2o_processed"
SCENE=${1:-"h1"}
SEQUENCE=${2:-"0"}
EXO_CAM=${3:-"cam0"}
STRATEGY=${4:-"single"}  # single | no_overlap | overlap

WINDOW_SIZE=49  # EgoX fixed window
TOTAL_FRAMES=450  # H2O default

echo "=========================================="
echo "H2O Batch Processing"
echo "=========================================="
echo "Scene: $SCENE"
echo "Sequence: $SEQUENCE"
echo "Exo camera: $EXO_CAM"
echo "Strategy: $STRATEGY"
echo "=========================================="

python data_preprocess/h2o_to_egox.py \
    --h2o_root /data448/rongyu_chen/dses/h2o/subject1 \
    --scene h1 \
    --sequence 0 \
    --exo_cam cam0 \
    --output_dir ./processed/h2o_h1_0_cam0 \
    --start_frame 0 \
    --end_frame 48

case "$STRATEGY" in
    "single")
        # Strategy A: Single clip (0-48)
        echo -e "\n📹 Processing single clip (frames 0-48)..."
        python data_preprocess/h2o_to_egox.py \
            --h2o_root "$H2O_ROOT" \
            --scene "$SCENE" \
            --sequence "$SEQUENCE" \
            --exo_cam "$EXO_CAM" \
            --output_dir "$OUTPUT_ROOT/${SCENE}_${SEQUENCE}_${EXO_CAM}_single" \
            --start_frame 0 \
            --end_frame 48 \
            --fps 30
        ;;

    "no_overlap")
        # Strategy B: Non-overlapping windows (0-48, 49-97, 98-146, ...)
        echo -e "\n📹 Processing non-overlapping windows..."
        CLIP_ID=0
        for START in $(seq 0 $WINDOW_SIZE $((TOTAL_FRAMES - WINDOW_SIZE))); do
            END=$((START + WINDOW_SIZE - 1))
            echo -e "\nClip $CLIP_ID: frames $START-$END"

            python data_preprocess/h2o_to_egox.py \
                --h2o_root "$H2O_ROOT" \
                --scene "$SCENE" \
                --sequence "$SEQUENCE" \
                --exo_cam "$EXO_CAM" \
                --output_dir "$OUTPUT_ROOT/${SCENE}_${SEQUENCE}_${EXO_CAM}_clip${CLIP_ID}" \
                --start_frame $START \
                --end_frame $END \
                --fps 30

            ((CLIP_ID++))
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

            python data_preprocess/h2o_to_egox.py \
                --h2o_root "$H2O_ROOT" \
                --scene "$SCENE" \
                --sequence "$SEQUENCE" \
                --exo_cam "$EXO_CAM" \
                --output_dir "$OUTPUT_ROOT/${SCENE}_${SEQUENCE}_${EXO_CAM}_overlap_clip${CLIP_ID}" \
                --start_frame $START \
                --end_frame $END \
                --fps 30

            ((CLIP_ID++))
        done
        echo -e "\n✅ Processed $CLIP_ID overlapping clips"
        ;;

    *)
        echo "❌ Unknown strategy: $STRATEGY"
        echo "   Valid options: single | no_overlap | overlap"
        exit 1
        ;;
esac

echo -e "\n=========================================="
echo "✅ Batch processing complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_ROOT"
echo ""
echo "Next: Run ViPE + rendering for each generated meta.json"
