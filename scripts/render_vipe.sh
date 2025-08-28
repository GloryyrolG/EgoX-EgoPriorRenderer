#!/bin/bash

# Set CUDA device to use
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="vipe_results/cam02_static_vda_fixedcam" 
OUTPUT_DIR="ego_view_rendering"
EGO_CAMERA_POSE_PATH="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Ego4d/DATA/annotations/ego_pose/train/camera_pose/ed3ec638-8363-4e1d-9851-c7936cbfad8c.json"
EXO_CAMERA_POSE_PATH="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Ego4d/DATA/takes/cmu_bike01_2/trajectory/gopro_calibs.csv"
ONLINE_CALIBRATION_PATH="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Ego4d/DATA/takes/cmu_bike01_2/trajectory/online_calibration.jsonl"
POINT_SIZE="0.5"
NUM_FRAMES="60"
#BATCH_SIZE="8"  # Batch size for GPU acceleration


python ego_view_rendering/render_vipe_pointcloud.py \
    --input_dir $INPUT_DIR \
    --out_dir $OUTPUT_DIR \
    --ego_camera_pose_path $EGO_CAMERA_POSE_PATH \
    --exo_camera_pose_path $EXO_CAMERA_POSE_PATH \
    --online_calibration_path $ONLINE_CALIBRATION_PATH \
    --point_size $POINT_SIZE \
    --num_frames $NUM_FRAMES \
    --fish_eye_rendering \
    #--batch_size $BATCH_SIZE \