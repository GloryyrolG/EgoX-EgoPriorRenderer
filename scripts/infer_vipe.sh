#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=5

# 실험 설정을 위한 associative array
declare -A experiments=(
    #["cmu_bike01_2"]="cam02 static_vda"
    
    # Cooking datasets
    #["fair_cooking_05_2"]=#"cam04 static_vda"
    ["georgiatech_cooking_01_01_2"]="cam03 static_vda" #"cam01 static_vda"
    ["iiith_cooking_01_1"]="cam01 static_vda"
    ["indiana_cooking_01_2"]="cam03 static_vda"
    ["minnesota_cooking_010_2"]="cam01 static_vda"
    ["nus_cooking_06_2"]="cam02 static_vda"
    ["sfu_cooking015_2"]="cam04 static_vda"
    ["uniandes_cooking_001_10"]="cam02 static_vda"
)

# 공통 설정
BASE_PATH="/home/nas4_user/kinamkim/DATA/Ego4D/data"
START_FRAME=0
END_FRAME=600

# 추론 실행 함수
run_inference() {
    local sequence=$1
    local camera=$2
    local pipeline=$3
    
    local video_path="${BASE_PATH}/${sequence}/frame_aligned_videos/downscaled/448/${camera}.mp4"
    
    echo "=========================================="
    echo "Processing: ${sequence}/${camera}"
    echo "Video path: $video_path"
    echo "Frame range: $START_FRAME to $END_FRAME"
    echo "Pipeline: $pipeline"
    echo "=========================================="
    
    # 비디오 파일 존재 여부 확인
    if [[ ! -f "$video_path" ]]; then
        echo "ERROR: Video file not found: $video_path"
        return 1
    fi
    
    # ViPE 추론 실행
    vipe infer "$video_path" \
        --start_frame $START_FRAME \
        --end_frame $END_FRAME \
        --assume_fixed_camera_pose \
        --pipeline $pipeline
        
    echo "Completed: ${sequence}/${camera}"
    echo ""
}

# 모든 실험 실행
for exp_name in "${!experiments[@]}"; do
    # 공백으로 분리해서 파라미터 추출
    IFS=' ' read -r camera pipeline <<< "${experiments[$exp_name]}"
    
    run_inference "$exp_name" "$camera" "$pipeline"
done

echo "All inference experiments completed!"

# 예시: 다른 비디오도 실행하고 싶다면 주석을 해제
# echo "Running additional demo video..."
# vipe infer /home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Geo4D/data/demo/drift-turn.mp4 \
#     --pipeline static_vda