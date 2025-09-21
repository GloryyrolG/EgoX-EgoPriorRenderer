#!/bin/bash

# Cooking 데이터셋 ego video 프레임 추출 스크립트
# 각 take_name에 대해 ego_view_rendering/take_name/gt_output에 프레임 저장

# 기본 경로 설정
DATA_DIR="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Ego4d/DATA"
OUTPUT_BASE_DIR="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/vipe/ego_view_rendering"
EXTRACT_SCRIPT_DIR="/home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/vipe/ego_view_rendering"

# 추출할 프레임 수
NUM_FRAMES=600

# Cooking 데이터셋 리스트
cooking_datasets=(
    "fair_cooking_05_2"
    "georgiatech_cooking_01_01_2" 
    "iiith_cooking_01_1"
    "indiana_cooking_01_2"
    "minnesota_cooking_010_2"
    "nus_cooking_06_2"
    "sfu_cooking015_2"
    "uniandes_cooking_001_10"
)

# 프레임 추출 함수
extract_frames_for_dataset() {
    local take_name=$1
    
    # 비디오 디렉토리
    local video_dir="${DATA_DIR}/takes/${take_name}/frame_aligned_videos/downscaled/448"
    
    # ego view 비디오 파일 자동 탐지 (aria*_214-1.mp4 패턴)
    local video_path=""
    if [ -d "$video_dir" ]; then
        # aria로 시작하는 ego view 파일 찾기
        video_path=$(find "$video_dir" -name "aria*_214-1.mp4" | head -1)
        
        # 못 찾으면 다른 패턴도 시도
        if [ -z "$video_path" ]; then
            video_path=$(find "$video_dir" -name "aria*.mp4" | head -1)
        fi
    fi
    
    # 출력 디렉토리 경로  
    local output_dir="${OUTPUT_BASE_DIR}/${take_name}/gt_output"
    
    echo "=========================================="
    echo "Processing: $take_name"
    echo "=========================================="
    echo "Video directory: $video_dir"
    
    # 디렉토리 내 파일 목록 표시
    if [ -d "$video_dir" ]; then
        echo "Available video files:"
        ls -la "$video_dir"/*.mp4 2>/dev/null || echo "  No .mp4 files found"
    else
        echo "❌ ERROR: Video directory not found: $video_dir"
        return 1
    fi
    
    echo "Selected video: $video_path"
    echo "Output: $output_dir"
    
    # 비디오 파일 존재 확인
    if [[ ! -f "$video_path" ]]; then
        echo "❌ ERROR: No suitable ego video file found in: $video_dir"
        echo "   Looked for: aria*_214-1.mp4 or aria*.mp4"
        return 1
    fi
    
    # 출력 디렉토리 생성
    mkdir -p "$output_dir"
    
    # 비디오 정보 확인
    echo "Getting video info..."
    python3 "${EXTRACT_SCRIPT_DIR}/extract_frames_from_mp4.py" \
        --video_path "$video_path" \
        --info_only
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to get video info for $take_name"
        return 1
    fi
    
    # 프레임 추출 실행
    echo "Extracting $NUM_FRAMES frames..."
    python3 "${EXTRACT_SCRIPT_DIR}/extract_frames_from_mp4.py" \
        --video_path "$video_path" \
        --output_dir "$output_dir" \
        --num_frames $NUM_FRAMES
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully extracted frames for $take_name"
        echo "   Output: $output_dir"
        # 추출된 프레임 개수 확인
        frame_count=$(find "$output_dir" -name "frame_*.png" | wc -l)
        echo "   Extracted frames: $frame_count"
    else
        echo "❌ ERROR: Failed to extract frames for $take_name"
        return 1
    fi
    
    echo ""
}

# 전체 프레임 추출 시작
echo "Starting frame extraction for cooking datasets..."
echo "Total datasets: ${#cooking_datasets[@]}"
echo "Frames per dataset: $NUM_FRAMES"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo ""

# 성공/실패 카운터
success_count=0
total_count=${#cooking_datasets[@]}

# 모든 데이터셋에 대해 프레임 추출 실행
for take_name in "${cooking_datasets[@]}"; do
    extract_frames_for_dataset "$take_name"
    
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
done

echo "=========================================="
echo "Frame extraction completed!"
echo "=========================================="
echo "Success: $success_count / $total_count datasets"

# 최종 결과 요약
echo ""
echo "Extraction summary:"
for take_name in "${cooking_datasets[@]}"; do
    output_dir="${OUTPUT_BASE_DIR}/${take_name}/gt_output"
    if [ -d "$output_dir" ]; then
        frame_count=$(find "$output_dir" -name "frame_*.png" | wc -l)
        echo "  ✅ $take_name: $frame_count frames"
    else
        echo "  ❌ $take_name: Failed"
    fi
done

echo ""
echo "All frame extraction tasks completed!"