#!/bin/bash

# Activate vipe conda environment
source /home/nas5/taewoongkang/anaconda3/bin/activate vipe

# 사용 가능한 GPU 디바이스 설정 (,로 구분)
AVAILABLE_GPUS="0,1"  # 사용하고 싶은 GPU ID들을 여기에 설정

# GPU당 최대 동시 실행 프로세스 수 (메모리 부족 방지)
MAX_PROCESSES_PER_GPU=15

# GPU 목록을 배열로 변환
IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
MAX_CONCURRENT_JOBS=$((NUM_GPUS * MAX_PROCESSES_PER_GPU))

echo "Using ${NUM_GPUS} GPUs: ${AVAILABLE_GPUS}"
echo "Max ${MAX_PROCESSES_PER_GPU} processes per GPU (total: ${MAX_CONCURRENT_JOBS} concurrent jobs)"

# cooking_best_exo.json 파일 경로
COOKING_JSON="/home/nas_main/taewoongkang/dohyeon/Exo-to-Ego/Ego4D_4DNeX_dataset/cooking_best_exo.json"

# Python을 사용해서 JSON 파일 파싱하고 실험 목록 생성
generate_experiments() {
    python3 << 'EOF'
import json
import sys

cooking_json_path = "/home/nas_main/taewoongkang/dohyeon/Exo-to-Ego/Ego4D_4DNeX_dataset/cooking_best_exo.json"

try:
    with open(cooking_json_path, 'r') as f:
        data = json.load(f)
    
    experiments = []
    for take_name, info in data.items():
        best_exo = info.get('best_exo')
        take_uuid = info.get('take_uuid')
        
        # best_exo가 null이 아닌 경우만 처리
        if best_exo is not None and take_uuid is not None:
            result_subdir = f"{best_exo}_moge_static_vda_fixedcam_slammap_exo_intr_gt"
            experiments.append(f"{take_name}|{result_subdir}|{take_uuid}")
    
    print('\n'.join(experiments))
    
except Exception as e:
    print(f"Error parsing JSON: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# 공통 설정
OUTPUT_DIR="ego_view_rendering"
POINT_SIZE="5.0"
START_FRAME="0"
END_FRAME="48"

# 렌더링 실행 함수 (특정 GPU에서 실행)
run_experiment() {
    local take_name=$1
    local result_subdir=$2
    local ego_uuid=$3
    local gpu_id=$4
    
    # meta.json에서 start_frame 정보 읽기
    local meta_json="/home/nas_main/taewoongkang/dohyeon/Exo-to-Ego/Ego4D_4DNeX_dataset/clip/meta.json"
    local start_frame=$(python3 -c "
import json
import sys
import re
try:
    with open('$meta_json', 'r') as f:
        data = json.load(f)
    
    take_name = '${take_name}'
    start_frame = None
    
    # aria**_214-1.mp4 패턴 찾기
    pattern = f'{take_name}/frame_aligned_videos/downscaled/448/aria\d+_214-1\.mp4'
    for key in data.keys():
        if re.match(pattern, key):
            start_frame = data[key]['start_frame']
            break
    
    print(start_frame if start_frame is not None else '0')
except:
    print('0')  # 오류시 기본값
")
    
    # source_start_frame과 source_end_frame 계산
    local SOURCE_START_FRAME=$start_frame
    local SOURCE_END_FRAME=$((start_frame + 48))
    
    # 경로 설정
    INPUT_DIR="vipe_results/${take_name}/${result_subdir}"
    EGO_CAMERA_POSE_PATH="/home/nas_main/kinamkim/dataset/Ego4D/dataset_train/annotations/ego_pose/train/camera_pose/${ego_uuid}.json"
    EXO_CAMERA_POSE_PATH="/home/nas_main/kinamkim/dataset/Ego4D/dataset_train/takes/${take_name}/trajectory/gopro_calibs.csv"
    ONLINE_CALIBRATION_PATH="/home/nas_main/kinamkim/dataset/Ego4D/dataset_train/takes/${take_name}/trajectory/online_calibration.jsonl"
    
    echo "[GPU $gpu_id] =========================================="
    echo "[GPU $gpu_id] Running experiment: ${take_name}/${result_subdir}"
    echo "[GPU $gpu_id] Input directory: $INPUT_DIR"
    echo "[GPU $gpu_id] Source frames: ${SOURCE_START_FRAME} - ${SOURCE_END_FRAME}"
    echo "[GPU $gpu_id] =========================================="
    
    # 입력 디렉토리 존재 여부 확인
    if [ ! -d "$INPUT_DIR" ]; then
        echo "[GPU $gpu_id] ERROR: Input directory not found: $INPUT_DIR"
        echo "[GPU $gpu_id] Skipping experiment: ${take_name}/${result_subdir}"
        return 1
    fi
    
    # 특정 GPU에서만 실행되도록 CUDA_VISIBLE_DEVICES 설정
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/render_vipe_pointcloud.py \
        --input_dir $INPUT_DIR \
        --out_dir $OUTPUT_DIR \
        --ego_camera_pose_path $EGO_CAMERA_POSE_PATH \
        --exo_camera_pose_path $EXO_CAMERA_POSE_PATH \
        --online_calibration_path $ONLINE_CALIBRATION_PATH \
        --point_size $POINT_SIZE \
        --start_frame $START_FRAME \
        --end_frame $END_FRAME \
        --source_start_frame $SOURCE_START_FRAME \
        --source_end_frame $SOURCE_END_FRAME \
        --fish_eye_rendering \
        --use_mean_bg \
        --only_bg
        
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] Completed: ${take_name}/${result_subdir}"
        return 0
    else
        echo "[GPU $gpu_id] Failed: ${take_name}/${result_subdir} (exit code: $exit_code)"
        return 1
    fi
}

# 백그라운드 프로세스 관리를 위한 배열
declare -a BACKGROUND_PIDS=()
declare -a RUNNING_JOBS=()

# Directory for per-experiment logs
LOG_DIR="render_logs"
mkdir -p "$LOG_DIR"

# JSON에서 실험 목록 생성
echo "Parsing cooking experiments from JSON..."
experiment_lines=$(generate_experiments)

if [ -z "$experiment_lines" ]; then
    echo "Error: No experiments found or failed to parse JSON"
    exit 1
fi

# 실험 목록을 배열로 변환
readarray -t experiment_list <<< "$experiment_lines"

# 변수 초기화
gpu_index=0
experiment_count=0
total_experiments=${#experiment_list[@]}
completed_experiments=0
failed_experiments=0

echo "Found $total_experiments cooking experiments to render"
echo "Starting controlled parallel execution with max $MAX_CONCURRENT_JOBS concurrent processes"

# 실험을 순차적으로 처리하되, 동시 실행 수를 제한
for experiment_line in "${experiment_list[@]}"; do
    # 동시 실행 수 제한 확인
    while [ ${#RUNNING_JOBS[@]} -ge $MAX_CONCURRENT_JOBS ]; do
        echo "Max concurrent jobs reached ($MAX_CONCURRENT_JOBS). Waiting for completion..."
        
        # 완료된 프로세스 확인 및 정리
        for i in "${!RUNNING_JOBS[@]}"; do
            job_info="${RUNNING_JOBS[$i]}"
            pid=$(echo "$job_info" | cut -d'|' -f1)
            
            if ! kill -0 $pid 2>/dev/null; then
                # 프로세스가 완료됨
                wait $pid
                exit_code=$?
                
                exp_info=$(echo "$job_info" | cut -d'|' -f2-)
                if [ $exit_code -eq 0 ]; then
                    ((completed_experiments++))
                    echo "Completed experiment: $exp_info"
                else
                    ((failed_experiments++))
                    echo "Failed experiment: $exp_info (exit code: $exit_code)"
                fi
                
                # 배열에서 제거
                unset RUNNING_JOBS[$i]
            fi
        done
        
        # 배열 재정렬 (빈 인덱스 제거)
        RUNNING_JOBS=("${RUNNING_JOBS[@]}")
        
        sleep 1
    done
    
    # 파라미터 파싱
    IFS='|' read -r take_name result_subdir ego_uuid <<< "$experiment_line"
    
    # 현재 GPU 선택 (라운드 로빈 방식)
    current_gpu=${GPU_ARRAY[$gpu_index]}
    
    echo "Starting experiment $((experiment_count + 1))/$total_experiments: $take_name on GPU $current_gpu"
    
    # 백그라운드에서 실험 실행, stdout/stderr를 개별 로그 파일로 리디렉션
    safe_name="${take_name//[^a-zA-Z0-9_]/_}-${result_subdir//[^a-zA-Z0-9_]/_}"
    out_file="$LOG_DIR/${safe_name}.out"
    err_file="$LOG_DIR/${safe_name}.err"

    run_experiment "$take_name" "$result_subdir" "$ego_uuid" "$current_gpu" >"$out_file" 2>"$err_file" &
    
    # 백그라운드 프로세스 PID와 실험 정보 저장
    job_pid=$!
    RUNNING_JOBS+=("$job_pid|$take_name/$result_subdir")
    
    # 다음 GPU로 이동 (라운드 로빈)
    gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))
    
    ((experiment_count++))
    
    # GPU 메모리 안정화를 위한 지연 (줄임)
    sleep 1
done

echo "All experiments submitted. Waiting for remaining processes to complete..."

# 남은 모든 프로세스가 완료될 때까지 대기
while [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    # 완료된 프로세스 확인 및 정리
    for i in "${!RUNNING_JOBS[@]}"; do
        job_info="${RUNNING_JOBS[$i]}"
        pid=$(echo "$job_info" | cut -d'|' -f1)
        
        if ! kill -0 $pid 2>/dev/null; then
            # 프로세스가 완료됨
            wait $pid
            exit_code=$?
            
            exp_info=$(echo "$job_info" | cut -d'|' -f2-)
            if [ $exit_code -eq 0 ]; then
                ((completed_experiments++))
                echo "Completed experiment: $exp_info"
            else
                ((failed_experiments++))
                echo "Failed experiment: $exp_info (exit code: $exit_code)"
            fi
            
            # 배열에서 제거
            unset RUNNING_JOBS[$i]
        fi
    done
    
    # 배열 재정렬 (빈 인덱스 제거)
    RUNNING_JOBS=("${RUNNING_JOBS[@]}")
    
    sleep 1
done

echo "=========================================="
echo "All rendering experiments completed!"
echo "Total experiments: $total_experiments"
echo "Completed successfully: $completed_experiments"
echo "Failed: $failed_experiments"
echo "Success rate: $(( completed_experiments * 100 / total_experiments ))%"
echo "=========================================="