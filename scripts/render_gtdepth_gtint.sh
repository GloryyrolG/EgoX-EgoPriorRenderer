#!/bin/bash
# Render ego prior from GT depth using GT ego intrinsics (same as ego_Prior_gtint for ViPE prior).
# H2O cam4 intrinsics: fx fy cx cy 1280 720 -> scaled to 448x448 (stretch, same as training).
# Source: 636.6593017578125 636.251953125 635.283881879317 366.8740353496978 1280 720
# Scaled: fx_448=222.831 fy_448=395.519 cx_448=222.651 cy_448=228.210

BASE="processed/h2o"
python scripts/render_vipe_pointcloud.py \
    --gtdepth_dir "$BASE" \
    --meta_json_path "$BASE/meta.json" \
    --out_dir "$BASE/videos" \
    --start_frame 24 \
    --end_frame 24 \
    --override_ego_intrinsics 222.831 395.519 222.651 228.210 \
    --no_aria
