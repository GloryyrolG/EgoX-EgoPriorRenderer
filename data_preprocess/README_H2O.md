# H2O Dataset Processing for EgoX

This guide explains how to process H2O dataset for EgoX egocentric video generation.

## 📋 Overview

The H2O dataset contains:
- **cam0-cam3**: 4 fixed exocentric cameras (1280x720)
- **cam4**: 1 head-mounted egocentric camera (1280x720)
- **Scene types**: h1, h2, k1, k2, o1, o2
- **Sequences**: Multiple sequences per scene (0, 1, 2, ...)

## 🔧 Format Conversion

### H2O → EgoX Mapping

| H2O Format | EgoX Format | Conversion |
|------------|-------------|------------|
| `cam_intrinsics.txt`<br/>`fx fy cx cy w h` | `camera_intrinsics`<br/>`[[fx,0,cx], [0,fy,cy], [0,0,1]]` | 6 values → 3x3 matrix |
| `cam_pose/*.txt`<br/>4x4 camera-to-world | `camera_extrinsics`<br/>3x4 world-to-camera | Inverse + take 3x4 |
| `rgb/*.png` | `exo.mp4` | ffmpeg conversion |
| cam4 `cam_pose/*.txt` | `ego_extrinsics`<br/>List of 3x4 per frame | Inverse + take 3x4 for each frame |

### Key Confirmations

✅ **H2O cam_pose = camera-to-world** (verified by checking translation vectors)
✅ **EgoX needs world-to-camera** → requires matrix inversion
✅ **cam0-cam3 are fixed** → single extrinsic matrix
✅ **cam4 is moving** → per-frame extrinsic matrices

## 🚀 Quick Start

### Process a Single Sequence

```bash
cd /data/rongyu_chen/projs/EgoX/EgoX-EgoPriorRenderer

# Process scene=h1, sequence=0, using cam0 as exo view
bash data_preprocess/scripts/process_h2o_single.sh h1 0 cam0
```

### Manual Step-by-Step

```bash
# 1. Convert H2O to EgoX format
python data_preprocess/h2o_to_egox.py \
    --subject subject1 \
    --scene h1 \
    --sequence 0 \
    --exo_cam cam0 \
    --start_frame 0

# 2. Run ViPE inference (extract depth + camera pose from exo video)
vipe infer \
    --video ./h2o_processed/h1_0_cam0/videos/h1_0_cam0/exo.mp4 \
    --output ./h2o_processed/h1_0_cam0/vipe_output \
    --assume_fixed_camera_pose \
    --pipeline lyra \
    --use_exo_intrinsic_gt "[[639.23, 0, 636.44], [0, 639.03, 367.50], [0, 0, 1]]" \
    --start_frame 0 \
    --end_frame 48

# 3. Render ego_Prior video from point cloud
python scripts/render_vipe_pointcloud.py \
    --meta_json ./h2o_processed/h1_0_cam0/meta.json \
    --vipe_result_dir ./h2o_processed/h1_0_cam0/vipe_output \
    --output_dir ./h2o_processed/h1_0_cam0/videos/h1_0_cam0

# 4. Run EgoX inference (in main EgoX repo)
cd /data/rongyu_chen/projs/EgoX
python infer.py \
    --meta_data_file ./EgoX-EgoPriorRenderer/h2o_processed/h1_0_cam0/meta.json \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results/h2o \
    --seed 42 \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
    --in_the_wild
```

## 📁 Output Structure

```
processed/                        # Like example
└── h2o/                          # Like egoexo4D
    ├── meta.json                 # EgoX metadata
    ├── videos/
    |   └── subject1_h1_0_cam0/
    |       ├── exo.mp4           # Exocentric video
    |       └── ego_Prior.mp4     # Ego prior (rendered from point cloud)
    └── depth_maps/
        └── subject1_h1_0_cam0/
            ├── 00000.npy
            └── ...

vipe_results/
└── subject1_h1_0_cam0/
    ├── depth/
    ├── vipe/
    └── ...
```

## 🔍 Verification Checklist

Before running EgoX inference, verify:

- [ ] `exo.mp4` video created (49 frames, 1280x720)
- [ ] `meta.json` contains:
  - [ ] `camera_intrinsics`: 3x3 matrix
  - [ ] `camera_extrinsics`: 3x4 matrix (single, for fixed cam)
  - [ ] `ego_intrinsics`: 3x3 matrix
  - [ ] `ego_extrinsics`: List of 49 × 3x4 matrices
- [ ] ViPE output contains depth maps
- [ ] `ego_Prior.mp4` rendered successfully

## 📊 Dataset Statistics

**subject1 structure:**
```
subject1/
├── h1, h2, k1, k2, o1, o2    # Scene types
│   └── 0, 1, 2, ...          # Sequence IDs
│       ├── cam0-cam3/        # Fixed exo cameras
│       │   ├── rgb/          # 450 frames
│       │   ├── depth/
│       │   ├── cam_pose/     # 450 pose files
│       │   └── cam_intrinsics.txt
│       └── cam4/             # Ego camera
│           ├── rgb/          # 450 frames
│           ├── cam_pose/     # 450 pose files (moving)
│           └── cam_intrinsics.txt
```

## 🎯 Camera Selection Strategy

Choose exocentric camera based on:
- **cam0**: Best frontal view of the subject
- **cam1-cam3**: Alternative viewpoints

You can process the same sequence with different exo cameras to compare results.

## 🛠️ Troubleshooting

### Issue: "Pose file not found"
- Check frame range: H2O has 450 frames, but EgoX processes 49 frames (0-48)
- Use `--start_frame 0 --end_frame 48`

### Issue: "Video creation failed"
- Ensure ffmpeg is installed: `conda install ffmpeg -c conda-forge`
- Check RGB images exist: `ls /data448/rongyu_chen/dses/h2o/subject1/h1/0/cam0/rgb/`

### Issue: "ViPE inference fails"
- GPU memory: ViPE requires ~40GB VRAM
- Use `--pipeline lyra_no_vda` to reduce memory usage

### Issue: "Ego prior rendering fails"
- Verify ego extrinsics are loaded (should be 49 frames)
- Check depth maps exist in ViPE output

## 📚 References

- **H2O Dataset**: https://github.com/taeinkwon/h2odataset
- **EgoX Paper**: https://arxiv.org/abs/2512.08269
- **EgoX Repo**: https://github.com/DAVIAN-Robotics/EgoX
- **ViPE**: Video Pose Estimation pipeline (bundled with EgoX-EgoPriorRenderer)

## 💡 Tips for Best Results

1. **Start with a short sequence** (49 frames) to validate the pipeline
2. **Choose cam0 first** as it typically has the best view
3. **Verify each step** before proceeding to the next
4. **Use ground truth intrinsics** (`--use_exo_intrinsic_gt`) for better ViPE results
5. **Fixed camera assumption** is critical for H2O (cam0-cam3 are stationary)

## 🔄 Batch Processing

To process multiple sequences, modify the script or use a loop:

```bash
for scene in h1 h2 k1; do
    for seq in 0 1 2; do
        bash data_preprocess/scripts/process_h2o_single.sh $scene $seq cam0
    done
done
```

This will process all combinations and create separate output directories.
