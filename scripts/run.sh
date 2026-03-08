#!/bin/bash
BASE_DIR="/workspace/gaussian_avatar/datasets/zju_mocap/CoreView_377"

for cam_dir in "$BASE_DIR"/Camera_B*; do
    # Skip existing mask output directories
    [[ "$cam_dir" == *_masks ]] && continue
    cam_name=$(basename "$cam_dir")
    output_dir="$BASE_DIR/${cam_name}_masks"

    # Skip if output dir exists and has the same number of images
    src_count=$(ls "$cam_dir"/*.jpg 2>/dev/null | wc -l)
    dst_count=$(ls "$output_dir"/*.png 2>/dev/null | wc -l)
    if [ "$src_count" -gt 0 ] && [ "$src_count" -eq "$dst_count" ]; then
        echo "=== Skipping $cam_name (already complete: $dst_count/$src_count) ==="
        continue
    fi

    echo "=== Processing $cam_name ==="
    uv run sam3-infer \
        --images "$cam_dir" \
        --text 'a person' \
        --output "$output_dir" \
        --alpha \
        --batch-size 8
done
