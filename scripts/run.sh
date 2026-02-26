uv run sam3-infer --images /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/male-4-casual/cam000/images \
    --text 'a person' \
    --output /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/male-4-casual/cam000/images2 \
    --alpha \
    --erode-pixels 2;

uv run sam3-infer --images /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/female-3-casual/cam000/images \
    --text 'a person' \
    --output /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/female-3-casual/cam000/images2 \
    --alpha \
    --erode-pixels 2;

uv run sam3-infer --images /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/female-4-casual/cam000/images \
    --text 'a person' \
    --output /home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected/female-4-casual/cam000/images2 \
    --alpha \
    --erode-pixels 2;