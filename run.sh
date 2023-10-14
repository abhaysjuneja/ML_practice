docker run --gpus all \
--mount type=bind,source="$(pwd)/",target=/ML \
--rm \
-itd ml_torch_gpu
