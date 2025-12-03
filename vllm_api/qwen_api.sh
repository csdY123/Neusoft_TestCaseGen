# conda  activate /mnt/lfd/anaconda3/envs/vllm_env/
PORT=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=$GPU
export TP=1
vllm serve /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/models/Qwen3-8B \
    -dp 1 \
    --tensor-parallel-size $TP \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --gpu-memory-utilization 0.4 \
    --max-model-len 32768 \
    --served-model-name Qwen3-8B \
    --host 0.0.0.0 \
    --port $PORT \
    # --enable-prefix-caching \
    # --enable-chunked-prefill
