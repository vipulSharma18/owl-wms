# Profiling

We are aiming to hit relatively high frame-rates on models. The purpose of this folder is for profiling tests.

## Example run:
python -m profiling.generic_forward

or with logs:

TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="all" TORCH_TRACE="/tmp/tracedir" python -m profiling.generic_forward > out.log 2>&1
tlparse /tmp/tracedir/dedicated_log_torch_trace_uedlymt_.log --overwrite

This generates logs at:
1. `physicsnemo_profiling_outputs/torch` for general trace of code, including the kernel level.
2. `tl_out` Torch logs parsed to give compilation logs and code for each kernel.
3. Torch inductor kernel wise logs in a path like, if benchmark_kernel is enabled with unique kernel names: ` /tmp/torchinductor_root/he/cheyf5tfvypbys2w6e4vn2g5t4scttjhaxvqsyd5how6f6xp7ock.py` containing both 1 and 2.

## Sample Model Checkpoints:
audio vae:
```
wget https://model-checkpoints.fly.storage.tigris.dev/cod_audio_20k_ema.pt -O checkpoints/owl_vaes/cod_audio_20k_ema.pt
```
image vae:
```
wget https://model-checkpoints.fly.storage.tigris.dev/cod_128x_30k_ema.pt -O checkpoints/owl_vaes/cod_128x_30k_ema.pt
```
200m av wm:
```
wget https://model-checkpoints.fly.storage.tigris.dev/av_dfot_85k_ema_200m.pt -O checkpoints/av_huge/av_dfot_85k_ema_200m.pt
```

### Reference docs for FP8 and Compiler Optimization:
* https://github.com/pytorch/ao
* https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/flux/README.md
* https://github.com/xdit-project/xDiT/tree/main
* https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html
* https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/diffusers/quantization/README.md

## Testing Models for Sanity Check:
Use popular video models for sanity checking methodology:
1. https://github.com/Wan-Video/Wan2.1

## Notes:
Use this PR to enable FP8 Rowwise scaling, need to use torch nightly: https://github.com/pytorch/pytorch/pull/155991

## TODO:
1. Debug FP8 autotune to wrap up FP8 inference.
2. Patch torch to use nightly or at least support row-wise scaling + cutlass code for FP8 on 5090.  -> low priority if not that much speed up. Although, just need to use nightly.

^ autoquant with all dtypes working -> best possible in native torch. now move on to tensorRT.

TensorRT:
4. Torch -> Compile/optimize with Torch TensorRT -> ONNX with serving on TensorRT backend or ONNX backend with TensorRT.

FP8 training using torchao.  -> training is low priority, we need fast models.