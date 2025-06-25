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

^ autoquant with all dtypes working -> best possible in native torch. now move on to tensorRT.

TensorRT:
4. Torch -> Compile/optimize with Torch TensorRT -> ONNX with serving on TensorRT backend or ONNX backend with TensorRT.

FP8 training using torchao.  -> training is low priority, we need fast models.


2. torchao bugs:

W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] WON'T CONVERT _quantized_linear_op /app/.venv/lib/python3.12/site-packages/torchao/quantization/autoquant.py line 875
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] ========== TorchDynamo Stack Trace ==========

W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]   File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 1860, in assert_not_wrapped_by_this_graph
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]     if is_fake(value) and maybe_get_fake_mode(value) is self.tx.fake_mode:
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]        ^^^^^^^^^^^^^^
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]   File "/app/.venv/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 193, in is_fake
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]     attrs, _ = type(x).__tensor_flatten__(x)
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]   File "/app/.venv/lib/python3.12/site-packages/torchao/utils.py", line 570, in __tensor_flatten__
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]     raise NotImplementedError("Subclasses must implement __tensor_flatten__")
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] torch._dynamo.exc.InternalTorchDynamoError: NotImplementedError: Subclasses must implement __tensor_flatten__
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] 
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] from user code:
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]    File "/app/.venv/lib/python3.12/site-packages/torchao/quantization/autoquant.py", line 881, in _quantized_linear_op
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339]     w_qtensor.weight,
W0625 03:25:44.055000 127224 .venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py:1339] 
I0625 03:25:44.056000 127224 .venv/lib/python3.12/site-packages/torch/_utils_internal.py:122] dynamo _convert_frame_assert._compile: {'co_name': '_dispatch__torch_function__', 'frame_id': 1, 'compile_id': '1/0', 'co_filename': '/app/.venv/lib/python3.12/site-packages/torchao/utils.py', 'co_firstlineno': 411, 'cache_size': 0, 'accumulated_cache_size': 0}
