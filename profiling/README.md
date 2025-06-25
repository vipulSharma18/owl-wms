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


**DEFAULT_AUTOQUANT_CLASS_LIST**  # int8 weights
Mean: 2.26ms, 4793.05MB    
Min: 2.06ms, 4793.05MB    
Max: 2.45ms, 4793.05MB    
Std: 0.03ms, 0.00MB    
Avg FPS: 442.88FPS    
Min FPS: 408.89FPS    
Max FPS: 485.61FPS    

**DEFAULT_INT4_AUTOQUANT_CLASS_LIST**  # mix of int8 and int4 weights
Mean: 2.11ms, 3909.25MB    
Min: 1.93ms, 3909.25MB    
Max: 2.31ms, 3909.25MB    
Std: 0.03ms, 0.00MB    
Avg FPS: 473.87FPS    
Min FPS: 433.59FPS    
Max FPS: 519.17FPS    

**GEMLITE_INT4_AUTOQUANT_CLASS_LIST**  # gemlite triton kernels
pass for now, bug in Float matmul
SingleProcess AUTOTUNE benchmarking takes 0.1109 seconds and 0.0903 seconds precompiling for 8 choices
Compiled module path: /tmp/torchinductor_root/ib/cibbekxgogi3mugl55hni6bylv7wb3er2vsrvwl5cucz3kb2wv3u.py
>>time: 0.007ms for <class 'torchao.quantization.autoquant.AQInt8DynamicallyQuantizedLinearWeight'> matmul, to_beat: 0.005ms
Compiled module path: /tmp/torchinductor_root/br/cbrluutbbxknx5enjyiv2lcdrqqzkyvegzj2k7paf2lsbi2y6xhq.py
>>time: 0.008ms for <class 'torchao.quantization.autoquant.AQGemliteInt4G64WeightOnlyQuantizedLinearWeight'>, to_beat: 0.005ms 
best_cls=<class 'torchao.quantization.autoquant.AQDefaultLinearWeight'>

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/profiling/generic_forward.py", line 100, in <module>
    profile_torch_compile_inductor_fp8_torchao(copy.deepcopy(world_model), copy.deepcopy(img_dec), copy.deepcopy(audio_dec), dummy, dummy_pred_audio)
  File "/app/profiling/inductor_compile.py", line 45, in profile_torch_compile_inductor_fp8_torchao
    compiled_world_model = torchao.autoquant(
                           ^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torchao/quantization/autoquant.py", line 1344, in autoquant
    model(*example_input)
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 375, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1879, in _call_impl
    return inner()
           ^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1827, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 749, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1871, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/output_graph.py", line 1846, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2398, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2002, in compile_fx
    return compile_fx(
           ^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2418, in compile_fx
    return aot_autograd(
           ^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/backends/common.py", line 109, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 1199, in aot_module_simplified
    compiled_fn = AOTAutogradCache.load(
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py", line 1140, in load
    compiled_fn = dispatch_and_compile()
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 1184, in dispatch_and_compile
    compiled_fn, _ = create_aot_dispatcher_function(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 576, in create_aot_dispatcher_function
    return _create_aot_dispatcher_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 836, in _create_aot_dispatcher_function
    compiled_fn, fw_metadata = compiler_fn(
                               ^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 245, in aot_dispatch_base
    compiled_fw = compiler(fw_module, updated_flat_args)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1851, in fw_compiler_freezing
    _recursive_joint_graph_passes(aot_autograd_model)
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 492, in _recursive_joint_graph_passes
    joint_graph_passes(gm)
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/joint_graph.py", line 601, in joint_graph_passes
    ).apply_graph_pass(patterns.apply)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/fx/passes/graph_transform_observer.py", line 85, in apply_graph_pass
    return pass_fn(self.gm.graph)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1961, in apply
    if is_match(m) and entry.extra_check(m):
                       ^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1498, in check_fn
    if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/pad_mm.py", line 148, in should_pad_addmm
    return should_pad_common(mat1, mat2, input) and should_pad_bench(
                                                    ^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/pad_mm.py", line 396, in should_pad_bench
    return _should_pad_bench(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/pad_mm.py", line 614, in _should_pad_bench
    ori_time = do_bench(orig_bench_fn)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/runtime/benchmarking.py", line 39, in wrapper
    return fn(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/runtime/benchmarking.py", line 243, in benchmark_gpu
    _callable()
  File "/app/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/pad_mm.py", line 585, in orig_bench_fn
    op(input, mat1, mat2)
  File "/app/.venv/lib/python3.12/site-packages/torch/_ops.py", line 1243, in __call__
    return self._op(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


**DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST**  # fp32, fp16, bf16 weights
pass: fails with missing as_proxy bug.
activation_shapes: torch.Size([1, 1536]), times_seen: 1
weight_shape: torch.Size([64, 1536]), dtype: torch.bfloat16, bias_shape: torch.Size([64])
>>time: 0.016ms for <class 'torchao.quantization.autoquant.AQFloat32LinearWeight'>, to_beat: infms 
>>time: 0.006ms for <class 'torchao.quantization.autoquant.AQBFloat16LinearWeight'>, to_beat: 0.016ms 
>>time: 0.016ms for <class 'torchao.quantization.autoquant.AQFloat16LinearWeight'>, to_beat: 0.006ms 
best_cls=<class 'torchao.quantization.autoquant.AQBFloat16LinearWeight'>

NotImplementedError: UnspecializedBuiltinNNModuleVariable(Linear)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/profiling/generic_forward.py", line 100, in <module>
    profile_torch_compile_inductor_fp8_torchao(copy.deepcopy(world_model), copy.deepcopy(img_dec), copy.deepcopy(audio_dec), dummy, dummy_pred_audio)
  File "/app/profiling/inductor_compile.py", line 45, in profile_torch_compile_inductor_fp8_torchao
    compiled_world_model = torchao.autoquant(
                           ^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torchao/quantization/autoquant.py", line 1344, in autoquant
    model(*example_input)
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 375, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1879, in _call_impl
    return inner()
           ^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1827, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 745, in compile_wrapper
    raise e.with_traceback(None) from e.__cause__  # User compiler error
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._dynamo.exc.Unsupported: Failed to convert args/kwargs to proxy
  Explanation: Missing `as_proxy()` implementation for some arg/kwarg.


  Developer debug context: call_function args: TensorVariable() GetAttrVariable(UnspecializedBuiltinNNModuleVariable(Linear), weight) ConstantVariable(NoneType: None) 

 For more details about this graph break, please visit: https://compile-graph-break-site.vercel.app/gb/GB0055

from user code:
   File "/app/owl_wms/models/gamerft_audio.py", line 41, in forward
    ctrl_cond = self.control_embed(mouse, btn)
  File "/app/owl_wms/nn/embeddings.py", line 185, in forward
    return self.mouse(mouse) + self.button(button)
  File "/app/owl_wms/nn/embeddings.py", line 153, in forward
    angle_emb = self.angle_proj(angle_emb)  # [b,n,dim//2]
  File "/app/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


**OTHER_AUTOQUANT_CLASS_LIST**  # fp8 weights
*  Torch Compile + TorchAO AutoQuant - WM    
Mean: 2.61ms, 4566.97MB    
Min: 2.32ms, 4566.97MB    
Max: 2.79ms, 4566.97MB    
Std: 0.04ms, 0.00MB    
Avg FPS: 383.28FPS    
Min FPS: 358.09FPS    
Max FPS: 430.39FPS    

**ALL_AUTOQUANT_CLASS_LIST**
Mean: 2.43ms, 5765.88MB    
Min: 2.21ms, 5765.88MB    
Max: 2.65ms, 5765.88MB    
Std: 0.04ms, 0.00MB    
Avg FPS: 411.87FPS    
Min FPS: 377.76FPS    
Max FPS: 452.74FPS    

# Explicit FP8 activations checkpointing for faster speed, no autoquant.
**Float8WeightOnlyConfig**
Mean: 2.55ms, 5625.50MB    
Min: 2.44ms, 5625.50MB    
Max: 2.80ms, 5625.50MB    
Std: 0.07ms, 0.00MB    
Avg FPS: 392.71FPS    
Min FPS: 357.34FPS    
Max FPS: 409.46FPS    

**Float8DynamicActivationFloat8WeightConfig**
*PerTensor*
Mean: 3.69ms, 3527.98MB    
Min: 3.49ms, 3527.98MB    
Max: 3.89ms, 3527.98MB    
Std: 0.03ms, 0.00MB    
Avg FPS: 271.30FPS    
Min FPS: 257.38FPS    
Max FPS: 286.78FPS    

*PerRow*
Mean: 4.32ms, 3530.16MB    
Min: 3.79ms, 3530.16MB    
Max: 4.50ms, 3530.16MB    
Std: 0.09ms, 0.00MB    
Avg FPS: 231.40FPS    
Min FPS: 222.16FPS    
Max FPS: 264.01FPS    