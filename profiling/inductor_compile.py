import os
import pickle
import torch
import torchao
from torchao.quantization.autoquant import (
    DEFAULT_AUTOQUANT_CLASS_LIST,  # int8 weights
    DEFAULT_INT4_AUTOQUANT_CLASS_LIST,  # mix of int8 and int4 weights
    DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,  # fp32, fp16, bf16 weights
    OTHER_AUTOQUANT_CLASS_LIST,  # fp8 weights
    ALL_AUTOQUANT_CLASS_LIST,
    AUTOQUANT_CACHE,
)
from torchao.quantization import (
    quantize_,
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig,
    PerTensor,
    PerRow,
)

from .profiler import profile_fn, print_results


def dump_autoquant_cache(path = "/tmp/autoquant_cache.pkl"):
    with open(path, "wb") as f:
        pickle.dump(AUTOQUANT_CACHE, f)


def reload_autoquant_cache(path = "/tmp/autoquant_cache.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            AUTOQUANT_CACHE.update(pickle.load(f))
    else:
        print(f"Autoquant cache not found at {path}")


def profile_torch_compile_inductor(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor

    compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile - WM")

    res_img = profile_fn(compiled_img_dec, dummy[0][0])
    print_results(res_img, "Torch Compile - IMG")

    res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    print_results(res_audio, "Torch Compile - AUDIO")


def profile_torch_compile_inductor_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor + FP8 with torchao

    reload_autoquant_cache()

    compiled_world_model = torchao.autoquant(
        torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True),
        example_input=dummy,
        qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
        set_inductor_config=False,
    )

    # compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    # compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    # compiled_img_dec = torchao.quantization.autoquant(
    #     compiled_img_dec,
    #     example_input=dummy[0][0],
    #     qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
    #     set_inductor_config=False
    # )

    # compiled_audio_dec = torchao.quantization.autoquant(
    #     compiled_audio_dec,
    #     example_input=dummy_pred_audio,
    #     qtensor_class_list=ALL_AUTOQUANT_CLASS_LIST,
    #     set_inductor_config=False
    # )

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile + TorchAO AutoQuant - WM")

    # res_img = profile_fn(compiled_img_dec, dummy[0][0])
    # print_results(res_img, "Torch Compile + TorchAO AutoQuant - IMG")

    # res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    # print_results(res_audio, "Torch Compile + TorchAO AutoQuant - AUDIO")

    dump_autoquant_cache()