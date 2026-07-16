"""RRWNet / CMRRWNet model + pretrained-weight loading.

Imports the architecture directly from the vendored external/rrwnet and
external/cmrrwnet submodules rather than re-implementing it (both MIT
licensed). `second_u` (the recursive refinement module) has tied weights
across iterations, so `iterations` is just a runtime loop count -- it doesn't
affect the state_dict shape, and pretrained weights load regardless of the
iteration count used at train/inference time.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "external" / "rrwnet"))
sys.path.insert(0, str(_REPO_ROOT / "external" / "cmrrwnet"))

from model import RRWNet  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_cmrrwnet_spec = _ilu.spec_from_file_location(
    "cmrrwnet_model", _REPO_ROOT / "external" / "cmrrwnet" / "model.py"
)
_cmrrwnet_model = _ilu.module_from_spec(_cmrrwnet_spec)
_cmrrwnet_spec.loader.exec_module(_cmrrwnet_model)
CMRRWNet = _cmrrwnet_model.CMRRWNet


def load_hrf_pretrained(model: RRWNet, strict: bool = False) -> RRWNet:
    """Warm-start a 3-channel-input RRWNet from j-morano/rrwnet-hrf weights."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download("j-morano/rrwnet-hrf", "model.safetensors")
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    # ConvBlock registers conv1/conv2 both directly and via nn.Sequential
    # (self.conv_block), so state_dict() lists the same tensor under two
    # names -- "missing" entries under *.conv_block.{0,2}.* are harmless
    # aliases of the already-loaded *.conv1/*.conv2, not unloaded weights.
    real_missing = [k for k in missing if ".conv_block." not in k]
    if real_missing or unexpected:
        print(f"[load_hrf_pretrained] missing={real_missing} unexpected={unexpected}")
    return model


def build_model(task: str, base_ch: int = 64, iterations: int = 5, pretrained: bool = True):
    """task: 'task1' (3ch CFP) or 'task2' (5ch CFP+FFA)."""
    if task == "task1":
        model = RRWNet(input_ch=3, output_ch=3, base_ch=base_ch, iterations=iterations)
        if pretrained:
            model = load_hrf_pretrained(model)
        return model
    elif task == "task2":
        model = CMRRWNet(input_ch=5, output_ch=3, base_ch=base_ch, iterations=iterations)
        if pretrained:
            # Only first_u.conv*_rgb branch shapes could partially match a
            # 3ch RRWNet checkpoint; safer to warm-start from a trained Task1
            # checkpoint's encoder explicitly (see scripts) rather than HRF
            # weights directly, since CMRRWNet's first_u has a different
            # architecture (NewUNetModule, 3-branch) than RRWNet's first_u.
            pass
        return model
    else:
        raise ValueError(f"Unknown task: {task}")
