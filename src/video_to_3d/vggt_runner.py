from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VGGTRunConfig:
    vggt_root: Path
    checkpoint: Path
    image_resolution: int = 512
    device: str = "cuda"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["vggt_root"] = str(self.vggt_root)
        payload["checkpoint"] = str(self.checkpoint)
        return payload


def add_vggt_root(vggt_root: Path) -> None:
    root = vggt_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"VGGT-Omega root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def run_vggt_omega(image_paths: list[Path], config: VGGTRunConfig) -> dict:
    add_vggt_root(config.vggt_root)

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if config.device != "cuda":
        raise RuntimeError("VGGT-Omega 1B-512 inference in this project expects --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Use a CUDA-enabled PyTorch build that matches the installed driver."
        )
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    if not image_paths:
        raise ValueError("At least one image path is required")

    print(f"[vggt] loading checkpoint: {config.checkpoint}", flush=True)
    model = VGGTOmega().eval()
    state_dict = torch.load(str(config.checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(config.device)

    print(f"[vggt] loading {len(image_paths)} frames at image_resolution={config.image_resolution}", flush=True)
    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        image_resolution=config.image_resolution,
    ).to(config.device)
    print(f"[vggt] preprocessed tensor shape: {tuple(images.shape)}", flush=True)

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    return predictions


def cuda_environment() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": repr(exc)}

    payload = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(0),
            }
        )
    return payload

