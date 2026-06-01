#!/usr/bin/env python3
"""Extract and smoke-test the standalone HunyuanImage 3.0 VAE.

The official Hugging Face repos do not publish a separate VAE checkpoint file.
This script downloads only the safetensors shards that contain ``vae.*`` keys,
extracts those tensors into ``vae.safetensors``, then optionally runs a small
encode/decode test without loading the 80B main model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Standalone HunyuanImage 3.0 VAE test")
    parser.add_argument(
        "--repo-id",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Hugging Face repo to extract the VAE from.",
    )
    parser.add_argument("--revision", default="main", help="Repo revision.")
    parser.add_argument(
        "--output-dir",
        default="./HunyuanImage-3-VAE",
        help="Directory for vae_config.json and vae.safetensors.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private access.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate vae.safetensors even if it already exists.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep the downloaded source shards under output-dir/_hf_staging.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing output-dir/vae.safetensors and skip extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which shards contain VAE weights.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Only export the standalone VAE checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device like cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype used for the smoke test.",
    )
    parser.add_argument(
        "--size",
        default="128",
        help=(
            "Smoke-test size: an integer square size, WIDTHxHEIGHT, or auto "
            "to keep the input image size. Non-multiple-of-16 image inputs are padded."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path to reconstruct with the VAE.",
    )
    parser.add_argument(
        "--save-recon",
        default=None,
        help="Optional reconstruction output path. Defaults under output-dir.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup encode/decode iterations before timing.",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=1,
        help="Timed encode/decode iterations. Averages are reported separately.",
    )
    return parser.parse_args()


def hf_download(repo_id: str, filename: str, revision: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub[cli]") from exc

    kwargs = dict(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
    )
    if token:
        kwargs["token"] = token
    return Path(hf_hub_download(**kwargs))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def extract_vae_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_out = output_dir / "vae_config.json"
    ckpt_out = output_dir / "vae.safetensors"

    if args.skip_download:
        if not config_out.exists() or not ckpt_out.exists():
            raise FileNotFoundError(f"Missing {config_out} or {ckpt_out}")
        return config_out, ckpt_out

    if config_out.exists() and ckpt_out.exists() and not args.force and not args.dry_run:
        print(f"Found existing standalone VAE at {output_dir}. Use --force to rebuild it.")
        return config_out, ckpt_out

    staging_dir = output_dir / "_hf_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    config_path = hf_download(args.repo_id, "config.json", args.revision, staging_dir, args.token)
    index_path = hf_download(
        args.repo_id,
        "model.safetensors.index.json",
        args.revision,
        staging_dir,
        args.token,
    )

    model_config = read_json(config_path)
    index = read_json(index_path)
    weight_map = index.get("weight_map", {})
    vae_weight_map = {key: shard for key, shard in weight_map.items() if key.startswith("vae.")}
    if not vae_weight_map:
        raise RuntimeError(f"No vae.* tensors found in {args.repo_id}")

    shard_names = sorted(set(vae_weight_map.values()))
    print(f"VAE tensors: {len(vae_weight_map)}")
    print("VAE source shards:")
    for shard_name in shard_names:
        print(f"  {shard_name}")

    if args.dry_run:
        print("Dry run complete; no large checkpoint shards were downloaded.")
        return config_out, ckpt_out

    vae_config = model_config["vae"]
    write_json(config_out, vae_config)

    from safetensors import safe_open
    from safetensors.torch import save_file

    shard_paths = {
        shard_name: hf_download(args.repo_id, shard_name, args.revision, staging_dir, args.token)
        for shard_name in shard_names
    }

    vae_state = {}
    for shard_name, shard_path in shard_paths.items():
        expected_keys = [key for key, value in vae_weight_map.items() if value == shard_name]
        print(f"Extracting {len(expected_keys)} VAE tensors from {shard_name}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in expected_keys:
                vae_state[key.removeprefix("vae.")] = f.get_tensor(key)

    metadata = {
        "format": "pt",
        "source_repo": args.repo_id,
        "source_revision": args.revision,
        "source_shards": ",".join(shard_names),
        "key_prefix_removed": "vae.",
    }
    save_file(vae_state, str(ckpt_out), metadata=metadata)
    write_json(
        output_dir / "manifest.json",
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "source_shards": shard_names,
            "num_tensors": len(vae_state),
            "checkpoint": ckpt_out.name,
            "config": config_out.name,
        },
    )
    print(f"Wrote standalone VAE checkpoint: {ckpt_out}")

    if not args.keep_shards:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return config_out, ckpt_out


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_vae(config_path: Path, ckpt_path: Path, device, dtype):
    from safetensors.torch import load_file

    from hunyuan_image_3.autoencoder_kl_3d import AutoencoderKLConv3D

    vae_config = read_json(config_path)
    vae = AutoencoderKLConv3D.from_config(vae_config)
    state_dict = load_file(str(ckpt_path), device="cpu")
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VAE state mismatch. missing={missing}, unexpected={unexpected}")
    vae.to(device=device, dtype=dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def pad_size_to_multiple(size: tuple[int, int], multiple: int = 16) -> tuple[int, int]:
    width, height = size
    return ceil_to_multiple(width, multiple), ceil_to_multiple(height, multiple)


def parse_size(size_arg: str, image_path: str | None) -> tuple[int, int]:
    size_text = str(size_arg).lower().strip()
    if size_text == "auto":
        if not image_path:
            raise ValueError("--size auto requires --image")
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size

    if "x" in size_text:
        width_text, height_text = size_text.split("x", 1)
        width, height = int(width_text), int(height_text)
    else:
        width = height = int(size_text)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid --size {size_arg!r}")
    return width, height


def tensor_to_image(tensor, crop_size: tuple[int, int] | None = None):
    from PIL import Image

    if tensor.ndim == 5:
        if tensor.shape[2] != 1:
            raise ValueError(f"Expected a single temporal frame, got shape {tuple(tensor.shape)}")
        tensor = tensor[:, :, 0]
    tensor = tensor[0].detach().float().cpu().clamp(-1, 1)
    if crop_size:
        crop_width, crop_height = crop_size
        tensor = tensor[:, :crop_height, :crop_width]
    array = ((tensor + 1) * 127.5).round().clamp(0, 255).byte()
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def image_to_tensor(path: str, target_size: tuple[int, int], device, dtype):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        original_size = image.size
        image = image.convert("RGB")
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)

    target_width, target_height = target_size
    padded_width, padded_height = pad_size_to_multiple(target_size)
    if (padded_width, padded_height) != target_size:
        pad_width = padded_width - target_width
        pad_height = padded_height - target_height
        array = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")

    array = array / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    meta = {
        "original_size": original_size,
        "target_size": target_size,
        "padded_size": (padded_width, padded_height),
    }
    return tensor.to(device=device, dtype=dtype), meta


def synchronize_if_needed(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_ms(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f} ms"
    avg = sum(values) / len(values)
    return f"avg {avg:.2f} ms over {len(values)} runs; runs=" + ", ".join(f"{value:.2f}" for value in values)


def run_smoke_test(args: argparse.Namespace, config_path: Path, ckpt_path: Path) -> None:
    import torch

    if args.warmup_runs < 0 or args.timing_runs <= 0:
        raise ValueError("--warmup-runs must be >= 0 and --timing-runs must be > 0")

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    vae = load_vae(config_path, ckpt_path, device, dtype)
    target_size = parse_size(args.size, args.image)

    if args.image:
        sample, image_meta = image_to_tensor(args.image, target_size, device, dtype)
        print(f"Original image size: {image_meta['original_size'][0]}x{image_meta['original_size'][1]} (W x H)")
        print(f"Test target size:   {image_meta['target_size'][0]}x{image_meta['target_size'][1]} (W x H)")
        if image_meta["padded_size"] != image_meta["target_size"]:
            print(f"Padded VAE size:    {image_meta['padded_size'][0]}x{image_meta['padded_size'][1]} (W x H)")
    else:
        padded_size = pad_size_to_multiple(target_size)
        if padded_size != target_size:
            print(f"Requested size {target_size[0]}x{target_size[1]} padded to {padded_size[0]}x{padded_size[1]}")
        padded_width, padded_height = padded_size
        generator = torch.Generator(device=device).manual_seed(0)
        sample = torch.rand((1, 3, padded_height, padded_width), generator=generator, device=device, dtype=dtype)
        sample = sample * 2 - 1
        image_meta = {"target_size": target_size, "padded_size": padded_size}

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            posterior = vae.encode(sample).latent_dist
            latents = posterior.mode()
            recon = vae.decode(latents, return_dict=False)[0]
        synchronize_if_needed(device)

        encoder_times_ms = []
        decoder_times_ms = []
        for _ in range(args.timing_runs):
            synchronize_if_needed(device)
            start = time.perf_counter()
            posterior = vae.encode(sample).latent_dist
            latents = posterior.mode()
            synchronize_if_needed(device)
            encoder_times_ms.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            recon = vae.decode(latents, return_dict=False)[0]
            synchronize_if_needed(device)
            decoder_times_ms.append((time.perf_counter() - start) * 1000)

    print(f"Input shape:  {tuple(sample.shape)}")
    print(f"Latent shape: {tuple(latents.shape)}")
    print(f"Output shape: {tuple(recon.shape)}")
    print(f"Output range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
    print(f"Encoder time: {format_ms(encoder_times_ms)}")
    print(f"Decoder time: {format_ms(decoder_times_ms)}")

    if args.image:
        recon_path = Path(args.save_recon) if args.save_recon else Path(args.output_dir) / "vae_reconstruction.png"
        tensor_to_image(recon, crop_size=image_meta["target_size"]).save(recon_path)
        print(f"Saved reconstruction: {recon_path}")


def main() -> None:
    args = parse_args()
    config_path, ckpt_path = extract_vae_checkpoint(args)
    if args.dry_run or args.no_test:
        return
    run_smoke_test(args, config_path, ckpt_path)


if __name__ == "__main__":
    main()
