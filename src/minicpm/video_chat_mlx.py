#!/usr/bin/env python3
"""Video chat with MiniCPM-o-4_5-MLX-4bit on macOS (Apple Silicon)."""

import argparse
import re
import time
from typing import List

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from minicpmo.utils import get_video_frame_audio_segments
from mlx_vlm import load
from mlx_vlm.generate import generate_step

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def to_pil_image(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            return Image.fromarray(frame, mode="L").convert("RGB")
        return Image.fromarray(frame).convert("RGB")
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def sample_frames(frames: List, max_frames: int) -> List[Image.Image]:
    if not frames:
        return []
    pil_frames = [to_pil_image(f) for f in frames]
    if len(pil_frames) <= max_frames:
        return pil_frames

    idx = np.linspace(0, len(pil_frames) - 1, num=max_frames, dtype=int)
    return [pil_frames[i] for i in idx]


def build_inputs(processor, frames: List[Image.Image], prompt: str, max_slice_nums: int):
    image_placeholders = "\n".join(["<image>./</image>"] * len(frames))
    text = (
        f"<|im_start|>user\n{image_placeholders}\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    inputs = processor(
        text=text,
        images=frames,
        max_slice_nums=max_slice_nums,
    )

    input_ids = mx.array(inputs["input_ids"].numpy())
    mask = mx.array(inputs["attention_mask"].numpy())

    pixel_values_list = inputs["pixel_values"]
    tgt_sizes_list = inputs["tgt_sizes"]
    image_bound = inputs["image_bound"]

    all_pv = []
    for batch_pvs in pixel_values_list:
        for pv in batch_pvs:
            pv_np = np.transpose(pv.numpy(), (1, 2, 0))
            all_pv.append(pv_np)

    if not all_pv:
        raise RuntimeError("No visual tokens produced. Check video/frame extraction.")

    max_h = max(p.shape[0] for p in all_pv)
    max_w = max(p.shape[1] for p in all_pv)
    padded = []
    for p in all_pv:
        pad_h = max_h - p.shape[0]
        pad_w = max_w - p.shape[1]
        if pad_h > 0 or pad_w > 0:
            p = np.pad(p, ((0, pad_h), (0, pad_w), (0, 0)))
        padded.append(p)

    pixel_values = mx.array(np.stack(padded, axis=0))

    patch_attention_mask = None
    if pixel_values is not None:
        bsz = pixel_values.shape[0]
        total_patches = (pixel_values.shape[1] // 14) * (pixel_values.shape[2] // 14)
        patch_attention_mask_np = np.zeros((bsz, total_patches), dtype=bool)
        offset = 0
        for ts_batch in tgt_sizes_list:
            if isinstance(ts_batch, torch.Tensor):
                for j in range(ts_batch.shape[0]):
                    idx = offset + j
                    if idx < bsz:
                        h, w = int(ts_batch[j][0]), int(ts_batch[j][1])
                        patch_attention_mask_np[idx, : h * w] = True
                offset += ts_batch.shape[0]
        patch_attention_mask = mx.array(patch_attention_mask_np)

    tgt_sizes = []
    for ts_batch in tgt_sizes_list:
        if isinstance(ts_batch, torch.Tensor):
            tgt_sizes.append(ts_batch.numpy())
    tgt_sizes = mx.array(np.concatenate(tgt_sizes, axis=0).astype(np.int32)) if tgt_sizes else None

    bounds = []
    for batch_bounds in image_bound:
        if isinstance(batch_bounds, torch.Tensor) and batch_bounds.numel() > 0:
            bounds.extend(batch_bounds.numpy().tolist())
        elif isinstance(batch_bounds, list):
            bounds.extend(batch_bounds)

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "mask": mask,
        "tgt_sizes": tgt_sizes,
        "image_bound": bounds,
        "patch_attention_mask": patch_attention_mask,
    }


def generate(model, processor, inputs, max_new_tokens: int, temp: float, enable_thinking: bool) -> str:
    kwargs = {}
    if inputs.get("tgt_sizes") is not None:
        kwargs["tgt_sizes"] = inputs["tgt_sizes"]
    if inputs.get("image_bound"):
        kwargs["image_bound"] = inputs["image_bound"]
    if inputs.get("patch_attention_mask") is not None:
        kwargs["patch_attention_mask"] = inputs["patch_attention_mask"]

    tokens = []
    start = time.time()

    for n, (token, _logprobs) in enumerate(
        generate_step(
            inputs["input_ids"],
            model,
            inputs["pixel_values"],
            inputs["mask"],
            temp=temp,
            **kwargs,
        )
    ):
        tok_val = token.item() if hasattr(token, "item") else int(token)
        tokens.append(tok_val)

        tok_str = processor.tokenizer.decode([tok_val])
        if tok_str in ["<|im_end|>", "<|endoftext|>", "<|tts_eos|>"]:
            break
        if n + 1 >= max_new_tokens:
            break

    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    if not enable_thinking:
        text = THINK_RE.sub("", text)

    elapsed = time.time() - start
    print(f"generated {len(tokens)} tokens in {elapsed:.2f}s")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with video using MiniCPM-o-4_5-MLX-4bit")
    parser.add_argument("--model", default="andrevp/MiniCPM-o-4_5-MLX-4bit", help="HF model ID or local path")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--prompt", default="Describe the video", help="Question about the video")
    parser.add_argument("--max-frames", type=int, default=24, help="Max frames sampled from video")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load(args.model, trust_remote_code=True)

    print("Extracting video frames...")
    video_frames, _, _ = get_video_frame_audio_segments(args.video, use_ffmpeg=True, adjust_audio_length=True)
    print("num frames:", len(video_frames))

    sampled_frames = sample_frames(video_frames, max_frames=args.max_frames)
    print("sampled frames:", len(sampled_frames))

    if not sampled_frames:
        raise RuntimeError("No frames extracted from video.")

    inputs = build_inputs(
        processor=processor,
        frames=sampled_frames,
        prompt=args.prompt,
        max_slice_nums=args.max_slice_nums,
    )

    answer = generate(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        temp=args.temperature,
        enable_thinking=args.enable_thinking,
    )
    print("\nanswer:\n", answer)


if __name__ == "__main__":
    main()
