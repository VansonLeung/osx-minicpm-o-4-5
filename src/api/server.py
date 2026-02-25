#!/usr/bin/env python3
"""OpenAI-compatible FastAPI server for MiniCPM-o-4_5-MLX-4bit."""

from __future__ import annotations

import os
import time
import uuid
import json
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import unquote, urlparse

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from huggingface_hub import snapshot_download

from minicpmo.utils import get_video_frame_audio_segments
from src.minicpm.video_chat_mlx import build_inputs, generate, sample_frames
from mlx_vlm import load


DEFAULT_MODEL_ID = os.getenv("MINICPM_MODEL", "andrevp/MiniCPM-o-4_5-MLX-4bit")
DEFAULT_MAX_FRAMES = int(os.getenv("MINICPM_MAX_FRAMES", "24"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MINICPM_MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.getenv("MINICPM_TEMPERATURE", "0.0"))
DEFAULT_MAX_SLICE_NUMS = int(os.getenv("MINICPM_MAX_SLICE_NUMS", "1"))
PRELOAD_ON_STARTUP = os.getenv("MINICPM_PRELOAD", "1") == "1"


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class InputVideoPart(BaseModel):
    type: Literal["input_video", "video", "video_url"]
    video_path: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None


ContentPart = Union[TextPart, InputVideoPart, Dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart], Dict[str, Any], None] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


app = FastAPI(title="MiniCPM OpenAI-Compatible API", version="1.0.0")

_model = None
_processor = None
_loaded_model_name = None


def _resolve_model_source(model_name: str) -> str:
    if os.path.exists(model_name):
        print(f"[model] using local path: {model_name}", flush=True)
        return model_name

    if "/" not in model_name:
        print(f"[model] treating as local model id/path: {model_name}", flush=True)
        return model_name

    print(f"[download] checking/downloading model: {model_name}", flush=True)
    started = time.time()
    local_path = snapshot_download(repo_id=model_name, resume_download=True)
    elapsed = time.time() - started
    print(f"[download] ready in {elapsed:.1f}s: {local_path}", flush=True)
    return local_path


def _ensure_model_loaded(model_name: str):
    global _model, _processor, _loaded_model_name
    if _model is None or _processor is None or _loaded_model_name != model_name:
        model_source = _resolve_model_source(model_name)
        print(f"[load] loading model into MLX: {model_name}", flush=True)
        started = time.time()
        _model, _processor = load(model_source, trust_remote_code=True)
        elapsed = time.time() - started
        _loaded_model_name = model_name
        print(f"[load] model ready in {elapsed:.1f}s", flush=True)
    return _model, _processor


@app.on_event("startup")
def _startup_preload() -> None:
    if PRELOAD_ON_STARTUP:
        _ensure_model_loaded(DEFAULT_MODEL_ID)


def _normalize_local_path(path_or_url: str) -> str:
    if path_or_url.startswith("file://"):
        parsed = urlparse(path_or_url)
        return unquote(parsed.path)
    return path_or_url


def _extract_prompt_and_video(messages: List[ChatMessage]) -> tuple[str, str]:
    prompt_parts: List[str] = []
    video_path: Optional[str] = None

    for msg in messages:
        if msg.role != "user":
            continue

        content = msg.content
        if isinstance(content, str):
            prompt_parts.append(content)
            continue

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                prompt_parts.append(text)
            raw_video = content.get("video_path") or content.get("path") or content.get("url")
            if isinstance(raw_video, str):
                video_path = raw_video
            continue

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ptype = (part.get("type") or "").lower()
                    if ptype == "text" and isinstance(part.get("text"), str):
                        prompt_parts.append(part["text"])
                    elif ptype in {"input_video", "video", "video_url"}:
                        raw_video = part.get("video_path") or part.get("path") or part.get("url")
                        if isinstance(raw_video, str):
                            video_path = raw_video

    prompt = "\n".join([x.strip() for x in prompt_parts if isinstance(x, str) and x.strip()])
    if not prompt:
        prompt = "Describe the video"

    if not video_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "No video provided. Add a user content part like "
                "{'type':'input_video','video_path':'/absolute/path/to/video.mp4'}"
            ),
        )

    video_path = _normalize_local_path(video_path)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")

    return prompt, video_path


def _run_chat(req: ChatCompletionRequest) -> tuple[str, int, int]:
    model_name = req.model or DEFAULT_MODEL_ID
    model, processor = _ensure_model_loaded(model_name)

    prompt, video_path = _extract_prompt_and_video(req.messages)

    frames, _, _ = get_video_frame_audio_segments(
        video_path,
        use_ffmpeg=True,
        adjust_audio_length=True,
    )
    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video.")

    sampled = sample_frames(frames, max_frames=DEFAULT_MAX_FRAMES)
    inputs = build_inputs(
        processor=processor,
        frames=sampled,
        prompt=prompt,
        max_slice_nums=DEFAULT_MAX_SLICE_NUMS,
    )

    max_new_tokens = req.max_new_tokens or req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = DEFAULT_TEMPERATURE if req.temperature is None else float(req.temperature)

    answer = generate(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        temp=temperature,
        enable_thinking=False,
    )

    prompt_tokens = int(inputs["input_ids"].size)
    completion_tokens = len(processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
    return answer, prompt_tokens, completion_tokens


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        answer, prompt_tokens, completion_tokens = _run_chat(req)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = req.model or DEFAULT_MODEL_ID
        created = int(time.time())

        def event_stream():
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    answer, prompt_tokens, completion_tokens = _run_chat(req)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model or DEFAULT_MODEL_ID,
        choices=[Choice(message=ChoiceMessage(content=answer))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response.model_dump()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "18800"))
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
