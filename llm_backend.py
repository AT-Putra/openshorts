"""LLM backend for the stages that used to be Gemini-only: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI, OpenRouter and routers
such as 9router all speak ``POST {base}/chat/completions``. When
``LLM_BASE_URL`` is set (or ``LLM_PROVIDER=openai``), the transcript scoring
and detail passes in ``main.get_viral_clips`` go here instead of Gemini, so a
self-hosted install can run the whole pipeline without a Google key.

Three capabilities, each switched on by its own variable, because a text
model that cannot see frames must never be handed frames:

* ``LLM_MODEL`` — text. The moment picker, thumbnail titles / concepts /
  description.
* ``LLM_VISION_MODEL`` — a chat model that accepts ``image_url`` parts. The
  layout picker (``layout_picker.py``), hook grounding
  (``hook_grounding.py``), the on-screen content detector
  (``screencast_layout.py``) and the silent-video path
  (``main.get_visual_clips``). The last two send Gemini the video file; an
  OpenAI-compatible chat endpoint takes no video, so they get a strip of
  timestamped frames instead (``timed_frames``): lower fidelity than watching
  the footage, but it runs.
* ``LLM_IMAGE_MODEL`` — ``POST {base}/images/generations`` for the thumbnail
  images. ``/images/edits`` (reference photos) is optional: a server without
  it answers 4xx and the caller falls back to a generation with no reference.

Unset, each stage keeps its old behaviour: Gemini when a key is present,
degrade otherwise. Nothing here is wired in cloud mode (``BILLING_ENABLED``).

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import base64
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Type, Union

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes
# gpt-image / DALL-E landscape size. The thumbnail is cover-cropped to 1280x720
# afterwards, so anything near 16:9 is fine; a server that rejects the value
# gets the request again without it.
DEFAULT_IMAGE_SIZE = "1792x1024"
# Frame strip for the two stages that send Gemini the whole video. 48 frames
# at 640px is ~12k tokens on a gpt-4o-class model, whatever the source runs to.
DEFAULT_VISION_MAX_FRAMES = 48
DEFAULT_VISION_FRAME_WIDTH = 640

# A content part: text, or JPEG bytes (rendered as an image_url data URI).
Part = Union[str, bytes]


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def vision_model_name() -> str:
    return (os.environ.get("LLM_VISION_MODEL") or "").strip()


def image_model_name() -> str:
    return (os.environ.get("LLM_IMAGE_MODEL") or "").strip()


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def vision_active() -> bool:
    """True when the frame-based stages should send frames to the server."""
    return active() and bool(vision_model_name())


def image_active() -> bool:
    """True when thumbnail images should come from ``/images/generations``."""
    return active() and bool(image_model_name())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url(),
            "visionModel": vision_model_name() or None,
            "imageModel": image_model_name() or None}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Optional[Type[BaseModel]]):
    if schema is not None:
        name = getattr(schema, "__name__", "response").lower()
        yield {"type": "json_schema",
               "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def _data_uri(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _user_content(prompt: str, parts: Optional[Sequence[Part]]):
    """The user message: a plain string when there is nothing to show, else
    OpenAI content parts with every ``bytes`` entry as an image_url data URI.
    The prompt goes LAST, after the frames, like the Gemini calls do."""
    if not parts:
        return prompt
    content = []
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            content.append({"type": "image_url", "image_url": {"url": _data_uri(bytes(p))}})
        else:
            content.append({"type": "text", "text": str(p)})
    content.append({"type": "text", "text": prompt})
    return content


def _message_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    if isinstance(text, list):  # some servers return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return text


def _cost(data: dict, model: str) -> dict:
    usage = data.get("usage") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }


def _chat(messages: list, model: str, schema: Optional[Type[BaseModel]],
          json_mode: bool) -> dict:
    """One ``/chat/completions`` call, walking the response_format ladder when
    JSON is wanted. Returns the decoded response body."""
    url = f"{base_url()}/chat/completions"
    formats = list(_response_formats(schema)) if json_mode else [None]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in formats:
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            return resp.json()
    raise RuntimeError(
        f"LLM server rejected every response_format variant: {last_rejection}")


def generate_json(prompt: str, schema: Optional[Type[BaseModel]], model: Optional[str] = None,
                  parts: Optional[Sequence[Part]] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    ``parts`` are shown before the prompt: JPEG ``bytes`` become images,
    strings become text (a timestamp label next to its frame). Any frame means
    the vision model, which the caller must have checked with
    ``vision_active()``. ``schema=None`` skips validation for prompts whose
    JSON shape is described only in prose (the thumbnail studio).

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    has_images = any(isinstance(p, (bytes, bytearray)) for p in (parts or ()))
    model = model or (vision_model_name() if has_images else model_name())
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": _user_content(prompt, parts)},
    ]
    data = _chat(messages, model, schema, json_mode=True)
    parsed = gemini_worker._parse_json_response_text(_message_text(data))
    if schema is not None:
        # Validate against the same schema Gemini enforces server-side, so a
        # local model that drops a field fails here with a readable error
        # (retried by the caller) instead of deep inside the clip pipeline.
        parsed = schema.model_validate(parsed).model_dump()
    return parsed, _cost(data, model)


def generate_text(prompt: str, model: Optional[str] = None,
                  parts: Optional[Sequence[Part]] = None) -> str:
    """One chat completion returned as plain text (the YouTube description)."""
    has_images = any(isinstance(p, (bytes, bytearray)) for p in (parts or ()))
    model = model or (vision_model_name() if has_images else model_name())
    messages = [{"role": "user", "content": _user_content(prompt, parts)}]
    data = _chat(messages, model, None, json_mode=False)
    text = _message_text(data).strip()
    if not text:
        raise RuntimeError("LLM server returned an empty message")
    return text


# --- images ------------------------------------------------------------------

def image_size() -> str:
    return (os.environ.get("LLM_IMAGE_SIZE") or "").strip() or DEFAULT_IMAGE_SIZE


def _rejected_field(resp: httpx.Response, fields: Iterable[str]) -> Optional[str]:
    """Which optional field a 400/422 complains about, if any."""
    if resp.status_code not in (400, 422):
        return None
    body = resp.text.lower()
    for f in fields:
        if f in body:
            return f
    return None


def _image_bytes(data: dict) -> bytes:
    """The first image of an Images API response, whether it came inline or
    as a URL (DALL-E's default)."""
    items = data.get("data") or []
    if not items:
        raise RuntimeError("image server returned no image")
    item = items[0] or {}
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = item.get("url") or ""
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url:
        with _client() as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                raise RuntimeError(f"image server {resp.status_code} fetching {url[:80]}")
            return resp.content
    raise RuntimeError("image server returned neither b64_json nor url")


def generate_image(prompt: str, references: Sequence[bytes] = (), model: Optional[str] = None,
                   size: Optional[str] = None) -> bytes:
    """One image from ``/images/generations``, or ``/images/edits`` when
    reference photos are given. Returns the encoded image bytes (PNG/JPEG/WebP,
    whatever the server sends; PIL sniffs it).

    ``size`` and ``response_format`` are optional in the sense that a server
    which rejects one gets the request again without it: gpt-image models
    refuse ``response_format`` (always inline), DALL-E defaults to a URL, and
    some routers know neither. A server without ``/images/edits`` raises with
    "no image" in the message, which is the phrase the thumbnail fallback
    keys on to retry without the person.
    """
    model = model or image_model_name()
    if not model:
        raise RuntimeError("LLM_IMAGE_MODEL is not set")
    size = size or image_size()
    optional = {"size": size, "response_format": "b64_json"}
    endpoint = "images/edits" if references else "images/generations"
    url = f"{base_url()}/{endpoint}"
    headers = {k: v for k, v in _headers().items() if k != "Content-Type"}

    with _client() as client:
        for _ in range(3):
            if references:
                files = [("image[]", (f"reference_{i}.jpg", ref, "image/jpeg"))
                         for i, ref in enumerate(references)]
                data = {"model": model, "prompt": prompt, "n": "1", **optional}
                resp = client.post(url, data=data, files=files, headers=headers)
            else:
                body = {"model": model, "prompt": prompt, "n": 1, **optional}
                resp = client.post(url, json=body, headers=headers)
            dropped = _rejected_field(resp, list(optional))
            if dropped:
                optional.pop(dropped)
                continue
            if resp.status_code >= 400:
                detail = resp.text[:300]
                if references:
                    raise RuntimeError(
                        f"image server {resp.status_code} on {endpoint} (no image): {detail}")
                raise RuntimeError(f"image server {resp.status_code} from {url}: {detail}")
            return _image_bytes(resp.json())
    raise RuntimeError("image server kept rejecting the request")


# --- frame strips ------------------------------------------------------------

def vision_max_frames() -> int:
    try:
        return max(4, int(os.environ.get("LLM_VISION_MAX_FRAMES") or DEFAULT_VISION_MAX_FRAMES))
    except ValueError:
        return DEFAULT_VISION_MAX_FRAMES


def vision_frame_width() -> int:
    try:
        return max(160, int(os.environ.get("LLM_VISION_FRAME_WIDTH") or DEFAULT_VISION_FRAME_WIDTH))
    except ValueError:
        return DEFAULT_VISION_FRAME_WIDTH


def timed_frames(video_path: str, n: Optional[int] = None,
                 width: Optional[int] = None) -> List[Tuple[float, bytes]]:
    """``(seconds, jpeg)`` for ``n`` frames spread evenly over the video, the
    stand-in for "watch the file" on a server that only takes images. Frames
    sit at the middle of each slot, so a 10-minute video with 48 frames shows
    one every 12.5 s starting at 6.25 s."""
    import cv2

    n = n or vision_max_frames()
    width = width or vision_frame_width()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out: List[Tuple[float, bytes]] = []
    try:
        if total <= 0:
            return out
        n = min(n, total)
        for i in range(n):
            idx = int((i + 0.5) * total / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / w))),
                                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append((idx / fps, buf.tobytes()))
    finally:
        cap.release()
    return out


def frame_strip_parts(frames: Sequence[Tuple[float, bytes]]) -> List[Part]:
    """Interleave ``t=12.5s`` labels with their frames, for ``generate_json``."""
    parts: List[Part] = []
    for t, jpg in frames:
        parts.append(f"t={t:.1f}s")
        parts.append(jpg)
    return parts


def frame_strip_preface(frames: Sequence[Tuple[float, bytes]], video_duration: float) -> str:
    """The sentence that turns a "watch this video" prompt into a "read these
    frames" prompt. Prepended to the Gemini prompt text, which is otherwise
    reused verbatim."""
    step = (video_duration / len(frames)) if frames else 0.0
    return (f"The video is NOT attached as a file. Instead it is given as {len(frames)} "
            f"frames sampled evenly across its {video_duration:.0f} seconds (one every "
            f"~{step:.1f}s); each frame is preceded by its timestamp as 't=<seconds>s'. "
            "Read every timestamp: all times you report must come from those labels, "
            "interpolating between neighbouring frames where a change falls between two.\n\n")
