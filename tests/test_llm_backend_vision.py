"""The OpenAI-compatible backend beyond text: frames in, images out.

``LLM_VISION_MODEL`` lets the four frame-based stages (layout picker, hook
grounding, on-screen content, silent videos) run on a chat model that takes
``image_url`` parts; ``LLM_IMAGE_MODEL`` lets the thumbnail studio draw with
``/images/generations``. These tests pin the wire shapes and the routing:
which stage calls what, and that nothing changes when the variables are unset.
"""
import base64
import io
import json

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_VISION_MODEL", "LLM_IMAGE_MODEL", "LLM_IMAGE_SIZE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def vision(local, monkeypatch):
    monkeypatch.setenv("LLM_VISION_MODEL", "cx/gpt-5.6-sol")


def _serve(handler, monkeypatch):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1968, "completion_tokens": 13},
    })


# --- activation --------------------------------------------------------------

def test_vision_and_image_are_off_until_their_model_is_named(local, monkeypatch):
    assert llm_backend.active() is True
    assert llm_backend.vision_active() is False
    assert llm_backend.image_active() is False
    assert llm_backend.describe()["visionModel"] is None
    monkeypatch.setenv("LLM_VISION_MODEL", "v")
    monkeypatch.setenv("LLM_IMAGE_MODEL", "i")
    assert llm_backend.vision_active() and llm_backend.image_active()
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1",
        "visionModel": "v", "imageModel": "i"}


def test_vision_needs_the_base_url_too(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_VISION_MODEL", "v")
    assert llm_backend.vision_active() is False


# --- frames on the wire ------------------------------------------------------

def test_frames_travel_as_image_url_parts_on_the_vision_model(vision, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"layout": "screencast", "confidence": 0.9, "why": "spreadsheet"})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json(
        "pick a layout", gemini_worker.LayoutChoice, parts=["t=1.0s", b"\xff\xd8jpg1", b"\xff\xd8jpg2"])

    body = seen["body"]
    assert body["model"] == "cx/gpt-5.6-sol"  # frames mean the vision model, not LLM_MODEL
    content = body["messages"][1]["content"]
    assert [p["type"] for p in content] == ["text", "image_url", "image_url", "text"]
    assert content[0]["text"] == "t=1.0s"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpg1").decode()
    assert content[-1]["text"] == "pick a layout"  # prompt last, after the frames
    assert parsed["layout"] == "screencast"
    assert cost["model"] == "cx/gpt-5.6-sol" and cost["input_tokens"] == 1968


def test_text_only_prompt_stays_a_plain_string_on_the_text_model(vision, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("score", gemini_worker.ScoreResponse)
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["messages"][1]["content"] == "score"


def test_no_schema_skips_validation_and_json_object_mode_is_first(local, monkeypatch):
    formats = []

    def handler(request):
        formats.append((json.loads(request.content).get("response_format") or {}).get("type"))
        return _completion({"concepts": [{"scene": "x"}], "extra": 1})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("design", None)
    assert parsed == {"concepts": [{"scene": "x"}], "extra": 1}
    assert formats == ["json_object"]  # no schema, so no json_schema attempt


def test_generate_text_returns_the_message_and_rejects_empty(local, monkeypatch):
    _serve(lambda r: _completion("  A description\nwith chapters  "), monkeypatch)
    assert llm_backend.generate_text("describe") == "A description\nwith chapters"
    _serve(lambda r: _completion(""), monkeypatch)
    with pytest.raises(RuntimeError):
        llm_backend.generate_text("describe")


# --- images ------------------------------------------------------------------

def test_generate_image_posts_to_generations_and_decodes_b64(local, monkeypatch):
    monkeypatch.setenv("LLM_IMAGE_MODEL", "cx/gpt-image-2.5-flare")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"PNGBYTES").decode()}]})

    _serve(handler, monkeypatch)
    assert llm_backend.generate_image("a red circle") == b"PNGBYTES"
    assert seen["url"] == "http://llm.test/v1/images/generations"
    assert seen["body"] == {"model": "cx/gpt-image-2.5-flare", "prompt": "a red circle", "n": 1,
                            "size": "1792x1024", "response_format": "b64_json"}


def test_generate_image_drops_a_rejected_optional_field_and_retries(local, monkeypatch):
    """gpt-image models refuse response_format; DALL-E 2 refuses this size."""
    monkeypatch.setenv("LLM_IMAGE_MODEL", "gpt-image-1")
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "Unknown parameter: 'response_format'."}})
        if "size" in body:
            return httpx.Response(400, json={"error": {"message": "Invalid value for 'size'."}})
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"ok").decode()}]})

    _serve(handler, monkeypatch)
    assert llm_backend.generate_image("p") == b"ok"
    assert [sorted(b) for b in bodies] == [
        ["model", "n", "prompt", "response_format", "size"],
        ["model", "n", "prompt", "size"],
        ["model", "n", "prompt"]]


def test_generate_image_fetches_a_url_result(local, monkeypatch):
    monkeypatch.setenv("LLM_IMAGE_MODEL", "dall-e-3")

    def handler(request):
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"data": [{"url": "http://cdn.test/out.png"}]})
        assert str(request.url) == "http://cdn.test/out.png"
        return httpx.Response(200, content=b"FROMURL")

    _serve(handler, monkeypatch)
    assert llm_backend.generate_image("p") == b"FROMURL"


def test_references_go_multipart_to_edits_and_a_missing_endpoint_says_no_image(local, monkeypatch):
    monkeypatch.setenv("LLM_IMAGE_MODEL", "cx/gpt-image-2.5-flare")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type", "")
        seen["raw"] = request.read()
        return httpx.Response(404, text="<html>not found</html>")

    _serve(handler, monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_image("p", references=[b"\xff\xd8face"])
    assert seen["url"] == "http://llm.test/v1/images/edits"
    assert seen["content_type"].startswith("multipart/form-data")
    assert b'name="image[]"' in seen["raw"] and b"\xff\xd8face" in seen["raw"]
    assert "no image" in str(exc.value)  # the phrase thumbnail.py retries on, without the person


def test_generate_image_without_a_model_is_a_clear_error(local):
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_image("p")
    assert "LLM_IMAGE_MODEL" in str(exc.value)


# --- frame strips ------------------------------------------------------------

def test_frame_strip_interleaves_timestamps_and_explains_itself():
    frames = [(6.25, b"a"), (18.75, b"b")]
    assert llm_backend.frame_strip_parts(frames) == ["t=6.2s", b"a", "t=18.8s", b"b"]
    preface = llm_backend.frame_strip_preface(frames, 25.0)
    assert "2 frames" in preface and "25 seconds" in preface and "~12.5s" in preface


def test_timed_frames_reads_a_real_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 32))
    for i in range(50):  # 5 s at 10 fps
        writer.write(np.full((32, 64, 3), i * 5, dtype=np.uint8))
    writer.release()

    frames = llm_backend.timed_frames(path, n=4, width=160)
    assert [round(t, 1) for t, _ in frames] == [0.6, 1.8, 3.1, 4.3]
    assert all(jpg.startswith(b"\xff\xd8") for _, jpg in frames)


# --- who routes where --------------------------------------------------------

def test_layout_picker_uses_the_vision_model_without_a_gemini_key(vision, monkeypatch):
    import layout_picker
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    monkeypatch.setattr(layout_picker, "sample_frames", lambda *a, **k: [b"f1", b"f2"])
    seen = {}

    def fake(prompt, schema, model=None, parts=None):
        seen["parts"] = parts
        seen["schema"] = schema
        return {"layout": "split", "confidence": 0.8, "why": "two shot"}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake)
    assert layout_picker.pick("v.mp4", 60) == "split"
    assert seen["parts"] == [b"f1", b"f2"] and seen["schema"] is gemini_worker.LayoutChoice


def test_layout_picker_still_degrades_without_any_model(local, monkeypatch):
    import layout_picker
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(layout_picker, "ENABLED", True)
    assert layout_picker.pick("v.mp4", 60) == "none"  # text-only server: no frames sent


def test_hook_grounding_uses_the_vision_model_without_a_gemini_key(vision, monkeypatch):
    import hook_grounding as hg
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(hg, "frames_at", lambda path, times, width=None: [b"jpg"] * len(times))
    seen = {}

    def fake(prompt, schema, model=None, parts=None):
        seen["frames"] = parts
        return {"on_screen": "a dialog", "viral_hook_text": "new hook",
                "video_title_for_youtube_short": "new title"}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake)
    clip = {"viral_hook_text": "old", "layout_ranges": [{"start": 0, "end": 30, "layout": "screencast"}]}
    assert hg.reground("clip.mp4", clip, {"segments": []}, 0, 30)["on_screen"] == "a dialog"
    assert clip["viral_hook_text"] == "new hook" and len(seen["frames"]) == 3


def test_on_screen_content_reads_a_frame_strip_on_the_vision_model(vision, monkeypatch):
    import screencast_layout
    monkeypatch.setattr(screencast_layout, "ENABLED", True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_backend, "timed_frames", lambda *a, **k: [(5.0, b"a"), (15.0, b"b")])
    seen = {}

    def fake(prompt, schema, model=None, parts=None):
        seen["prompt"] = prompt
        seen["parts"] = parts
        return {"ranges": [{"start": 4, "end": 16, "what": "spreadsheet", "width_fraction": 0.95},
                           {"start": 0, "end": 20, "what": "logo", "width_fraction": 0.1}]}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake)
    ranges = screencast_layout.detect_content_ranges("v.mp4", 20.0)
    assert ranges == [(4.0, 16.0, "spreadsheet", 0.95)]  # the width gate still applies
    assert seen["parts"] == ["t=5.0s", b"a", "t=15.0s", b"b"]
    assert seen["prompt"].startswith("The video is NOT attached as a file")
    assert "width_fraction" in seen["prompt"]


def test_silent_video_pick_reads_a_frame_strip_on_the_vision_model(vision, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_backend, "timed_frames", lambda *a, **k: [(10.0, b"a"), (30.0, b"b")])
    seen = {}

    def fake(prompt, schema, model=None, parts=None):
        seen["prompt"] = prompt
        seen["schema"] = schema
        return {"shorts": [{"start": 5, "end": 40, "viral_hook_text": "h"},
                           {"start": 30, "end": 500, "viral_hook_text": "clamped"},
                           {"start": 39.5, "end": 40, "viral_hook_text": "dropped"}]}, {"local": True}

    monkeypatch.setattr(llm_backend, "generate_json", fake)
    result = main.get_visual_clips("v.mp4", 40.0)
    assert [(s["start"], s["end"]) for s in result["shorts"]] == [(5.0, 40.0), (30.0, 40.0)]
    assert result["cost_analysis"] == {"local": True}
    assert seen["schema"] is gemini_worker.VisualResponse
    assert "2 frames" in seen["prompt"] and "NO speech" in seen["prompt"]


def test_thumbnail_text_and_images_route_to_the_local_server(local, monkeypatch, tmp_path):
    thumbnail = pytest.importorskip("thumbnail")
    from PIL import Image
    monkeypatch.setenv("LLM_IMAGE_MODEL", "cx/gpt-image-2.5-flare")
    monkeypatch.chdir(tmp_path)
    calls = {"json": [], "image": []}

    def fake_json(prompt, schema, model=None, parts=None):
        calls["json"].append(parts)
        return {"concepts": [{"text": "WOW", "text_position": "left", "text_color": "yellow",
                              "scene": "a laptop", "why": "w"}]}, None

    def fake_image(prompt, references=(), model=None, size=None):
        calls["image"].append((prompt, list(references)))
        buf = io.BytesIO()
        Image.new("RGB", (1672, 941), "blue").save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr(llm_backend, "generate_json", fake_json)
    monkeypatch.setattr(llm_backend, "generate_image", fake_image)

    out = thumbnail.generate_thumbnail(None, "My title", "sess", count=1, burn_text=True)
    assert out[0]["text"] == "WOW" and out[0]["fallback"] is False
    assert calls["json"] == [None]  # concept design: text only
    prompt, refs = calls["image"][0]
    assert "a laptop" in prompt and "Do NOT render any text" in prompt and refs == []
    saved = Image.open(tmp_path / "output" / "thumbnails" / "sess" / out[0]["url"].rsplit("/", 1)[1])
    assert saved.size == (1280, 720)


def test_thumbnail_reference_falls_back_without_the_person_when_edits_is_missing(local, monkeypatch, tmp_path):
    thumbnail = pytest.importorskip("thumbnail")
    from PIL import Image
    monkeypatch.setenv("LLM_IMAGE_MODEL", "cx/gpt-image-2.5-flare")
    monkeypatch.chdir(tmp_path)
    face = tmp_path / "face.jpg"
    Image.new("RGB", (64, 64), "red").save(face)
    attempts = []

    def fake_image(prompt, references=(), model=None, size=None):
        attempts.append(len(references))
        if references:
            raise RuntimeError("image server 404 on images/edits (no image): not found")
        buf = io.BytesIO()
        Image.new("RGB", (1280, 720), "green").save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr(llm_backend, "generate_json",
                        lambda *a, **k: ({"concepts": [{"scene": "s", "text": "T"}]}, None))
    monkeypatch.setattr(llm_backend, "generate_image", fake_image)
    out = thumbnail.generate_thumbnail(None, "t", "sess", face_image_path=str(face), count=1)
    assert attempts == [1, 0] and out[0]["fallback"] is True


def test_thumbnail_titles_send_frames_only_with_a_vision_model(local, monkeypatch):
    thumbnail = pytest.importorskip("thumbnail")
    seen = []
    monkeypatch.setattr(llm_backend, "generate_json",
                        lambda prompt, schema, model=None, parts=None: (seen.append(parts) or ({"titles": ["a"]}, None)))
    assert thumbnail._ask_json(None, "p", [b"f"])[0] == {"titles": ["a"]}
    assert seen == [None]  # text-only server: frames dropped, transcript still answers
    monkeypatch.setenv("LLM_VISION_MODEL", "v")
    thumbnail._ask_json(None, "p", [b"f"])
    assert seen[-1] == [b"f"]


def test_thumbnail_without_any_text_model_is_a_clear_error(monkeypatch):
    thumbnail = pytest.importorskip("thumbnail")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError) as exc:
        thumbnail._gemini_client(None)
    assert "GEMINI_API_KEY or LLM_BASE_URL" in str(exc.value)
