from unittest.mock import patch

import pipeline.ocr as ocr


def test_remote_is_off_by_default(monkeypatch):
    monkeypatch.delenv("OCR_ALLOW_REMOTE", raising=False)
    assert "OCR_ALLOW_REMOTE" in ocr._validate_vlm_config()
    assert ocr.get_ocr_status()["engine"] == "local"


def test_remote_client_refused_without_opt_in(monkeypatch):
    monkeypatch.delenv("OCR_ALLOW_REMOTE", raising=False)
    with patch("pipeline.ocr.create_openai_client") as client:
        try:
            ocr.extract_text_with_vlm("x")
        except ValueError:
            pass
        client.assert_not_called()


def test_loop_filter_drops_digit_loops_but_keeps_dot_leaders():
    assert ocr._is_loop("۱۶۲۴۳۵۲۶۳" * 3)
    assert not ocr._is_loop("....۱۲۰۰ = ۱۴۰.... ÷ اس")


def test_ensemble_env_switch(monkeypatch):
    monkeypatch.delenv("OCR_ENSEMBLE", raising=False)
    assert ocr._ensemble_engines() == []
    monkeypatch.setenv("OCR_ENSEMBLE", "1")
    assert "qwen3vl-fa" in ocr._ensemble_engines()
    monkeypatch.setenv("OCR_ENSEMBLE", "a, b")
    assert ocr._ensemble_engines() == ["a", "b"]


def test_fuse_lines_votes_and_drops_loops_and_empties():
    runs = [["من به خانه رفتم", "", "x"], ["من به خانه رفتم", "سلام", "x"], ["من به حانه رفتم", "۱۶۲۴۳۵۲۶۳" * 3, ""]]
    out = ocr._fuse_lines(runs)
    assert out[0] == "من به خانه رفتم"
    assert out[1] == "سلام"  # empty and looping candidates are ignored


def test_ensemble_survives_one_failing_engine(monkeypatch):
    def fake(engine, crops):
        if engine == "bad":
            raise RuntimeError("boom")
        return ["متن" for _ in crops]
    monkeypatch.setattr(ocr, "_read_all", fake)
    assert ocr._ensemble_lines([object(), object()], ["bad", "good"]) == ["متن", "متن"]


def test_junk_filter_counting_runs_and_prompt_echo():
    assert ocr._is_loop("۱۲۳۴۵۶۷۸۹۱۰۱۱۱۲")
    assert ocr._is_loop("متن فارسی داخل این تصویر را عیناً رونویسی کن.")
    assert not ocr._is_loop("۱۲۰ نفر در کلاس بودند")


def test_small_line_crops_are_scaled_to_trained_height():
    from PIL import Image
    out = ocr._fit_line_height(Image.new("RGB", (300, 20), "white"))
    assert out.height == 96 and out.width == 1440 // 1  # x4.8, aspect kept
    big = Image.new("RGB", (300, 100), "white")
    assert ocr._fit_line_height(big) is big


def test_literal_empty_string_answer_is_junk():
    assert ocr._is_loop("رشته خالی")


def test_big_handwriting_figure_is_rescued_as_text(monkeypatch):
    import numpy as np
    from PIL import Image
    wide = [[0, i * 40, 600, i * 40 + 30] for i in range(5)]
    monkeypatch.setattr(ocr, "_line_boxes", lambda region: wide)
    assert ocr._looks_like_text(Image.new("RGB", (600, 300)))
    monkeypatch.setattr(ocr, "_line_boxes", lambda region: [[0, 0, 100, 90], [0, 100, 90, 190]])
    assert not ocr._looks_like_text(Image.new("RGB", (600, 300)))
    # end to end through run_ocr: a large "figure" with line-shaped content gets OCR'd
    monkeypatch.setattr(ocr, "_line_boxes", lambda region: wide)
    monkeypatch.setattr(ocr, "recognize_line", lambda crop: "متن")
    monkeypatch.delenv("OCR_ENSEMBLE", raising=False)
    block = {"type": "figure", "bbox": [0, 0, 600, 400], "confidence": 0.9}
    out = ocr.run_ocr(np.full((500, 700, 3), 255, np.uint8), [block])
    assert out[0]["type"] == "text" and out[0]["text"] == "\n".join(["متن"] * 5)


