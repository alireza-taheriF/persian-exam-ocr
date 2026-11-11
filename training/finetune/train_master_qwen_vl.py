#!/usr/bin/env python3
"""
Master Persian Handwriting QLoRA Fine-Tuning Pipeline for Google Colab
Trains Qwen2.5-VL-3B-Instruct or Qwen2.5-VL-7B-Instruct on >55,000 Persian handwriting lines
Targeting >90% Word/Character Accuracy on unruled and ruled academic sheets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

PUNCT = re.compile(r"[.،,؛;:!؟?«»()\[\]{}\-–—\"'…]")


def normalize_persian_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9").replace("\u200c", " ")
    return re.sub(r"\s+", " ", t).strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_metrics(refs: list[str], preds: list[str]) -> dict:
    total_ed, total_len = 0, 0
    total_word_ed, total_word_len = 0, 0
    for r, p in zip(refs, preds):
        r_clean = normalize_persian_text(r)
        p_clean = normalize_persian_text(p)
        total_ed += levenshtein_distance(r_clean, p_clean)
        total_len += max(1, len(r_clean))

        rw = r_clean.split()
        pw = p_clean.split()
        dp = [[0] * (len(pw) + 1) for _ in range(len(rw) + 1)]
        for i in range(len(rw) + 1):
            dp[i][0] = i
        for j in range(len(pw) + 1):
            dp[0][j] = j
        for i in range(1, len(rw) + 1):
            for j in range(1, len(pw) + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (0 if rw[i - 1] == pw[j - 1] else 1))
        total_word_ed += dp[len(rw)][len(pw)]
        total_word_len += max(1, len(rw))

    cer = total_ed / max(1, total_len)
    wer = total_word_ed / max(1, total_word_len)
    return {
        "cer": round(cer, 4),
        "char_acc": round((1.0 - cer) * 100, 2),
        "wer": round(wer, 4),
        "word_acc": round((1.0 - wer) * 100, 2),
    }


def prepare_image(img: Image.Image, max_px: int = 150_000) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (64, 64), (255, 255, 255))
    if w < 32 or h < 32:
        new_w = max(w, 32)
        new_h = max(h, 32)
        padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        padded.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
        img = padded
    if img.width * img.height > max_px:
        k = (max_px / (img.width * img.height)) ** 0.5
        nw = max(32, round(img.width * k))
        nh = max(32, round(img.height * k))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img


class PersianHTRDataset(Dataset):
    def __init__(self, data_list: list[dict], root_dir: Path, processor, prompt: str, max_pixels: int = 150_000):
        self.data = data_list
        self.root = root_dir
        self.proc = processor
        self.prompt = prompt
        self.max_px = max_pixels

    def __len__(self):
        return len(self.data)

    def _fit_image(self, img: Image.Image) -> Image.Image:
        return prepare_image(img, self.max_px)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        img_p = self.root / row["image_path"] if "image_path" in row else self.root / row["image"]
        try:
            img = Image.open(img_p).convert("RGB")
            img = self._fit_image(img)
        except Exception:
            img = Image.new("RGB", (128, 64), (255, 255, 255))

        answer = str(row.get("text", "")).strip()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        return {"img": img, "messages": messages, "answer": answer}


def train_colab(args):
    print("=" * 60)
    print(f"🚀 Master Persian HTR Fine-Tuning | Model: {args.model_id}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("⚠️ Warning: CUDA GPU not detected! Make sure you selected a GPU runtime in Colab.")

    # 1. Quantization Configuration (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {args.model_id} in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 2. LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load Datasets
    root = Path(args.data_root)
    raw_rows = []
    manifests = [
        "dataset/external/hw_lines/train.jsonl",
        "dataset/external/hw_lines/validation.jsonl",
        "dataset/dastkhat_train.jsonl",
        "dataset/train_v3.jsonl",
        "dataset/train.jsonl",
    ]
    for manifest_name in manifests:
        mp = root / manifest_name if not manifest_name.startswith("/") else Path(manifest_name)
        if mp.exists():
            cnt = 0
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_rows.append(json.loads(line))
                        cnt += 1
            print(f"Loaded {cnt:,} records from {manifest_name}")

    # Strictly filter for existing images and non-empty ground-truth text
    train_rows = []
    for r in raw_rows:
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        p_str = str(r.get("image_path", r.get("image", "")))
        img_p = Path(p_str) if p_str.startswith("/") else (root / p_str)
        if img_p.is_file():
            train_rows.append(r)

    print(f"✅ Filtered clean training dataset: {len(train_rows):,} verified samples with images & labels")
    random.seed(42)
    random.shuffle(train_rows)

    gold_rows = []
    gold_path = root / "dataset" / "gold.jsonl"
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    g = json.loads(l)
                    g_img_p = Path(g["image_path"]) if g["image_path"].startswith("/") else (root / g["image_path"])
                    if g_img_p.is_file():
                        gold_rows.append(g)
        print(f"🏆 Loaded Gold Evaluation Benchmark: {len(gold_rows)} validated lines on disk")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    model.train()

    global_step = 0
    accum_steps = args.gradient_accumulation_steps
    accum_count = 0
    step_loss = 0.0
    best_word_acc = 0.0
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = "متن دست‌نویس فارسی داخل این تصویر را عیناً رونویسی کن:"

    print(f"Beginning training loop ({args.epochs} epochs, batch accumulation={accum_steps})...")
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        for idx, row in enumerate(train_rows):
            p_str = str(row.get("image_path", row.get("image", "")))
            img_p = Path(p_str) if p_str.startswith("/") else (root / p_str)
            try:
                img = Image.open(img_p).convert("RGB")
                img = prepare_image(img)
            except Exception:
                continue

            answer = str(row.get("text", "")).strip()
            if not answer:
                continue
            msgs = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            try:
                text = processor.apply_chat_template(msgs, tokenize=False)
                inputs = processor(images=[img], text=[text], return_tensors="pt").to(model.device)

                labels = inputs["input_ids"].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100
                inputs["labels"] = labels

                outputs = model(**inputs)
                loss = outputs.loss / accum_steps
                loss.backward()
                step_loss += loss.item()
                accum_count += 1
            except Exception:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if accum_count == accum_steps or (idx + 1) == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    speed = (global_step * accum_steps) / max(1, elapsed)
                    print(f"Step {global_step:4d} | Epoch {epoch+1}/{args.epochs} | Loss: {step_loss * accum_steps:.4f} | Speed: {speed:.1f} samples/s", flush=True)
                    step_loss = 0.0

                if global_step % args.eval_every == 0 and gold_rows:
                    print(f"\n--- Running Benchmark Evaluation on Gold Set (Step {global_step}) ---")
                    model.eval()
                    preds, refs = [], []
                    eval_sample = gold_rows[:60]
                    for g in eval_sample:
                        try:
                            g_img_p = root / g["image_path"] if not g["image_path"].startswith("/") else Path(g["image_path"])
                            g_img = Image.open(g_img_p).convert("RGB")
                            g_img = prepare_image(g_img)
                            eval_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                            eval_text = processor.apply_chat_template(eval_msgs, tokenize=False, add_generation_prompt=True)
                            eval_inputs = processor(images=[g_img], text=[eval_text], return_tensors="pt").to(model.device)
                            with torch.no_grad():
                                gen = model.generate(**eval_inputs, max_new_tokens=64, do_sample=False)
                            pred = processor.batch_decode(gen[:, eval_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                            preds.append(pred)
                            refs.append(g["text"])
                        except Exception:
                            continue
                    if refs and preds:
                        metrics = compute_metrics(refs, preds)
                        print(f"🏆 Step {global_step} Gold Metrics -> Word Acc: {metrics['word_acc']}% | Char Acc: {metrics['char_acc']}% | CER: {metrics['cer']}", flush=True)
                        print(f"   Ref sample:  {refs[0]}")
                        print(f"   Pred sample: {preds[0]}\n", flush=True)

                        if metrics["word_acc"] > best_word_acc:
                            best_word_acc = metrics["word_acc"]
                            best_ckpt = out_dir / "best_model"
                            print(f"⭐ New Best Benchmark: {best_word_acc}%! Saving to {best_ckpt}...", flush=True)
                            model.save_pretrained(best_ckpt)
                            processor.save_pretrained(best_ckpt)

                    model.train()

                if global_step % args.save_every == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    print(f"Saving checkpoint to {ckpt}...")
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)

    print(f"\n🎉 Training complete! Saving final master LoRA adapter to {out_dir / 'final_master'}...")
    model.save_pretrained(out_dir / "final_master")
    processor.save_pretrained(out_dir / "final_master")
    print("Export complete. You can now download final_master and drop it into your local exam-ocr server!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora_persian_htr")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    train_colab(args)
