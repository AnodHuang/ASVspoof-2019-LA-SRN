# train_hubert_stream_4090.py
import os
import json
import time
import math
import argparse
from glob import glob
import io

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification

import soundfile as sf


# ==============
# 默认：离线 + 国内环境
# ==============
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

AUDIO_COL = "wav"
PARQUET_KEY_COL = "__key__"
JSONL_KEY_COL = "member"
JSONL_LABEL_COL = "key"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./ASV_Spoof_2019_LA_SNR_50MB")
    p.add_argument("--model_dir", type=str, default="./hubert")
    p.add_argument("--out", type=str, default="./hubert_stream_out")

    p.add_argument("--sr", type=int, default=16000)

    # ✅ 4090：回到 6 秒（甚至 8 秒都可以，你先用 6）
    p.add_argument("--max_sec", type=float, default=6.0)

    p.add_argument("--epochs", type=int, default=3)

    # ✅ 4090：大 batch（常用 16；如果你显存很充足也可以 24/32）
    p.add_argument("--batch", type=int, default=16)

    # ✅ 4090：通常不需要累积；保持 1 最快最简单
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every_epoch", action="store_true", default=True)

    # ✅ 4090：更大的 shuffle buffer（更随机）
    p.add_argument("--train_buffer_shuffle", type=int, default=50000)

    p.add_argument("--val_take", type=int, default=0)
    p.add_argument("--fp16", action="store_true", default=True)

    # ✅ 4090：可以开多点 workers（Windows 上建议 2~4；Linux 可 4~8）
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)

    p.add_argument("--train_size_hint", type=int, default=45600)

    return p.parse_args()


def find_parquet_files(data_dir: str, split: str):
    base = os.path.join(data_dir, "default")
    pat = {"train": "partial-train", "validation": "partial-validation", "test": "partial-test"}[split]
    files = sorted(glob(os.path.join(base, pat, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"没找到 {split} parquet: {os.path.join(base, pat)}/*.parquet")
    return files


def find_jsonl(data_dir: str, split: str):
    cands = [
        os.path.join(data_dir, "index", f"{split}.jsonl"),
        os.path.join(data_dir, f"{split}.jsonl"),
        os.path.join(data_dir, "default", "index", f"{split}.jsonl"),
        os.path.join(data_dir, "default", f"{split}.jsonl"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"找不到 {split}.jsonl（建议放到 {data_dir}/index/{split}.jsonl）")


def load_member2label(jsonl_path: str):
    m2l = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m = obj.get(JSONL_KEY_COL, None)
            k = obj.get(JSONL_LABEL_COL, None)
            if m is None or k is None:
                continue

            # 统一为 0/1：1=bonafide, 0=spoof
            if isinstance(k, (int, np.integer)):
                label = 1 if int(k) == 1 else 0
            else:
                s = str(k).lower()
                label = 1 if s == "bonafide" else 0

            m2l[str(m)] = int(label)

    if not m2l:
        raise ValueError(f"{jsonl_path} 没读到任何 member->label")
    return m2l


def decode_wav_any(w, target_sr: int):
    """
    兼容 parquet 内 wav 字段可能是：
    - dict {'bytes':..., 'path':...}  ✅我们希望保持这种，不要 datasets 自动 decode
    - bytes/bytearray
    - dict {'array':..., 'sampling_rate':...}（少见）
    """
    if isinstance(w, dict):
        if "bytes" in w and w["bytes"] is not None:
            x, sr0 = sf.read(io.BytesIO(w["bytes"]), dtype="float32")
            return x, sr0
        if "array" in w and w["array"] is not None:
            x = np.asarray(w["array"], dtype=np.float32)
            sr0 = int(w.get("sampling_rate", target_sr))
            return x, sr0

    if isinstance(w, (bytes, bytearray)):
        x, sr0 = sf.read(io.BytesIO(w), dtype="float32")
        return x, sr0

    x = np.asarray(w, dtype=np.float32)
    return x, target_sr


def cheap_resample(x: np.ndarray, sr0: int, sr1: int):
    if sr0 == sr1:
        return x
    n1 = int(round(len(x) * (sr1 / sr0)))
    if n1 <= 1:
        return x[:1]
    idx = np.linspace(0, len(x) - 1, n1).astype(np.float64)
    x0 = np.arange(len(x), dtype=np.float64)
    y = np.interp(idx, x0, x).astype(np.float32)
    return y


def disable_audio_decoding(ds, audio_col: str, sr: int):
    """
    兼容新旧 datasets：
    - 新版：IterableDataset 有 .decode(False)
    - 老版：没有 .decode，用 cast_column(Audio(decode=False)) 关闭音频解码
    """
    if hasattr(ds, "decode"):
        try:
            return ds.decode(False)
        except TypeError:
            # 有些版本 decode 参数形式不同，失败就继续走 cast_column
            pass

    if hasattr(ds, "cast_column"):
        # 老版 datasets：关键就是 Audio(decode=False)
        try:
            return ds.cast_column(audio_col, Audio(decode=False))
        except TypeError:
            # 极老版本 Audio 没有 decode 参数：退一步，只设 sampling_rate
            return ds.cast_column(audio_col, Audio(sampling_rate=sr))

    return ds


class StreamCollator:
    def __init__(self, feature_extractor, member2label, sr=16000, max_sec=6.0):
        self.fe = feature_extractor
        self.m2l = member2label
        self.sr = sr
        self.max_len = int(sr * max_sec)

    def __call__(self, batch):
        audios = []
        labels = []

        for ex in batch:
            kk = str(ex.get(PARQUET_KEY_COL, "")) + ".wav"
            if kk == "" or kk not in self.m2l:
                raise ValueError(
                    f"jsonl 找不到 member={kk} 的标签（检查 parquet.__key__ 与 jsonl.member 是否一致）"
                )
            labels.append(self.m2l[kk])

            w = ex.get(AUDIO_COL, None)
            if w is None:
                raise ValueError(f"样本缺少音频列 {AUDIO_COL}")

            x, sr0 = decode_wav_any(w, self.sr)
            x = np.asarray(x, dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(axis=-1)

            if sr0 != self.sr:
                x = cheap_resample(x, sr0, self.sr)

            if len(x) >= self.max_len:
                x = x[: self.max_len]
            else:
                x = np.pad(x, (0, self.max_len - len(x)))

            audios.append(x)

        inputs = self.fe(audios, sampling_rate=self.sr, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


@torch.no_grad()
def eval_loop(model, dl, device, fp16: bool):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in dl:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=fp16):
            logits = model(**batch).logits

        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)

    acc = float((preds == labels).mean()) if len(labels) else float("nan")

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    roc_auc = float("nan")
    if len(labels) and len(np.unique(labels)) == 2:
        order = np.argsort(probs)
        y = labels[order]
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        if n_pos > 0 and n_neg > 0:
            ranks = np.arange(1, len(y) + 1)
            sum_ranks_pos = ranks[y == 1].sum()
            roc_auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    model.train()
    return {"acc": acc, "f1": f1, "roc_auc": roc_auc, "n": int(len(labels))}


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA 不可用"
    device = torch.device("cuda")
    print("CUDA OK:", torch.cuda.get_device_name(0))

    # 4090：建议开启这俩
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_files = find_parquet_files(args.data_dir, "train")
    val_files = find_parquet_files(args.data_dir, "validation")
    train_jsonl = find_jsonl(args.data_dir, "train")
    val_jsonl = find_jsonl(args.data_dir, "validation")

    train_m2l = load_member2label(train_jsonl)
    val_m2l = load_member2label(val_jsonl)
    print("labels loaded:", len(train_m2l), len(val_m2l))

    # ✅ 关键：禁用自动音频解码（兼容新旧 datasets）
    train_stream = load_dataset("parquet", data_files={"train": train_files}, streaming=True)["train"]
    train_stream = disable_audio_decoding(train_stream, AUDIO_COL, args.sr)
    train_stream = train_stream.shuffle(buffer_size=args.train_buffer_shuffle, seed=42)

    val_stream = load_dataset("parquet", data_files={"validation": val_files}, streaming=True)["validation"]
    val_stream = disable_audio_decoding(val_stream, AUDIO_COL, args.sr)
    if args.val_take and args.val_take > 0:
        val_stream = val_stream.take(int(args.val_take))

    fe = AutoFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)

    model = HubertForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=2,
        id2label={0: "spoof", 1: "bonafide"},
        label2id={"spoof": 0, "bonafide": 1},
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)

    model.train()

    train_collator = StreamCollator(fe, train_m2l, sr=args.sr, max_sec=args.max_sec)
    val_collator = StreamCollator(fe, val_m2l, sr=args.sr, max_sec=args.max_sec)

    train_dl = DataLoader(
        train_stream,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_collator,
    )
    val_dl = DataLoader(
        val_stream,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=val_collator,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    os.makedirs(args.out, exist_ok=True)

    best_auc = -1.0
    global_step = 0

    steps_per_epoch = max(1, math.ceil(args.train_size_hint / max(1, args.batch)))
    print(f"steps_per_epoch={steps_per_epoch} (train_size_hint={args.train_size_hint}, batch={args.batch})")
    print(f"effective_batch = {args.batch} * {args.grad_accum} = {args.batch * args.grad_accum}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
        t0 = time.time()
        running = 0.0
        seen = 0

        it = iter(train_dl)
        optim.zero_grad(set_to_none=True)

        for step_in_epoch in range(steps_per_epoch):
            batch = next(it)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=args.fp16):
                loss = model(**batch).loss
                loss_scaled = loss / args.grad_accum  # 4090默认accum=1，不影响

            scaler.scale(loss_scaled).backward()

            if (step_in_epoch + 1) % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running += float(loss.item()) * batch["labels"].size(0)
            seen += int(batch["labels"].size(0))
            global_step += 1

            if global_step % args.log_every == 0:
                avg = running / max(1, seen)
                dt = time.time() - t0
                spd = seen / max(1e-9, dt)
                mem = torch.cuda.memory_allocated() / (1024**3)
                print(
                    f"step {global_step:6d} | loss(avg)={avg:.4f} | samples={seen} | "
                    f"{spd:.1f} samp/s | mem={mem:.2f} GB"
                )

        if steps_per_epoch % args.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        if args.eval_every_epoch:
            metrics = eval_loop(model, val_dl, device, fp16=args.fp16)
            print(f"[VAL] n={metrics['n']} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

            cur_auc = metrics["roc_auc"]
            if not np.isnan(cur_auc) and cur_auc > best_auc:
                best_auc = cur_auc
                save_dir = os.path.join(args.out, "best")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                fe.save_pretrained(save_dir)
                print(f"✅ saved best to: {save_dir} (roc_auc={best_auc:.4f})")

        last_dir = os.path.join(args.out, "last")
        os.makedirs(last_dir, exist_ok=True)
        model.save_pretrained(last_dir)
        fe.save_pretrained(last_dir)
        print(f"saved last to: {last_dir}")

    print("\nDONE. best roc_auc =", best_auc)


if __name__ == "__main__":
    main()
