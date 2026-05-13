 import os
import io
import time
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import pyarrow.parquet as pq

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# =========================
# 0) 配置
# =========================
PARQUET_DIR = r"D:\capstone\asv_spoof\parquet"

# ✅ 如果是原始模型
MODEL_DIR = r"D:\capstone\models\wav2veclarge_srn"
# ✅ 如果是你 fine-tune 后的模型
# MODEL_DIR = r"D:\capstone\models\wav2vec2_snr"

SPLIT = "test"
BATCH_SIZE = 32        # RTX 4060 推荐 16~32
CPU_THREADS = 8

KEY_SPOOF_VALUE = 1    # key=1 → spoof

PARQUET_FILE = os.path.join(PARQUET_DIR, f"{SPLIT}-00000-of-00001.parquet")
CHECK_LABEL_CONSISTENCY = True


# =========================
# 1) 音频解码
# =========================
def decode_audio(bytes_blob, path_str):
    if bytes_blob is not None:
        wav, sr = sf.read(io.BytesIO(bytes_blob), dtype="float32")
    else:
        wav, sr = sf.read(path_str, dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def resample(wav, sr, target_sr):
    if sr == target_sr:
        return wav
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    new_len = int(len(wav) * target_sr / sr)
    x_new = np.linspace(0, 1, new_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def key_to_label(k):
    return 1 if int(k) == KEY_SPOOF_VALUE else 0


def system_id_to_label(sid):
    return 0 if str(sid).strip() == "-" else 1


# =========================
# 2) 设备 & 模型
# =========================
torch.set_num_threads(CPU_THREADS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

use_amp = device.type == "cuda"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

target_sr = feature_extractor.sampling_rate  # 16000


# =========================
# 3) 读 parquet
# =========================
pf = pq.ParquetFile(PARQUET_FILE)
num_rows = pf.metadata.num_rows
num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Parquet: {PARQUET_FILE}")
print(f"Rows: {num_rows}, Batches: {num_batches}")


# =========================
# 4) 推理
# =========================
tp = fp = tn = fn = 0
correct = total = 0
mismatch = checked = 0

t0 = time.time()
with torch.no_grad():
    pbar = tqdm(total=num_batches, desc=f"Predicting [{SPLIT}]", unit="batch")

    for rb in pf.iter_batches(batch_size=BATCH_SIZE, columns=["audio", "key", "system_id"]):
        audio_struct = rb.column(rb.schema.get_field_index("audio"))
        key_arr = rb.column(rb.schema.get_field_index("key"))
        sys_arr = rb.column(rb.schema.get_field_index("system_id"))

        bytes_arr = audio_struct.field("bytes")
        path_arr  = audio_struct.field("path")

        waves, labels = [], []

        for b, p, k, sid in zip(
            bytes_arr.to_pylist(),
            path_arr.to_pylist(),
            key_arr.to_pylist(),
            sys_arr.to_pylist(),
        ):
            y = key_to_label(k)
            labels.append(y)

            if CHECK_LABEL_CONSISTENCY:
                checked += 1
                if y != system_id_to_label(sid):
                    mismatch += 1

            wav, sr = decode_audio(b, p)
            wav = resample(wav, sr, target_sr)
            waves.append(wav)

        inputs = feature_extractor(
            waves,
            sampling_rate=target_sr,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels_t = torch.tensor(labels, device=device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits

        preds = logits.argmax(dim=-1)

        total += labels_t.numel()
        correct += (preds == labels_t).sum().item()

        tp += ((preds == 1) & (labels_t == 1)).sum().item()
        fp += ((preds == 1) & (labels_t == 0)).sum().item()
        tn += ((preds == 0) & (labels_t == 0)).sum().item()
        fn += ((preds == 0) & (labels_t == 1)).sum().item()

        pbar.update(1)

    pbar.close()

elapsed = time.time() - t0


# =========================
# 5) 指标
# =========================
eps = 1e-12
acc = correct / max(total, 1)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2 * precision * recall / (precision + recall + eps)
fnr = fn / (fn + tp + eps)
fpr = fp / (fp + tn + eps)

print("\n===== Summary =====")
print(f"Accuracy   : {acc:.6f} ({correct}/{total})")
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"Time       : {elapsed:.2f}s, {total/elapsed:.2f} samples/s")

if CHECK_LABEL_CONSISTENCY:
    print(f"Label check: key vs system_id mismatches = {mismatch}/{checked}")

print("\n===== Metrics (pos=spoof=1) =====")
print(f"Precision  : {precision:.6f}")
print(f"Recall     : {recall:.6f}")
print(f"FNR        : {fnr:.6f}")
print(f"FPR        : {fpr:.6f}")
print(f"F1-score   : {f1:.6f}")

'''
===== Summary =====
Accuracy   : 0.896753 (63882/71237)
TP=63882, FP=7355, TN=0, FN=0
Time       : 4854.95s, 14.67 samples/s
Label check: key vs system_id mismatches = 0/71237

===== Metrics (pos=spoof=1) =====
Precision  : 0.896753
Recall     : 1.000000
FNR        : 0.000000
FPR        : 1.000000
F1-score   : 0.945567

进程已结束，退出代码为 0


'''