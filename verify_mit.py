import os
import io
import time
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import pyarrow.parquet as pq

from transformers import AutoFeatureExtractor, ASTForAudioClassification

# =========================
# 0) 你只改这里
# =========================
PARQUET_DIR = r"D:\capstone\asv_spoof\parquet"
MODEL_DIR   = r"D:\capstone\models\mit"

SPLIT = "test"          # "train" / "validation" / "test"
BATCH_SIZE = 32         # 4090 可 64
CPU_THREADS = 8         # 影响音频解码/预处理

# key 的定义：根据你的数据分布 & system_id 对齐： key=1 是 spoof，key=0 是 bonafide
# （system_id: '-' 是 bonafide；'Axx' 是 spoof）
KEY_SPOOF_VALUE = 1

PARQUET_FILE = os.path.join(PARQUET_DIR, f"{SPLIT}-00000-of-00001.parquet")

# 是否做 system_id 与 key 的一致性检查（不影响推理，只打印检查结果）
CHECK_LABEL_CONSISTENCY = True


# =========================
# 1) 音频解码/重采样（不落盘）
# =========================
def decode_audio(bytes_blob: bytes | None, path_str: str | None):
    if bytes_blob is not None:
        wav, sr = sf.read(io.BytesIO(bytes_blob), dtype="float32", always_2d=False)
    else:
        if not path_str or not os.path.exists(path_str):
            raise RuntimeError("audio.bytes 为空，且 audio.path 不存在/不可用")
        wav, sr = sf.read(path_str, dtype="float32", always_2d=False)

    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def simple_resample(wav: np.ndarray, sr: int, new_sr: int) -> np.ndarray:
    if sr == new_sr:
        return wav
    if wav.size == 0:
        return wav
    x_old = np.linspace(0, 1, num=wav.shape[0], endpoint=False)
    new_len = int(round(wav.shape[0] * (new_sr / sr)))
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def key_to_label01(k) -> int:
    # parquet 里 key 是 int64，但有时 to_pylist 可能给 int 或 str
    v = int(k)
    return 1 if v == KEY_SPOOF_VALUE else 0


def system_id_to_label01(sid: str) -> int:
    sid = str(sid).strip()
    return 0 if sid == "-" else 1  # '-' bonafide, 'Axx' spoof


# =========================
# 2) 设备 & 模型
# =========================
torch.set_num_threads(CPU_THREADS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

use_amp = (device.type == "cuda")

extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = ASTForAudioClassification.from_pretrained(MODEL_DIR).to(device).eval()
target_sr = getattr(extractor, "sampling_rate", 16000)

# =========================
# 3) 读 parquet
# =========================
pf = pq.ParquetFile(PARQUET_FILE)
num_rows = pf.metadata.num_rows
num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Parquet: {PARQUET_FILE}")
print(f"Rows: {num_rows}, Batches: {num_batches}, BatchSize: {BATCH_SIZE}")

# =========================
# 4) 推理 + 指标统计
# =========================
correct = 0
total = 0
tp = fp = tn = fn = 0  # pos=spoof=1

# 可选：检查 key 与 system_id 是否一致
mismatch = 0
checked = 0

t0 = time.time()
with torch.no_grad():
    pbar = tqdm(total=num_batches, desc=f"Predicting [{SPLIT}]", unit="batch")

    for rb in pf.iter_batches(batch_size=BATCH_SIZE, columns=["audio", "key", "system_id"]):
        audio_struct = rb.column(rb.schema.get_field_index("audio"))
        key_arr = rb.column(rb.schema.get_field_index("key"))
        sys_arr = rb.column(rb.schema.get_field_index("system_id"))

        bytes_arr = audio_struct.field("bytes") if audio_struct.type.get_field_index("bytes") != -1 else None
        path_arr  = audio_struct.field("path")  if audio_struct.type.get_field_index("path")  != -1 else None

        keys = key_arr.to_pylist()
        sysids = sys_arr.to_pylist()
        bytes_list = bytes_arr.to_pylist() if bytes_arr is not None else [None] * len(keys)
        path_list  = path_arr.to_pylist()  if path_arr  is not None else [None] * len(keys)

        waves = []
        labels = []

        for b, p, k, sid in zip(bytes_list, path_list, keys, sysids):
            y = key_to_label01(k)
            labels.append(y)

            if CHECK_LABEL_CONSISTENCY:
                y2 = system_id_to_label01(sid)
                checked += 1
                if y != y2:
                    mismatch += 1

            wav, sr = decode_audio(b, p)
            wav = simple_resample(wav, sr, target_sr)
            waves.append(wav)

        inputs = extractor(
            waves,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits

        preds = torch.argmax(logits, dim=-1)

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
# 5) 计算指标
# =========================
acc = correct / max(total, 1)

eps = 1e-12
precision = tp / (tp + fp + eps)
recall    = tp / (tp + fn + eps)  # TPR
f1        = 2 * precision * recall / (precision + recall + eps)
fnr       = fn / (fn + tp + eps)
fpr       = fp / (fp + tn + eps)

print("\n===== Summary =====")
print(f"Split      : {SPLIT}")
print(f"Accuracy   : {acc:.6f}  ({correct}/{total})")
print(f"Confusion  : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"Time       : {elapsed:.2f}s, {total / max(elapsed,1e-9):.2f} samples/s")

if CHECK_LABEL_CONSISTENCY:
    print(f"Label check: key vs system_id mismatches = {mismatch}/{checked}")

print("\n===== Metrics (pos=spoof=1) =====")
print(f"Precision  : {precision:.6f}")
print(f"FNR        : {fnr:.6f}")
print(f"FPR        : {fpr:.6f}")
print(f"F1-score   : {f1:.6f}")


'''
===== Summary =====
Split      : test
Accuracy   : 0.922498  (65716/71237)
Confusion  : TP=58549, FP=188, TN=7167, FN=5333
Time       : 1473.21s, 48.35 samples/s
Label check: key vs system_id mismatches = 0/71237

===== Metrics (pos=spoof=1) =====
Precision  : 0.996799
FNR        : 0.083482
FPR        : 0.025561
F1-score   : 0.954974

进程已结束，退出代码为 0

'''