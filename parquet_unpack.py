import os
import re
import json
import gzip
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
from tqdm import tqdm


# =========================
# 0) 配置（只改这里）
# =========================

AUG_AUDIO_ROOT = r"D:\capstone\asv_spoof_n"
ORIG_PARQUET_DIR = r"D:\capstone\asv_spoof\parquet"
OUT_ROOT = r"D:\capstone\sharded_dataset_50mb"

SPLITS = ["train", "validation", "test"]
AUDIO_SUFFIXES = {".wav", ".flac"}

# ⭐ 每个 tar 的目标大小：50MB
TARGET_TAR_SIZE_MB = 50
TARGET_BYTES = TARGET_TAR_SIZE_MB * 1024 * 1024

META_KEEP_COLS = ["speaker_id", "audio_file_name", "system_id", "key"]

# 文件名示例：LA_E_2834763__1__A10_snr0.wav
SNR_PAT = re.compile(r"_snr(-?\d+(?:\.\d+)?)$", re.IGNORECASE)


# =========================
# 1) 工具函数
# =========================

def parse_aug_filename(fname: str) -> Tuple[Optional[str], Optional[float]]:
    """
    LA_E_2834763__1__A10_snr0.wav
    -> ("la_e_2834763", 0.0)
    """
    stem = Path(fname).stem
    m = SNR_PAT.search(stem)
    if not m:
        return None, None

    snr = float(m.group(1))
    utt = stem[:m.start()].split("__", 1)[0].strip().lower()
    return utt if utt else None, snr


def iter_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def load_orig_metadata(split: str) -> Dict[str, Dict[str, Any]]:
    files = list(Path(ORIG_PARQUET_DIR).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet for split={split}")

    df = pd.concat(
        [pd.read_parquet(p) for p in files],
        ignore_index=True
    )

    keep = [c for c in META_KEEP_COLS if c in df.columns]
    index: Dict[str, Dict[str, Any]] = {}

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Build meta index ({split})"):
        key = str(r["audio_file_name"]).strip().lower()
        if not key:
            continue
        index[key] = {k: r[k] for k in keep}

    return index


# =========================
# 2) 核心：打 50MB shard + 写 index
# =========================

def build_shards_and_index():
    out_root = Path(OUT_ROOT)
    shards_dir = out_root / "shards"
    index_dir = out_root / "index"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n=== Processing split: {split} ===")

        split_audio_dir = Path(AUG_AUDIO_ROOT) / split
        if not split_audio_dir.exists():
            print(f"[SKIP] no dir: {split_audio_dir}")
            continue

        meta_index = load_orig_metadata(split)
        audio_files = iter_audio_files(split_audio_dir)

        print(f"Audio files: {len(audio_files)}")
        print(f"Meta index size: {len(meta_index)}")

        shard_id = 0
        cur_size = 0

        tar_path = shards_dir / f"{split}-{shard_id:05d}.tar"
        tar = tarfile.open(tar_path, "w")

        idx_path = index_dir / f"{split}.jsonl.gz"
        gz = gzip.open(idx_path, "wt", encoding="utf-8")

        written = 0
        no_match = 0
        no_snr = 0

        try:
            for p in tqdm(audio_files, desc=f"Shard {split}", unit="file"):
                utt, snr = parse_aug_filename(p.name)
                if utt is None:
                    no_snr += 1
                    continue

                meta = meta_index.get(utt)
                if meta is None:
                    no_match += 1
                    continue

                rel = p.relative_to(split_audio_dir).as_posix()
                member = f"{split}/{rel}"

                tar.add(p, arcname=member)
                size = p.stat().st_size
                cur_size += size

                record = dict(meta)
                record.update({
                    "snr": snr,
                    "tar": tar_path.name,
                    "member": member,
                    "aug_audio_file_name": p.name,
                })
                gz.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                # ⭐ 超过 50MB：切 shard
                if cur_size >= TARGET_BYTES:
                    tar.close()
                    shard_id += 1
                    tar_path = shards_dir / f"{split}-{shard_id:05d}.tar"
                    tar = tarfile.open(tar_path, "w")
                    cur_size = 0

        finally:
            tar.close()
            gz.close()

        print(
            f"[{split}] written={written}, "
            f"no_snr={no_snr}, no_match={no_match}, "
            f"shards={shard_id + 1}"
        )

    print("\n✅ All done.")


# =========================
# 3) main
# =========================

if __name__ == "__main__":
    build_shards_and_index()
