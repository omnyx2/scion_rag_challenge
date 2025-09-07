#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_downloader_batch.py — RAG 데이터셋 다운로더 (배치, HF Hub + Kaggle)

기능
- JSON 설정파일로 여러 데이터셋을 한 번에 다운로드
- Hugging Face Hub: datasets.load_dataset로 CSV 저장 (full / sample 옵션)
- Kaggle: kagglehub.dataset_download로 내려받고 메타 기록
- meta.json이 있으면 작업 스킵 (--force로 재실행)
- 각 작업별 정책: max_bytes(선언 크기 제한), sample_rows, no_full 등

설치
  pip install -U datasets huggingface_hub pandas pyarrow kagglehub

Kaggle 인증
  - kagglehub는 OS 키체인/환경설정 자동탐색. 필요시 https://www.kaggle.com/docs/api 참고

JSON 포맷 (예시는 실행 섹션 참고)
- 공통: "dataset" (필수), "source": "hf" | "kaggle" (옵션; 기본 hf)
- HF 전용: "config", "splits" (리스트, 기본 ["train"]), "max_bytes" (바이트, 기본 무제한), "no_full" (bool), "sample_rows" (int)
- Kaggle 전용: "dataset" = "owner/name" (예: "allen-institute-for-ai/CORD-19-research-challenge")
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset, load_dataset_builder
import kagglehub

ISO = "%Y-%m-%dT%H:%M:%S%z"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_csv(ds, out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    if hasattr(ds, "to_csv"):
        ds.to_csv(str(out_csv))
    else:
        ds.to_pandas().to_csv(out_csv, index=False, encoding="utf-8")


def try_declared_size(repo_id: str, config: Optional[str]) -> Optional[int]:
    try:
        info = load_dataset_builder(repo_id, config).info
        return (
            int(
                getattr(info, "size_in_bytes", None)
                or getattr(info, "download_size", None)
                or 0
            )
            or None
        )
    except Exception:
        return None


def dir_stats(p: Path) -> Tuple[int, int]:
    """Return (file_count, total_size_bytes) for directory p (recursive)."""
    files = 0
    total = 0
    if p.exists():
        for root, _, fnames in os.walk(p):
            for f in fnames:
                files += 1
                try:
                    total += (Path(root) / f).stat().st_size
                except FileNotFoundError:
                    pass
    return files, total


def save_meta(path: Path, info: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def download_hf_job(
    job: Dict[str, Any],
    out_root: Path,
    token: Optional[str],
    defaults: Dict[str, Any],
    force: bool,
) -> None:
    repo = job["dataset"]
    config = job.get("config")
    splits = job.get("splits", defaults.get("splits", ["train"]))
    max_bytes = job.get("max_bytes", defaults.get("max_bytes"))
    sample_rows = int(job.get("sample_rows", defaults.get("sample_rows", 0)))
    save_full = not bool(job.get("no_full", defaults.get("no_full", False)))

    repo_base = repo.split("/")[-1]
    conf_name = (config or "default").replace("/", "_")

    declared_size = try_declared_size(repo, config)

    for split in splits:
        split_name = split.replace("/", "_")
        out_dir = out_root / repo_base / conf_name / split_name
        ensure_dir(out_dir)
        meta_path = out_dir / "meta.json"

        if meta_path.exists() and not force:
            print(f"[SKIP] HF {repo}/{conf_name}/{split} (meta.json 존재)")
            continue

        if (
            max_bytes is not None
            and max_bytes > 0
            and declared_size is not None
            and declared_size > max_bytes
        ):
            print(
                f"[SKIP] HF {repo}/{conf_name}/{split}: 선언크기 {declared_size}B > 제한 {max_bytes}B"
            )
            continue

        try:
            lk = {"split": split}
            if token:
                lk["token"] = token  # datasets v3
            ds = load_dataset(repo, config, **lk)
        except Exception as e:
            print(f"[ERROR] HF {repo}/{conf_name}/{split}: {e}")
            continue

        full_csv = out_dir / "full.csv" if save_full else None
        sample_csv = out_dir / f"sample_{sample_rows}.csv" if sample_rows > 0 else None

        saved_full = False
        if full_csv:
            try:
                to_csv(ds, full_csv)
                saved_full = True
                print(f"[OK] HF full.csv -> {full_csv}")
            except Exception as e:
                print(f"[WARN] HF full.csv 저장 실패: {e}")

        saved_sample = False
        if sample_csv:
            try:
                n = min(sample_rows, len(ds))
                to_csv(ds.select(range(n)), sample_csv)
                saved_sample = True
                print(f"[OK] HF 샘플 {n}행 -> {sample_csv}")
            except Exception as e:
                print(f"[WARN] HF 샘플 저장 실패: {e}")

        meta = {
            "source": "hf",
            "dataset": repo,
            "config": config or "default",
            "split": split,
            "columns": list(getattr(ds, "column_names", [])),
            "num_rows": len(ds),
            "declared_size_bytes": declared_size,
            "saved_csv": str(full_csv) if saved_full else None,
            "sample_csv": str(sample_csv) if saved_sample else None,
            "sample_rows": sample_rows if saved_sample else 0,
            "created_at": datetime.now(timezone.utc).strftime(ISO),
        }
        save_meta(meta_path, meta)
        print(f"[OK] HF meta.json -> {meta_path}")


def download_kaggle_job(job: Dict[str, Any], out_root: Path, force: bool) -> None:
    repo = job["dataset"]  # e.g., "allen-institute-for-ai/CORD-19-research-challenge"
    out_dir = out_root / repo.replace("/", "_")
    ensure_dir(out_dir)
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not force:
        print(f"[SKIP] Kaggle {repo} (meta.json 존재)")
        return

    try:
        local_path = Path(kagglehub.dataset_download(repo))
    except Exception as e:
        print(f"[ERROR] Kaggle {repo}: {e}")
        return

    fcnt, fsize = dir_stats(local_path)
    meta = {
        "source": "kaggle",
        "dataset": repo,
        "local_path": str(local_path),
        "file_count": fcnt,
        "total_size_bytes": fsize,
        "created_at": datetime.now(timezone.utc).strftime(ISO),
    }
    save_meta(meta_path, meta)
    print(f"[OK] Kaggle meta.json -> {meta_path} (다운로드 위치: {local_path})")


def main():
    ap = argparse.ArgumentParser(description="RAG 다운로더 (배치, HF + Kaggle)")
    ap.add_argument("--config-file", required=True, help="데이터셋 리스트 JSON 경로")
    ap.add_argument("--outdir", default="./data", help="출력 루트 디렉터리")
    ap.add_argument(
        "--hf-token",
        default=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
        help="HF 토큰(옵션)",
    )
    ap.add_argument(
        "--sample-rows", type=int, default=0, help="(HF 기본값) 샘플 CSV 행 수"
    )
    ap.add_argument(
        "--no-full", action="store_true", help="(HF 기본값) full.csv 저장 생략"
    )
    ap.add_argument(
        "--max-bytes", type=int, default=None, help="(HF 기본값) 선언 크기 제한"
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train",
        help="(HF 기본값) 콤마구분 splits, ex) train,validation",
    )
    ap.add_argument("--force", action="store_true", help="meta.json 있어도 재실행")
    args = ap.parse_args()

    defaults = {
        "sample_rows": args.sample_rows,
        "no_full": args.no_full,
        "max_bytes": args.max_bytes,
        "splits": [s.strip() for s in args.splits.split(",") if s.strip()] or ["train"],
    }

    out_root = Path(args.outdir)
    ensure_dir(out_root)

    with open(args.config_file, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    for job in jobs:
        source = job.get("source", "hf").lower()
        if source == "kaggle":
            print(f"\n=== Kaggle: {job['dataset']} ===")
            download_kaggle_job(job, out_root, args.force)
        else:
            print(f"\n=== HF: {job['dataset']} ===")
            download_hf_job(job, out_root, args.hf_token, defaults, args.force)


if __name__ == "__main__":
    main()
