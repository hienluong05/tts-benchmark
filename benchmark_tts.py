"""Benchmark native TTS streaming adapters (TTFA and RTF).

Usage:
  python benchmark_tts.py --adapter fastpitch_adapter:create_adapter --adapter omnivoice_adapter:create_adapter --texts texts.txt

Each adapter module exports create_adapter(**options), returning an object with
name, sample_rate, and stream(text). stream() must yield only audio that is
already available to the client; do not chunk a full waveform afterwards.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class AudioChunk:
    """Host-resident PCM. For bytes, specify channels and sample_width_bytes."""
    data: Any
    sample_rate: int
    channels: int = 1
    sample_width_bytes: int | None = None


class StreamingTTSAdapter(Protocol):
    name: str
    sample_rate: int
    def stream(self, text: str) -> Iterable[Any]: ...


@dataclass
class BenchmarkResult:
    model: str
    text_id: int
    text: str
    run_id: int
    ttfa_s: float
    total_time_s: float
    audio_duration_s: float
    e2e_rtf: float
    post_ttfa_rtf: float
    first_chunk_duration_s: float
    num_chunks: int
    num_samples_per_channel: int
    sample_rate: int
    streaming_mode: str
    segment_count: int


def cuda_sync() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def chunk_info(value: Any, default_rate: int) -> tuple[int, int]:
    """Return frames per channel and sample rate for playable data."""
    chunk = value if isinstance(value, AudioChunk) else AudioChunk(value, default_rate)
    if chunk.sample_rate <= 0 or chunk.channels <= 0:
        raise ValueError("sample_rate and channels must be positive")
    data = chunk.data
    if isinstance(data, bytes):
        width = chunk.sample_width_bytes or 2  # plain bytes are treated as PCM16
        divisor = width * chunk.channels
        if len(data) % divisor:
            raise ValueError("PCM byte length does not match its declared layout")
        frames = len(data) // divisor
    elif torch is not None and isinstance(data, torch.Tensor):
        if data.is_cuda:
            raise ValueError("Adapter yielded CUDA data; move it to CPU before yield")
        if data.numel() % chunk.channels:
            raise ValueError("Tensor length does not match channel count")
        frames = data.numel() // chunk.channels
    elif isinstance(data, np.ndarray):
        if data.size % chunk.channels:
            raise ValueError("Array length does not match channel count")
        frames = data.size // chunk.channels
    else:
        raise TypeError(f"Unsupported audio chunk: {type(data).__name__}")
    return frames, chunk.sample_rate

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")


def split_for_application_streaming(text: str, max_segment_chars: int) -> list[str]:
    """Split deterministically at sentence boundaries, then at spaces if needed."""
    if max_segment_chars < 1:
        raise ValueError("max_segment_chars must be positive")
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]
    if not sentences:
        raise ValueError("Cannot segment empty text")
    segments: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_segment_chars:
            cut = sentence.rfind(" ", 0, max_segment_chars + 1)
            cut = cut if cut > 0 else max_segment_chars
            head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                segments.append(current)
                current = ""
            if head:
                segments.append(head)
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_segment_chars:
            current = candidate
        else:
            segments.append(current)
            current = sentence
    if current:
        segments.append(current)
    return segments


def iter_application_stream(adapter: StreamingTTSAdapter, segments: Sequence[str]) -> Iterable[Any]:
    """Sequential phrase synthesis: the first completed phrase is playable TTFA."""
    for segment in segments:
        yield from adapter.stream(segment)

def benchmark_once(adapter: StreamingTTSAdapter, text: str, text_id: int, run_id: int, segments: Sequence[str] | None = None) -> BenchmarkResult:
    """Timer includes request dispatch through receipt of the final audio chunk."""
    cuda_sync()
    started = time.perf_counter()
    first_audio_at: float | None = None
    sample_rate: int | None = None
    first_duration = 0.0
    total_frames = 0
    chunks = 0

    audio_stream = iter_application_stream(adapter, segments) if segments is not None else adapter.stream(text)
    for raw_chunk in audio_stream:
        # A yielded CUDA tensor is rejected. This sync makes GPU completion part
        # of TTFA and protects timings for normal in-process PyTorch adapters.
        cuda_sync()
        frames, rate = chunk_info(raw_chunk, adapter.sample_rate)
        if not frames:
            continue
        if sample_rate is None:
            sample_rate = rate
        elif sample_rate != rate:
            raise ValueError("sample_rate changed within a single stream")
        now = time.perf_counter()
        if first_audio_at is None:
            first_audio_at = now
            first_duration = frames / rate
        total_frames += frames
        chunks += 1

    cuda_sync()
    ended = time.perf_counter()
    if first_audio_at is None or sample_rate is None:
        raise RuntimeError(f"{adapter.name} produced no non-empty playable audio")
    duration = total_frames / sample_rate
    total = ended - started
    return BenchmarkResult(
        model=adapter.name, text_id=text_id, text=text, run_id=run_id,
        ttfa_s=first_audio_at - started, total_time_s=total,
        audio_duration_s=duration,
        e2e_rtf=total / duration if duration else math.inf,
        post_ttfa_rtf=(ended - first_audio_at) / duration if duration else math.inf,
        first_chunk_duration_s=first_duration, num_chunks=chunks,
        num_samples_per_channel=total_frames, sample_rate=sample_rate,
        streaming_mode="segment_streaming" if segments is not None else "model_stream",
        segment_count=len(segments) if segments is not None else 1,
    )


def load_adapter(spec: str, options: dict[str, Any]) -> StreamingTTSAdapter:
    try:
        module_name, factory_name = spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(f"Invalid adapter {spec!r}; expected module:create_adapter") from exc
    adapter = factory(**options)
    for field in ("name", "sample_rate", "stream"):
        if not hasattr(adapter, field):
            raise TypeError(f"Adapter {spec!r} is missing {field!r}")
    return adapter


def summarize(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    """Aggregate by model and input. RTF uses ratio of totals, not mean RTF."""
    groups: dict[tuple[str, int, str], list[BenchmarkResult]] = {}
    for item in results:
        groups.setdefault((item.model, item.text_id, item.streaming_mode), []).append(item)
    rows = []
    for (model, text_id, streaming_mode), group in groups.items():
        ttfa = [item.ttfa_s for item in group]
        total_duration = sum(item.audio_duration_s for item in group)
        rows.append({
            "model": model, "streaming_mode": streaming_mode, "text_id": text_id, "runs": len(group), "segments": group[0].segment_count, "text": group[0].text,
            "ttfa_mean_ms": 1000 * statistics.mean(ttfa),
            "ttfa_p50_ms": 1000 * float(np.percentile(ttfa, 50)),
            "ttfa_p95_ms": 1000 * float(np.percentile(ttfa, 95)),
            "e2e_rtf_ratio_of_totals": sum(item.total_time_s for item in group) / total_duration,
            "post_ttfa_rtf_ratio_of_totals": sum(item.total_time_s - item.ttfa_s for item in group) / total_duration,
            "audio_duration_mean_s": statistics.mean(item.audio_duration_s for item in group),
            "first_chunk_mean_ms": 1000 * statistics.mean(item.first_chunk_duration_s for item in group),
        })
    return rows


def write_outputs(out_dir: Path, results: Sequence[BenchmarkResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = {"runs": [asdict(item) for item in results], "summary": summarize(results)}
    for name, rows in batches.items():
        (out_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native streaming TTS adapters")
    parser.add_argument("--adapter", action="append", required=True, metavar="MODULE:FACTORY")
    parser.add_argument("--adapter-options", action="append", default=[], help="JSON kwargs; supply once or once per --adapter in order")
    parser.add_argument("--texts", type=Path, required=True, help="UTF-8 file; one test sentence per line")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--application-streaming", action="store_true", help="Synthesize deterministic text segments sequentially")
    parser.add_argument("--max-segment-chars", type=int, default=120, help="Maximum characters per application-streamed segment")
    args = parser.parse_args()
    option_blobs = args.adapter_options or ["{}"]
    if len(option_blobs) not in (1, len(args.adapter)):
        raise ValueError("Provide --adapter-options once or once for every --adapter")
    adapter_options = [json.loads(blob) for blob in option_blobs]
    if len(adapter_options) == 1:
        adapter_options *= len(args.adapter)
    if any(not isinstance(options, dict) for options in adapter_options) or args.warmup < 0 or args.repeat < 1:
        raise ValueError("adapter options must be objects; warmup >= 0; repeat >= 1")
    texts = [line.strip() for line in args.texts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise ValueError("No non-empty test texts")
    if args.application_streaming and args.max_segment_chars < 1:
        raise ValueError("--max-segment-chars must be positive")

    results: list[BenchmarkResult] = []
    for spec, options in zip(args.adapter, adapter_options):
        adapter = load_adapter(spec, options)
        print(f"\n{adapter.name}: warmup={args.warmup}, repeat={args.repeat}")
        warmup_segments = split_for_application_streaming(texts[0], args.max_segment_chars) if args.application_streaming else None
        for _ in range(args.warmup):
            audio_stream = iter_application_stream(adapter, warmup_segments) if warmup_segments is not None else adapter.stream(texts[0])
            for _chunk in audio_stream:
                cuda_sync()
        for text_id, text in enumerate(texts, 1):
            for run_id in range(1, args.repeat + 1):
                segments = split_for_application_streaming(text, args.max_segment_chars) if args.application_streaming else None
                item = benchmark_once(adapter, text, text_id, run_id, segments)
                results.append(item)
                print(f"text={text_id} run={run_id:02d} mode={item.streaming_mode} segments={item.segment_count} TTFA={item.ttfa_s * 1000:.1f}ms E2E_RTF={item.e2e_rtf:.4f} chunks={item.num_chunks}")
    write_outputs(args.out_dir, results)
    print(f"Wrote {len(results)} runs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
