"""Adapter for NVIDIA DeepLearningExamples FastPitch TorchScript checkpoints.

This adapter deliberately yields one full waveform: the reference FastPitch +
HiFi-GAN pipeline is not native streaming.

Required options:
  repo_dir: path to DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
  fastpitch_checkpoint: TorchScript FastPitch checkpoint
  hifigan_checkpoint: TorchScript HiFi-GAN checkpoint
Optional: device='cuda', sample_rate=22050, pace=1.0, speaker=0,
          symbol_set='english_basic', text_cleaners='english_cleaners',
          p_arpabet=0.0, amp=False
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


class FastPitchAdapter:
    name = "FastPitch+HiFi-GAN (non-streaming)"

    def __init__(self, *, repo_dir: str, fastpitch_checkpoint: str,
                 hifigan_checkpoint: str, device: str = "cuda",
                 sample_rate: int = 22050, pace: float = 1.0,
                 speaker: int = 0, symbol_set: str = "english_basic",
                 text_cleaners: str = "english_cleaners", p_arpabet: float = 0.0,
                 amp: bool = False, **_: Any) -> None:
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA was requested but is not available")
        root = Path(repo_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"FastPitch repo_dir does not exist: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from common.text.text_processing import get_text_processing
        except ImportError as exc:
            raise ImportError("repo_dir must be NVIDIA DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch") from exc
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.pace = pace
        self.speaker = speaker
        self.amp = amp
        self.text_processor = get_text_processing(symbol_set, text_cleaners, p_arpabet)
        self.fastpitch = torch.jit.load(fastpitch_checkpoint, map_location=self.device).eval()
        self.hifigan = torch.jit.load(hifigan_checkpoint, map_location=self.device).eval()

    @torch.inference_mode()
    def stream(self, text: str):
        tokens = torch.tensor(self.text_processor.encode_text(text), dtype=torch.long,
                              device=self.device).unsqueeze(0)
        autocast_enabled = self.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
            output = self.fastpitch(tokens, pace=self.pace, speaker=self.speaker)
            mel = output[0] if isinstance(output, tuple) else output
            audio = self.hifigan(mel)
        # Transfer before yield: the benchmark's TTFA means playable client audio.
        yield audio.squeeze().float().cpu().numpy()


def create_adapter(**options: Any) -> FastPitchAdapter:
    return FastPitchAdapter(**options)
