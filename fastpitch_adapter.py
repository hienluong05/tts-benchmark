"""Adapter for the local third_party/FastPitch fork.

Default checkpoint format is eager PyTorch (.pt), loaded with this fork's
models.load_and_setup_model(), so the fork's fixes are used. TorchScript remains
available for existing .ts exports. The reference FastPitch + HiFi-GAN pipeline
is not native streaming: one full waveform is yielded per text segment.

Required: repo_dir, fastpitch_checkpoint, hifigan_checkpoint.
Optional: checkpoint_format='pyt' (default) or 'ts', cmudict_path and
heteronyms_path for p_arpabet, hifigan_config when an
 eager HiFi-GAN checkpoint does not embed its architecture configuration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


class FastPitchAdapter:
    name = "FastPitch+HiFi-GAN (non-streaming)"

    def __init__(self, *, repo_dir: str, fastpitch_checkpoint: str,
                 hifigan_checkpoint: str, checkpoint_format: str = "pyt",
                 hifigan_config: str | None = None, cmudict_path: str | None = None,
                 heteronyms_path: str | None = None, device: str = "cuda",
                 sample_rate: int = 22050, pace: float = 1.0,
                 speaker: int = 0, symbol_set: str = "english_basic",
                 text_cleaners: str | list[str] = ["english_cleaners_v2"], p_arpabet: float = 1.0,
                 amp: bool = False, **_: Any) -> None:
        if checkpoint_format not in {"pyt", "ts"}:
            raise ValueError("checkpoint_format must be 'pyt' or 'ts'")
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA was requested but is not available")
        root = Path(repo_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"FastPitch repo_dir does not exist: {root}")
        for checkpoint in (fastpitch_checkpoint, hifigan_checkpoint):
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if hifigan_config is not None and not Path(hifigan_config).is_file():
            raise FileNotFoundError(f"HiFi-GAN config does not exist: {hifigan_config}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from common.text import cmudict
            from common.text.text_processing import get_text_processing
            import models
        except ImportError as exc:
            raise ImportError("repo_dir must be the local third_party/FastPitch source root") from exc

        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.pace = pace
        self.speaker = speaker
        self.amp = amp
        if p_arpabet > 0.0:
            cmu_file = Path(cmudict_path) if cmudict_path else root / "cmudict" / "cmudict-0.7b"
            heteronym_file = Path(heteronyms_path) if heteronyms_path else root / "cmudict" / "heteronyms"
            if not cmu_file.is_file():
                raise FileNotFoundError(
                    f"CMUDict is required when p_arpabet={p_arpabet}; missing {cmu_file}. "
                    "Download cmudict-0.7b and pass cmudict_path."
                )
            cmudict.initialize(str(cmu_file), str(heteronym_file) if heteronym_file.is_file() else None)
        self.text_processor = get_text_processing(symbol_set, text_cleaners, p_arpabet)
        if checkpoint_format == "ts":
            self.fastpitch = models.load_and_setup_ts_model("FastPitch", fastpitch_checkpoint, amp, self.device)
            self.hifigan = models.load_and_setup_ts_model("HiFi-GAN", hifigan_checkpoint, amp, self.device)
        else:
            self.fastpitch = self._load_eager(models, "FastPitch", fastpitch_checkpoint, amp)
            self.hifigan = self._load_eager(models, "HiFi-GAN", hifigan_checkpoint, amp, hifigan_config)

    def _load_eager(self, models: Any, model_name: str, checkpoint: str,
                    amp: bool, hifigan_config: str | None = None) -> Any:
        """Use the fork's loader while keeping adapter CLI options out of sys.argv."""
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            if model_name == "HiFi-GAN" and hifigan_config is not None:
                sys.argv.extend(["--hifigan-config", hifigan_config])
            model, _config, _train_setup = models.load_and_setup_model(
                model_name, parser, checkpoint, amp, self.device,
                forward_is_infer=True, jitable=False,
            )
            return model
        finally:
            sys.argv = original_argv

    @torch.inference_mode()
    def stream(self, text: str):
        tokens = torch.tensor(self.text_processor.encode_text(text), dtype=torch.long,
                              device=self.device).unsqueeze(0)
        autocast_enabled = self.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
            output = self.fastpitch(tokens, pace=self.pace, speaker=self.speaker)
            mel = output[0] if isinstance(output, tuple) else output
            audio = self.hifigan(mel)
        # Transfer before yield: benchmark TTFA means audio playable by the client.
        yield audio.squeeze().float().cpu().numpy()


def create_adapter(**options: Any) -> FastPitchAdapter:
    return FastPitchAdapter(**options)
