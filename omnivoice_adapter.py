"""Adapter for the public OmniVoice Python API.

OmniVoice currently documents `generate()` returning a list of complete 24 kHz
numpy waveforms; it does not expose a streaming generator. This adapter yields
that completed waveform once, so TTFA is correctly reported as full latency.

Options: model_id='k2-fsa/OmniVoice', device='cuda:0', dtype='float16',
ref_audio=None, ref_text=None, instruct=None, generate_options={...}.
Use generate_options for num_step, speed, duration, language_id, and similar
keyword arguments accepted by your installed OmniVoice version.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class OmniVoiceAdapter:
    name = "OmniVoice (public generate API; non-streaming)"
    sample_rate = 24000

    def __init__(self, *, model_id: str = "k2-fsa/OmniVoice", device: str = "cuda:0",
                 dtype: str = "float16", ref_audio: str | None = None,
                 ref_text: str | None = None, instruct: str | None = None,
                 generate_options: dict[str, Any] | None = None, **_: Any) -> None:
        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise ImportError("Install OmniVoice first: pip install omnivoice") from exc
        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype: {dtype}") from exc
        self.model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=torch_dtype)
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.instruct = instruct
        self.generate_options = dict(generate_options or {})

    @torch.inference_mode()
    def stream(self, text: str):
        kwargs = dict(self.generate_options)
        if self.ref_audio:
            kwargs["ref_audio"] = self.ref_audio
        if self.ref_text:
            kwargs["ref_text"] = self.ref_text
        if self.instruct:
            kwargs["instruct"] = self.instruct
        audio = self.model.generate(text=text, **kwargs)
        if isinstance(audio, (list, tuple)):
            if len(audio) != 1:
                raise RuntimeError(f"Expected one waveform for one text, received {len(audio)}")
            audio = audio[0]
        waveform = np.asarray(audio).squeeze()
        if waveform.ndim != 1:
            raise RuntimeError(f"Expected mono waveform, got shape {waveform.shape}")
        yield waveform.astype(np.float32, copy=False)


def create_adapter(**options: Any) -> OmniVoiceAdapter:
    return OmniVoiceAdapter(**options)
