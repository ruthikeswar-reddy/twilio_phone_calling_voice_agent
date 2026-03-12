"""
Audio utilities for PCM ↔ μ-law conversion.

PYTHON 3.13 FIX:
  audioop was removed from stdlib in Python 3.13.
  audioop-lts (pip install audioop-lts) is a drop-in replacement.
  This file handles both cases transparently.
"""

try:
    import audioop
except ImportError:
    # Python 3.13+ — audioop removed from stdlib
    import audioop_lts as audioop  # pip install audioop-lts


def pcm_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Convert PCM 16-bit signed little-endian → μ-law 8-bit."""
    return audioop.lin2ulaw(pcm_bytes, 2)


def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
    """Convert μ-law 8-bit → PCM 16-bit signed little-endian."""
    return audioop.ulaw2lin(ulaw_bytes, 2)


def resample_pcm(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM 16-bit audio from one sample rate to another."""
    if from_rate == to_rate:
        return pcm_bytes
    resampled, _ = audioop.ratecv(pcm_bytes, 2, 1, from_rate, to_rate, None)
    return resampled


def normalize_audio_level(pcm_bytes: bytes, target_rms: float = 3000.0) -> bytes:
    """Normalize PCM audio to target RMS level."""
    rms = audioop.rms(pcm_bytes, 2)
    if rms == 0:
        return pcm_bytes
    factor = min(target_rms / rms, 3.0)
    return audioop.mul(pcm_bytes, 2, factor)
