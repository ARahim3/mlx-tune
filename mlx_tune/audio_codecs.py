"""
Audio Codec Adapters for MLX-Tune

Provides a uniform interface for audio codecs (SNAC, DAC, BiCodec, Mimi)
used by TTS models. Each adapter wraps a specific codec implementation
and provides encode/decode plus interleave/deinterleave operations.

Usage:
    from mlx_tune.audio_codecs import create_codec
    from mlx_tune.audio_profiles import TTS_PROFILES

    codec_adapter = create_codec(TTS_PROFILES["orpheus"], snac_model)
    tokens = codec_adapter.encode(audio)
    audio = codec_adapter.decode(tokens)
"""

from typing import List, Protocol, Tuple, Union, runtime_checkable
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

from mlx_tune.audio_profiles import TTSModelProfile


# ---------------------------------------------------------------------------
# Codec Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CodecAdapter(Protocol):
    """
    Protocol for audio codec adapters.

    All TTS codec interactions go through this interface, making it easy
    to add new codecs without changing training code.
    """

    @property
    def sample_rate(self) -> int:
        """Audio sample rate of this codec."""
        ...

    @property
    def num_codebooks(self) -> int:
        """Number of VQ codebook levels."""
        ...

    def encode(self, audio: Union[np.ndarray, "mx.array"], sr: int) -> List[int]:
        """
        Encode audio waveform to a flat list of token IDs.

        Args:
            audio: Audio waveform (1D numpy array or mx.array)
            sr: Sample rate of input audio

        Returns:
            Flat list of interleaved audio token IDs
        """
        ...

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """
        Decode a flat list of token IDs back to audio waveform.

        Args:
            token_ids: Flat interleaved audio token IDs

        Returns:
            Audio waveform as numpy array
        """
        ...

    def interleave(self, codes: List) -> List[int]:
        """
        Interleave hierarchical VQ codes into a flat token sequence.

        Args:
            codes: List of per-level code arrays from the raw codec

        Returns:
            Flat list of interleaved token IDs (with offset applied)
        """
        ...

    def deinterleave(self, token_ids: List[int]) -> List[np.ndarray]:
        """
        De-interleave flat token sequence back to per-level code arrays.

        Args:
            token_ids: Flat interleaved token IDs

        Returns:
            List of per-level code arrays (numpy)
        """
        ...


# ---------------------------------------------------------------------------
# SNAC Codec Adapter (Orpheus)
# ---------------------------------------------------------------------------

class SNACCodecAdapter:
    """
    Adapter for the SNAC audio codec used by Orpheus-TTS.

    Wraps the mlx-audio SNAC model and handles interleaving/deinterleaving
    of hierarchical VQ codes according to the profile's pattern.
    """

    def __init__(self, profile: TTSModelProfile, snac_model):
        self._profile = profile
        self._snac = snac_model

    @property
    def sample_rate(self) -> int:
        return getattr(self._snac, "sampling_rate", self._profile.sample_rate)

    @property
    def num_codebooks(self) -> int:
        return self._profile.num_codebooks

    def encode(self, audio: Union[np.ndarray, "mx.array"], sr: int) -> List[int]:
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio, dtype=mx.float32)

        # Ensure shape [batch, channels, samples]
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            audio = audio.reshape(1, audio.shape[0], audio.shape[1])

        codes = self._snac.encode(audio)
        mx.eval(codes)

        return self.interleave(codes)

    def decode(self, token_ids: List[int]) -> np.ndarray:
        code_arrays = self.deinterleave(token_ids)

        mx_codes = []
        for codes in code_arrays:
            mx_codes.append(mx.array(codes.reshape(1, -1)))

        audio = self._snac.decode(mx_codes)
        mx.eval(audio)

        return np.array(audio).squeeze()

    def interleave(self, codes: List) -> List[int]:
        """Interleave 3-level SNAC codes: [L0, L1, L1, L2, L2, L2, L2] per frame."""
        num_levels = len(codes)
        if num_levels == 0:
            return []

        code_arrays = []
        for c in codes:
            c_np = np.array(c)
            if c_np.ndim > 1:
                c_np = c_np.flatten()
            code_arrays.append(c_np)

        coarsest_len = len(code_arrays[0])

        tokens = []
        for t in range(coarsest_len):
            for level in range(num_levels):
                ratio = len(code_arrays[level]) // coarsest_len
                for sub in range(ratio):
                    idx = t * ratio + sub
                    if idx < len(code_arrays[level]):
                        code_val = int(code_arrays[level][idx])
                        token_id = (
                            code_val
                            + self._profile.audio_token_offset
                            + level * self._profile.codebook_size
                        )
                        tokens.append(token_id)

        return tokens

    def deinterleave(self, token_ids: List[int]) -> List[np.ndarray]:
        pattern = self._profile.interleave_pattern
        tokens_per_frame = sum(pattern)
        num_frames = len(token_ids) // tokens_per_frame

        level_codes = [[] for _ in range(len(pattern))]

        for t in range(num_frames):
            base = t * tokens_per_frame
            if base + tokens_per_frame > len(token_ids):
                break

            offset = 0
            for level, count in enumerate(pattern):
                for i in range(count):
                    code = (
                        token_ids[base + offset + i]
                        - self._profile.audio_token_offset
                        - level * self._profile.codebook_size
                    )
                    level_codes[level].append(code)
                offset += count

        return [np.array(codes, dtype=np.int32) for codes in level_codes]


# ---------------------------------------------------------------------------
# DAC Codec Adapter (OuteTTS)
# ---------------------------------------------------------------------------

class DACCodecAdapter:
    """
    Adapter for the DAC (Descript Audio Codec) used by OuteTTS.

    OuteTTS uses text-based audio tokens (<|c1_X|>, <|c2_X|>) rather than
    numeric offsets. The encode/decode methods work with the DAC model
    and return codebook indices that get formatted as text tokens by the
    data collator.
    """

    def __init__(self, profile: TTSModelProfile, dac_model):
        self._profile = profile
        self._dac = dac_model

    @property
    def sample_rate(self) -> int:
        return self._profile.sample_rate

    @property
    def num_codebooks(self) -> int:
        return self._profile.num_codebooks

    def encode(self, audio: Union[np.ndarray, "mx.array"], sr: int) -> List[int]:
        """Encode audio to flat codebook indices (c1 and c2 interleaved)."""
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio, dtype=mx.float32)

        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)

        codes = self._dac.encode(audio)
        mx.eval(codes)

        # codes is mx.array of shape (batch, num_codebooks, time)
        # Split into per-codebook arrays for interleaving
        codes_np = np.array(codes).squeeze(0)  # (num_codebooks, time)
        per_cb = [codes_np[i] for i in range(codes_np.shape[0])]

        return self.interleave(per_cb)

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """Decode codebook indices back to audio."""
        code_arrays = self.deinterleave(token_ids)
        # Stack codebooks: shape (batch, num_codebooks, time)
        stacked = np.stack(code_arrays, axis=0).reshape(1, self.num_codebooks, -1)
        codes_mx = mx.array(stacked)

        audio = self._dac.decode(codes_mx)
        mx.eval(audio)
        return np.array(audio).squeeze()

    def interleave(self, codes: List) -> List[int]:
        """Flat interleave: [c1_t0, c2_t0, c1_t1, c2_t1, ...]"""
        code_arrays = []
        for c in codes:
            c_np = np.array(c).flatten()
            code_arrays.append(c_np)

        if not code_arrays:
            return []

        min_len = min(len(c) for c in code_arrays)
        tokens = []
        for t in range(min_len):
            for level in range(len(code_arrays)):
                tokens.append(int(code_arrays[level][t]))

        return tokens

    def deinterleave(self, token_ids: List[int]) -> List[np.ndarray]:
        """Split flat sequence back to per-codebook arrays."""
        n = self.num_codebooks
        level_codes = [[] for _ in range(n)]
        for i, tid in enumerate(token_ids):
            level_codes[i % n].append(tid)
        return [np.array(codes, dtype=np.int32) for codes in level_codes]


# ---------------------------------------------------------------------------
# BiCodec Adapter (Spark-TTS)
# ---------------------------------------------------------------------------

class BiCodecAdapter:
    """
    Adapter for the BiCodec used by Spark-TTS.

    BiCodec produces global (speaker) tokens and semantic (content) tokens.
    Like OuteTTS, these are formatted as text tokens by the data collator.
    """

    def __init__(self, profile: TTSModelProfile, bicodec_model):
        self._profile = profile
        self._bicodec = bicodec_model

    @property
    def sample_rate(self) -> int:
        return self._profile.sample_rate

    @property
    def num_codebooks(self) -> int:
        return self._profile.num_codebooks

    def encode(self, audio: Union[np.ndarray, "mx.array"], sr: int) -> List[int]:
        """Encode audio to global + semantic token indices."""
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio, dtype=mx.float32)

        global_tokens, semantic_tokens = self._bicodec.tokenize(audio)
        mx.eval(global_tokens)
        mx.eval(semantic_tokens)

        # Flatten: global_tokens may be (1,1,N), semantic_tokens may be (1,M)
        global_flat = np.array(global_tokens).flatten().tolist()
        semantic_flat = np.array(semantic_tokens).flatten().tolist()

        # Concatenate: global tokens first, then semantic tokens
        return global_flat + semantic_flat

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """Decode global + semantic tokens back to audio."""
        # Split: first num_global tokens are global, rest are semantic
        # Global tokens are fixed-size (typically 32 from BiCodec)
        num_global = 32  # BiCodec global codebook size
        global_tokens = mx.array(token_ids[:num_global]).reshape(1, 1, -1)
        semantic_tokens = mx.array(token_ids[num_global:]).reshape(1, -1)

        audio = self._bicodec.detokenize(global_tokens, semantic_tokens)
        mx.eval(audio)
        return np.array(audio).squeeze()

    def interleave(self, codes: List) -> List[int]:
        """Concatenate global tokens then semantic tokens (not interleaved)."""
        result = []
        for c in codes:
            c_np = np.array(c).flatten()
            result.extend(int(x) for x in c_np)
        return result

    def deinterleave(self, token_ids: List[int]) -> List[np.ndarray]:
        """Split at midpoint (global first half, semantic second half)."""
        mid = len(token_ids) // 2
        return [
            np.array(token_ids[:mid], dtype=np.int32),
            np.array(token_ids[mid:], dtype=np.int32),
        ]


# ---------------------------------------------------------------------------
# Mimi Codec Adapter (Sesame/CSM)
# ---------------------------------------------------------------------------

class MimiCodecAdapter:
    """
    Adapter for the Mimi codec used by Sesame/CSM-1B.

    Mimi uses 32 codebooks with residual vector quantization.
    Tokens are numeric IDs with offset: token + codebook * audio_vocab_size.
    """

    def __init__(self, profile: TTSModelProfile, mimi_model):
        self._profile = profile
        self._mimi = mimi_model

    @property
    def sample_rate(self) -> int:
        return getattr(self._mimi, "sample_rate",
                       getattr(getattr(self._mimi, "cfg", None), "sample_rate",
                               self._profile.sample_rate))

    @property
    def num_codebooks(self) -> int:
        return self._profile.num_codebooks

    def encode(self, audio: Union[np.ndarray, "mx.array"], sr: int) -> List[int]:
        """Encode audio to multi-codebook token sequence."""
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio, dtype=mx.float32)

        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)

        codes = self._mimi.encode(audio)  # shape: (num_codebooks, time)
        mx.eval(codes)

        return self.interleave(codes)

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """Decode multi-codebook tokens back to audio."""
        code_arrays = self.deinterleave(token_ids)
        # Stack to (num_codebooks, time)
        stacked = np.stack(code_arrays, axis=0)
        codes_mx = mx.array(stacked)

        audio = self._mimi.decode(codes_mx)
        mx.eval(audio)
        return np.array(audio).squeeze()

    def interleave(self, codes: List) -> List[int]:
        """Frame-by-frame: all codebook tokens for frame t, then frame t+1, etc."""
        code_arrays = []
        for c in codes:
            c_np = np.array(c).flatten()
            code_arrays.append(c_np)

        if not code_arrays:
            return []

        min_len = min(len(c) for c in code_arrays)
        vocab = self._profile.codebook_size
        tokens = []
        for t in range(min_len):
            for cb in range(len(code_arrays)):
                token_id = int(code_arrays[cb][t]) + cb * vocab
                tokens.append(token_id)

        return tokens

    def deinterleave(self, token_ids: List[int]) -> List[np.ndarray]:
        """Split frame-by-frame tokens back to per-codebook arrays."""
        n = self.num_codebooks
        vocab = self._profile.codebook_size
        level_codes = [[] for _ in range(n)]

        for i, tid in enumerate(token_ids):
            cb = i % n
            code = tid - cb * vocab
            level_codes[cb].append(code)

        return [np.array(codes, dtype=np.int32) for codes in level_codes]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_codec(profile: TTSModelProfile, codec_model) -> "CodecAdapter":
    """
    Create a codec adapter for the given profile.

    Args:
        profile: TTS model profile
        codec_model: Loaded codec model instance (e.g. SNAC, DAC, Mimi)

    Returns:
        A CodecAdapter instance

    Raises:
        ValueError: If the profile's codec_type is not supported
    """
    if profile.codec_type == "snac":
        return SNACCodecAdapter(profile, codec_model)
    elif profile.codec_type == "dac":
        return DACCodecAdapter(profile, codec_model)
    elif profile.codec_type == "bicodec":
        return BiCodecAdapter(profile, codec_model)
    elif profile.codec_type == "mimi":
        return MimiCodecAdapter(profile, codec_model)
    else:
        raise ValueError(
            f"Unsupported codec type: {profile.codec_type}. "
            f"Supported: snac, dac, bicodec, mimi"
        )
