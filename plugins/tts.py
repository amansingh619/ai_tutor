import asyncio
import logging
import os
import re
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger(__name__)

LIVEKIT_SAMPLE_RATE = 24000

LANGUAGE_MODELS: dict[str, str] = {
    "en": "en_US-lessac-medium.onnx",
    "es": "es_MX-claude-high.onnx",
}
DEFAULT_LANGUAGE = "en"


_SPANISH_WORDS = re.compile(
    r"\b(hola|adiós|gracias|buenos|buenas|cómo|qué|tengas|estás|día|hablar|soy|tengo)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"'([^']+)'")
_LANG_TAG_RE = re.compile(r"\[lang=[a-z]{2}\]")  # strip any LLM-injected tags


def _detect_lang(text: str) -> str:
    if _SPANISH_WORDS.search(text):
        return "es"
    return "en"


def _split_by_language(text: str) -> list[tuple[str, str]]:
    """
    Split full TTS input into (segment, lang_code) pairs.
    Strips any [lang=xx] tags, then splits on quoted phrases,
    detecting language per segment.
    """
    # Strip any lang tags injected upstream
    text = _LANG_TAG_RE.sub("", text).strip()

    segments: list[tuple[str, str]] = []
    last = 0

    for match in _QUOTED_RE.finditer(text):
        before = text[last:match.start()]
        if before.strip():
            segments.append((before, _detect_lang(before)))

        quoted = match.group(1)
        segments.append((quoted, _detect_lang(quoted)))
        last = match.end()

    tail = text[last:]
    if tail.strip():
        segments.append((tail, _detect_lang(tail)))

    if not segments:
        segments.append((text, _detect_lang(text)))

    return segments


class PiperTTS(tts.TTS):
    """
    Automatically splits mixed-language text and routes each segment
    to the correct ONNX model.
    """

    def __init__(
        self,
        *,
        language: str = DEFAULT_LANGUAGE,
        models_dir: str = "models",
        piper_executable: str = "piper",
        sample_rate: int = LIVEKIT_SAMPLE_RATE,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._language = language
        self._models_dir = models_dir
        self._piper_executable = piper_executable

    @property
    def language(self) -> str:
        return self._language

    @property
    def provider(self) -> str:
        return "piper"

    def _model_path(self, lang: str) -> str:
        model_file = LANGUAGE_MODELS.get(lang[:2].lower(), LANGUAGE_MODELS[DEFAULT_LANGUAGE])
        return os.path.join(self._models_dir, model_file)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return PiperTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            piper_executable=self._piper_executable,
            models_dir=self._models_dir,
        )


class PiperTTSStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        piper_executable: str,
        models_dir: str,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._piper_executable = piper_executable
        self._models_dir = models_dir

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )

        segments = _split_by_language(self._input_text)
        logger.debug("TTS segments: %s", [(lang, t[:40]) for t, lang in segments])

        all_pcm: list[bytes] = []

        for segment_text, lang in segments:
            if not segment_text.strip():
                continue

            model_file = LANGUAGE_MODELS.get(lang, LANGUAGE_MODELS[DEFAULT_LANGUAGE])
            model_path = os.path.join(self._models_dir, model_file)

            logger.debug("Synthesizing [%s] %r with %s", lang, segment_text[:40], model_file)
            pcm = await self._synthesize_segment(segment_text, model_path)
            all_pcm.append(pcm)

        if all_pcm:
            output_emitter.push(b"".join(all_pcm))

        output_emitter.flush()

    async def _synthesize_segment(self, text: str, model_path: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                self._piper_executable,
                "-m", model_path,
                "-f", out_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(input=text.encode("utf-8"))

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Piper TTS failed (exit {proc.returncode}): {stderr.decode()}"
                )

            return self._wav_to_pcm16(out_path, self._tts.sample_rate)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def _wav_to_pcm16(self, wav_path: str, target_sr: int) -> bytes:
        audio, src_sr = sf.read(wav_path, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if src_sr != target_sr:
            g = gcd(src_sr, target_sr)
            audio = resample_poly(audio, target_sr // g, src_sr // g).astype(np.float32)

        return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()