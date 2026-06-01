import asyncio
import concurrent.futures
import logging
from math import gcd

import numpy as np
from faster_whisper import WhisperModel
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class FasterWhisperSTT(stt.STT):
    """
    Using STT using faster-whisper which auto-detects spoken language
    """

    def __init__(
        self,
        *,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._default_language = language
        self._beam_size = beam_size
        logger.info("Loading faster-whisper model '%s' on %s (%s)", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = (language if language is not NOT_GIVEN else self._default_language) or None
        audio_np = self._frames_to_float32(buffer)

        loop = asyncio.get_event_loop()
        text, detected_lang = await loop.run_in_executor(
            _executor, self._transcribe_sync, audio_np, lang
        )
        logger.info("STT [%s]: %s", detected_lang, text)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=detected_lang,
                    confidence=1.0,
                )
            ],
        )

    def _transcribe_sync(self, audio: np.ndarray, language: str | None) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=language,
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language

    def _frames_to_float32(self, frames: AudioBuffer) -> np.ndarray:
        if not isinstance(frames, (list, tuple)):
            frames = [frames]

        if not frames:
            return np.zeros(0, dtype=np.float32)

        all_data = b"".join(bytes(f.data) for f in frames)
        sample_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels

        audio = np.frombuffer(all_data, dtype=np.int16).astype(np.float32)

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        if sample_rate != 16000:
            from scipy.signal import resample_poly
            g = gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)

        return audio / 32768.0
