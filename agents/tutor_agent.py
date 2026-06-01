import asyncio
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    llm,
)
from livekit.plugins import silero
from config import settings
from engines.lesson_engine import LessonEngine
from engines.quiz_engine import QuizEngine
from plugins.hf_llm import HuggingFaceLLM
from plugins.stt import FasterWhisperSTT
from plugins.tts import PiperTTS
from router.intent_router import Intent, IntentRouter

logger = logging.getLogger(__name__)


class TutorAgent(Agent):
    """
    helps in generating output results based on intent
    """

    def __init__(
        self,
        *,
        initial_instructions: str,
        router: IntentRouter,
        tts: PiperTTS,
    ) -> None:
        super().__init__(instructions=initial_instructions)
        self._router = router
        self._tts = tts

    async def on_enter(self) -> None:
        greeting = (
            f"Hello! I'm your {settings.TARGET_LANGUAGE} language tutor. "
            "Tell me if you want to learn something new, practice a conversation, "
            "or you want take a quiz, or want to clarify a question?"
        )
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        text = _extract_text(new_message)
        logger.info(f"[AGENT] User said: {text}")
        if not text:
            logger.info("[AGENT] empty msg")
            return
        
        detected_lang = self._router.detect_language(text)
        logger.info(f"[AGENT] Using: {detected_lang} → updating TTS voice")


def _extract_text(message: llm.ChatMessage) -> str:
    """Function to extract message"""
    parts: list[str] = []
    for item in message.content:
        if isinstance(item, str):
            parts.append(item)
        elif hasattr(item, "transcript") and item.transcript:
            parts.append(item.transcript)
        elif hasattr(item, "text") and isinstance(item.text, str):
            parts.append(item.text)
    return " ".join(parts).strip()


async def entrypoint(ctx: JobContext) -> None:
    settings.validate()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session_id = ctx.room.name or ctx.job.id
    logger.info("Session started: %s", session_id)

    session_data = {
        "session_id": session_id,
        "target_lang": settings.TARGET_LANGUAGE,
        "native_lang": settings.NATIVE_LANGUAGE,
        "level": "beginner",
        "mode": "conversation",
    }

    router = IntentRouter()
    lesson_engine = LessonEngine(
        target_language=settings.TARGET_LANGUAGE,
        native_language=settings.NATIVE_LANGUAGE,
    )
    quiz_engine = QuizEngine()

    tts_plugin = PiperTTS(
        language=settings.PIPER_DEFAULT_LANGUAGE,
        models_dir=settings.PIPER_MODELS_DIR,
        piper_executable=settings.PIPER_EXECUTABLE,
    )

    # Greeting uses CONVERSATION intent so the initial system prompt is sensible
    initial_instructions = router.system_prompt_for(
        intent=Intent.CONVERSATION,
        target_language=settings.TARGET_LANGUAGE,
        native_language=settings.NATIVE_LANGUAGE,
        level=session_data.get("level", "beginner"),
    )

    agent = TutorAgent(
        initial_instructions=initial_instructions,
        router=router,
        tts=tts_plugin,
    )

    pipeline = AgentSession(
        vad=silero.VAD.load(),
        stt=FasterWhisperSTT(
            model=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        ),
        llm=HuggingFaceLLM(
            model=settings.HF_LLM_MODEL,
            token=settings.HF_TOKEN,
            router=router,
            lesson_engine=lesson_engine,
            quiz_engine=quiz_engine,
            session_data=session_data,
            target_language=settings.TARGET_LANGUAGE,
            native_language=settings.NATIVE_LANGUAGE,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        ),
        tts=tts_plugin,
    )

    await pipeline.start(agent=agent, room=ctx.room)
    await asyncio.sleep(float("inf"))
