import asyncio
import logging
from typing import Any

from huggingface_hub import InferenceClient
from livekit.agents import llm
from livekit.agents.llm import Tool
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from engines.lesson_engine import LessonEngine
from engines.quiz_engine import QuizEngine
from router.intent_router import Intent, IntentRouter

logger = logging.getLogger(__name__)


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    """Returns recent user message from the chat context"""
    for msg in reversed(chat_ctx.messages()):
        if msg.role == "user":
            content = (msg.text_content or "").strip()
            if content:
                return content
    return ""


def _build_system_prompt(
    router: IntentRouter,
    lesson_engine: LessonEngine,
    quiz_engine: QuizEngine,
    user_text: str,
    target_language: str,
    native_language: str,
    level: str,
) -> tuple[str, Intent, str]:
    """
   returns system_prompt, intent, detected_lang
    """
    intent, detected_lang = router.route(user_text)

    base = router.system_prompt_for(
        intent=intent,
        target_language=target_language,
        native_language=native_language,
        level=level,
    )

    if intent == Intent.TEACHING:
        topic = lesson_engine.next_topic(level)
        if topic:
            engine_ctx = lesson_engine.build_lesson_prompt(topic, level)
            lesson_engine.advance(level)
            system_prompt = f"{base} {engine_ctx}"
        else:
            system_prompt = base

    elif intent == Intent.QUIZ:
        quiz_ctx = quiz_engine.build_quiz_prompt(level)
        system_prompt = f"{base} {quiz_ctx}"

    else:
        system_prompt = base

    return system_prompt, intent, detected_lang


class HuggingFaceLLM(llm.LLM):
    """
    This class perform following steps
      1. Extracts the latest user utterance from chat context
      2. Routes intent via IntentRouter
      3. Invokes LessonEngine / QuizEngine for curriculum context
      4. Injects an engine-built system message
      5. Calls the HF Inference API (sync client, thread-offloaded) with retry
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        token: str,
        router: IntentRouter,
        lesson_engine: LessonEngine,
        quiz_engine: QuizEngine,
        session_data: dict,
        target_language: str = "Spanish",
        native_language: str = "English",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._router = router
        self._lesson_engine = lesson_engine
        self._quiz_engine = quiz_engine
        self._session_data = session_data
        self._target_language = target_language
        self._native_language = native_language
        self._client = InferenceClient(api_key=token)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "huggingface"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        return HuggingFaceLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            client=self._client,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            router=self._router,
            lesson_engine=self._lesson_engine,
            quiz_engine=self._quiz_engine,
            session_data=self._session_data,
            target_language=self._target_language,
            native_language=self._native_language,
        )


class HuggingFaceLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: HuggingFaceLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        client: InferenceClient,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        base_delay: float,
        router: IntentRouter,
        lesson_engine: LessonEngine,
        quiz_engine: QuizEngine,
        session_data: dict,
        target_language: str,
        native_language: str,
    ):
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._router = router
        self._lesson_engine = lesson_engine
        self._quiz_engine = quiz_engine
        self._session_data = session_data
        self._target_language = target_language
        self._native_language = native_language

    async def _run(self) -> None:
        user_text = _last_user_text(self._chat_ctx)
        logger.debug("User text: %r", user_text)

        # intent routing 
        level = self._session_data.get("level", "beginner")
        system_prompt, intent, detected_lang = _build_system_prompt(
            router=self._router,
            lesson_engine=self._lesson_engine,
            quiz_engine=self._quiz_engine,
            user_text=user_text,
            target_language=self._target_language,
            native_language=self._native_language,
            level=level,
        )

        logger.info("Intent=%s | Lang=%s | Level=%s", intent.value, detected_lang, level)

        # building input message with previous history
        history = [
            {"role": msg.role, "content": msg.text_content or ""}
            for msg in self._chat_ctx.messages()
            if msg.role != "system" and msg.text_content
        ]
        messages = [{"role": "system", "content": system_prompt}] + history

        # calling hugging face Inference API w
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from model")

                logger.info(
                    "LLM [%s/%s]: %s",
                    intent.value, detected_lang, content
                )

                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=str(23),
                        delta=llm.ChoiceDelta(role="assistant", content=content),
                    )
                )
                return

            except Exception as e:
                logger.warning("LLM attempt %d/%d failed: %s", attempt, self._max_retries, e)
                if attempt < self._max_retries:
                    await asyncio.sleep(self._base_delay * (2 ** (attempt - 1)))

        raise RuntimeError(f"HuggingFace LLM failed after {self._max_retries} attempts")